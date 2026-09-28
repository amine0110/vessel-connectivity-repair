"""Greedy connectivity repair with a cumulative original-foreground growth gate."""
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .bridges import rasterize_tube
from .components import filter_distant_islands, label_components, remove_small_components
from .io_nifti import validate_affine
from .prior import propose_bridges


@dataclass(frozen=True)
class RepairParameters:
    """Physical bridge limits and optional component cleanup settings."""
    tube_radius_mm: float = 1.0
    max_added_fg_fraction: float = 0.05
    min_component_size: int = 1
    max_gap_mm: float = 20.0
    filter_island_mm: float = 0.0
    cleanup: bool = False
    connectivity: int = 26

    def validate(self) -> None:
        """Reject invalid or unsupported parameter values."""
        for name in ("tube_radius_mm", "max_added_fg_fraction", "max_gap_mm", "filter_island_mm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.tube_radius_mm == 0:
            raise ValueError("tube_radius_mm must be positive")
        if not isinstance(self.min_component_size, int) or self.min_component_size < 1:
            raise ValueError("min_component_size must be an integer >= 1")
        if self.connectivity != 26:
            raise ValueError("Only 26-connectivity is supported")


@dataclass
class RepairResult:
    """Repaired mask and statistics measured against the original input."""
    mask: NDArray[np.bool_]
    before_components: int
    after_components: int
    original_foreground: int
    voxels_added: int
    voxels_removed: int
    accepted_bridges: list[dict]
    rejected_growth: int
    rejected_merge: int
    parameters: RepairParameters

    def summary(self) -> dict:
        """Return JSON-compatible diagnostics, excluding the voxel array."""
        return {"before_components": self.before_components,
                "after_components": self.after_components,
                "original_foreground": self.original_foreground,
                "voxels_added": self.voxels_added, "voxels_removed": self.voxels_removed,
                "added_fg_fraction": self.voxels_added / max(1, self.original_foreground),
                "accepted_bridges": self.accepted_bridges,
                "rejected_growth": self.rejected_growth, "rejected_merge": self.rejected_merge,
                "parameters": asdict(self.parameters)}


def repair_mask(mask: NDArray, affine: NDArray,
                parameters: RepairParameters | None = None) -> RepairResult:
    """Repair a 3D binary mask using geometry in mm without modifying the input.

    Each bridge must join its two target labels and reduce component count.
    Cumulative added voxels (relative to original input) cannot exceed the growth
    budget. After acceptance candidates are recomputed against current geometry.
    """
    p = parameters or RepairParameters()
    p.validate()
    affine = validate_affine(affine)
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("Expected a finite 3D mask")
    original = mask > 0
    original_count = int(np.count_nonzero(original))
    if original_count == 0:
        return RepairResult(original.copy(), 0, 0, 0, 0, 0, [], 0, 0, p)
    # Straight segments stay within foreground bounds; pad for tube radius.
    extent = ndimage.find_objects(original.astype(np.uint8), max_label=1)[0]
    pad = np.ceil(p.tube_radius_mm * np.linalg.norm(np.linalg.inv(affine[:3, :3]), axis=1)).astype(int) + 1
    crop = tuple(slice(max(0, s.start - int(d)), min(n, s.stop + int(d)))
                 for s, d, n in zip(extent, pad, original.shape))
    offset = np.array([s.start for s in crop])
    local_affine = affine.copy()
    local_affine[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]
    baseline = original[crop]
    current = baseline.copy()
    labels, sizes = label_components(current)
    before = len(sizes) - 1
    current = filter_distant_islands(current, local_affine, p.filter_island_mm)
    labels, sizes = label_components(current)
    accepted = []
    rejected_growth = rejected_merge = 0
    added_count = 0
    budget = p.max_added_fg_fraction * original_count
    while len(sizes) > 2:
        candidates = propose_bridges(labels, sizes, local_affine, p.max_gap_mm)
        changed = False
        for candidate in candidates:
            coords = rasterize_tube(current.shape, local_affine, candidate.start,
                                    candidate.end, p.tube_radius_mm)
            indices = tuple(coords.T)
            new_coords = coords[~current[indices]]
            new_indices = tuple(new_coords.T)
            added = int(np.count_nonzero(~baseline[new_indices]))
            if added_count + added > budget + 1e-9:
                rejected_growth += 1
                continue
            current[new_indices] = True
            trial_labels, trial_sizes = label_components(current)
            merged = (trial_labels[candidate.start] == trial_labels[candidate.end]
                      and trial_labels[candidate.start] != 0
                      and len(trial_sizes) < len(sizes))
            if not merged:
                current[new_indices] = False
                rejected_merge += 1
                continue
            accepted.append({"start_voxel": (np.array(candidate.start) + offset).tolist(),
                             "end_voxel": (np.array(candidate.end) + offset).tolist(),
                             "gap_mm": candidate.gap_mm, "new_foreground_voxels": added})
            added_count += added
            labels, sizes = trial_labels, trial_sizes
            changed = True
            break
        if not changed:
            break
    if p.cleanup:
        current = remove_small_components(current, p.min_component_size)
    _, sizes = label_components(current)
    output = original.copy()
    output[crop] = current
    return RepairResult(output, before, len(sizes) - 1, original_count,
                        int(np.count_nonzero(current & ~baseline)),
                        int(np.count_nonzero(baseline & ~current)), accepted,
                        rejected_growth, rejected_merge, p)
