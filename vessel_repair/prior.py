"""Distance-transform boundary prior; no learned mesh, GCN, or FOMAML."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .components import boundary_points, voxel_to_world


@dataclass(frozen=True)
class BridgeCandidate:
    """A nearest-boundary straight path between two current component labels."""
    component_a: int
    component_b: int
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    gap_mm: float


def propose_bridges(labels: NDArray, sizes: NDArray, affine: NDArray,
                    max_gap_mm: float) -> list[BridgeCandidate]:
    """Propose nearest boundary pairs, ordered by physical center-to-center gap.

    World-space bounding boxes prune impossible pairs. KD trees give exact
    nearest boundary pairs. Ties are deterministic for a fixed SciPy version.
    """
    points = boundary_points(labels, sizes)
    world = {key: voxel_to_world(value, affine) for key, value in points.items()}
    trees = {key: cKDTree(value) for key, value in world.items()}
    bounds = {key: (value.min(axis=0), value.max(axis=0)) for key, value in world.items()}
    candidates = []
    keys = sorted(points)
    for index, a in enumerate(keys):
        for b in keys[index + 1:]:
            lo_a, hi_a = bounds[a]
            lo_b, hi_b = bounds[b]
            separation = np.maximum(0, np.maximum(lo_a - hi_b, lo_b - hi_a))
            if np.linalg.norm(separation) > max_gap_mm:
                continue
            distances, nearest = trees[b].query(world[a], workers=1)
            i = int(np.argmin(distances))
            if distances[i] <= max_gap_mm:
                candidates.append(BridgeCandidate(
                    a, b, tuple(int(v) for v in points[a][i]),
                    tuple(int(v) for v in points[b][nearest[i]]), float(distances[i])))
    return sorted(candidates, key=lambda c: (c.gap_mm, c.component_a, c.component_b,
                                             c.start, c.end))
