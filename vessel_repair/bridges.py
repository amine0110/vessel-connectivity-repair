"""Rasterization of straight physical-space tubes, including anisotropic images."""
import numpy as np
from numpy.typing import NDArray

from .components import voxel_to_world
from .io_nifti import validate_affine
from .prior import BridgeCandidate, propose_bridges

__all__ = ["BridgeCandidate", "propose_bridges", "rasterize_tube"]


def rasterize_tube(shape: tuple[int, ...], affine: NDArray, start: tuple[int, ...],
                   end: tuple[int, ...], radius_mm: float) -> NDArray[np.int64]:
    """Return unique voxel indices for a capsule of the requested physical radius.

    Voxel centers within radius of the segment are included. A sampled digital
    centerline ensures 26-connectivity when radius is below the voxel spacing;
    this core may extend outside the analytic radius by half a voxel diagonal.
    The full affine handles anisotropy, rotations, reflections and shear.
    """
    affine = validate_affine(affine)
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Tube radius must be finite and positive")
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    padding = radius_mm * np.linalg.norm(np.linalg.inv(affine[:3, :3]), axis=1)
    lo = np.maximum(0, np.floor(np.minimum(a, b) - padding).astype(int))
    hi = np.minimum(shape, np.ceil(np.maximum(a, b) + padding).astype(int) + 1)
    wa, wb = voxel_to_world(np.array([a, b]), affine)
    delta = wb - wa
    length_squared = float(delta @ delta)
    chunks = []
    # Slice the bounding box to avoid large temporary coordinate arrays.
    for x in range(lo[0], hi[0]):
        yz = np.indices(tuple(hi[1:] - lo[1:])).reshape(2, -1).T + lo[1:]
        coords = np.column_stack((np.full(len(yz), x), yz))
        world = voxel_to_world(coords, affine)
        t = np.clip((world - wa) @ delta / length_squared, 0, 1) if length_squared else 0
        closest = wa + np.asarray(t)[..., None] * delta
        inside = np.sum((world - closest) ** 2, axis=1) <= radius_mm ** 2 + 1e-10
        chunks.append(coords[inside])
    steps = max(1, int(np.ceil(np.max(np.abs(b - a)) * 4)))
    core = np.rint(np.linspace(a, b, steps + 1)).astype(int)
    core = core[np.all((core >= 0) & (core < np.asarray(shape)), axis=1)]
    chunks.append(core)
    return np.unique(np.concatenate(chunks, axis=0), axis=0)
