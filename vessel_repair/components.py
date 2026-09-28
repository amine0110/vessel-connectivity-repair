"""26-connected components and optional physical-distance island filtering."""
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def label_components(mask: NDArray) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Return 26-connected labels and sizes indexed by label (background size 0)."""
    labels, count = ndimage.label(mask, structure=STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels, sizes


def voxel_to_world(points: NDArray, affine: NDArray) -> NDArray[np.float64]:
    """Transform an N x 3 array of voxel centers into physical coordinates."""
    return np.asarray(points) @ affine[:3, :3].T + affine[:3, 3]


def boundary_points(labels: NDArray, sizes: NDArray) -> dict[int, NDArray]:
    """Find each component's one-voxel boundary using a chessboard distance transform.

    The transform only selects boundary voxels; all distances between components
    are measured with the full physical affine, never with chessboard distance.
    """
    result = {}
    for component, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None or sizes[component] == 0:
            continue
        local = np.pad(labels[slices] == component, 1)
        distance = ndimage.distance_transform_cdt(local, metric="chessboard")
        points = np.argwhere(distance == 1) - 1
        offset = np.array([axis.start for axis in slices])
        result[component] = points + offset
    return result


def filter_distant_islands(mask: NDArray, affine: NDArray, threshold_mm: float) -> NDArray:
    """Remove whole components whose nearest boundary is farther from the largest.

    A zero threshold disables filtering. Distances are between voxel centers.
    """
    if not np.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError("Island threshold must be finite and nonnegative")
    if threshold_mm == 0:
        return mask.copy()
    labels, sizes = label_components(mask)
    if len(sizes) <= 2:
        return mask.copy()
    points = boundary_points(labels, sizes)
    main = int(np.argmax(sizes))
    tree = cKDTree(voxel_to_world(points[main], affine))
    keep = np.ones(len(sizes), dtype=bool)
    keep[0] = False
    for component, boundary in points.items():
        if component != main:
            distances, _ = tree.query(voxel_to_world(boundary, affine), workers=1)
            keep[component] = float(distances.min()) <= threshold_mm
    return keep[labels]


def remove_small_components(mask: NDArray, minimum_size: int) -> NDArray:
    """Remove whole components smaller than minimum_size voxels."""
    labels, sizes = label_components(mask)
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]
