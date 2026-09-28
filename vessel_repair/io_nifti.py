"""Binary NIfTI I/O with spatial transforms and header metadata preserved."""
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def validate_affine(affine: NDArray) -> NDArray[np.float64]:
    """Validate a finite, invertible 3D voxel-to-world affine."""
    affine = np.asarray(affine, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or np.linalg.matrix_rank(affine[:3, :3]) != 3):
        raise ValueError("Expected a finite, invertible 4x4 spatial affine")
    return affine


def affine_in_mm(image: nib.spatialimages.SpatialImage) -> NDArray[np.float64]:
    """Return geometry in mm; unknown NIfTI spatial units are assumed mm."""
    unit = image.header.get_xyzt_units()[0]
    factor = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}[unit]
    affine = validate_affine(image.affine).copy()
    affine[:3] *= factor
    return affine


def load_binary(path: str | Path) -> tuple[NDArray[np.bool_], nib.spatialimages.SpatialImage]:
    """Load a 3D mask, binarize values >0, and retain its reference image."""
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {image.shape}")
    validate_affine(image.affine)
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or infinite values")
    return data > 0, image


def save_binary(mask: NDArray, reference: nib.spatialimages.SpatialImage,
                path: str | Path) -> None:
    """Save uint8 0/1 with reference affine, header, qform and sform codes."""
    if mask.shape != reference.shape:
        raise ValueError("Output shape must match the reference image")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"], header["cal_max"] = 0, 1
    image = reference.__class__((mask > 0).astype(np.uint8), reference.affine, header)
    # Preserve the stored transforms, including an unset transform (code zero).
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
