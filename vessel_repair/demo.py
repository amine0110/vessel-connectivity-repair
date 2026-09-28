"""Deterministic synthetic broken vessel and end-to-end demonstration."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from .io_nifti import affine_in_mm, load_binary, save_binary
from .repair import RepairParameters, repair_mask

DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "demo_out"


def synthetic_mask(include_island: bool = True) -> NDArray[np.bool_]:
    """Create two collinear radius-4 tubes separated by three empty voxel planes."""
    x, y, z = np.ogrid[:64, :64, :64]
    cross_section = (y - 32) ** 2 + (z - 32) ** 2 <= 4 ** 2
    mask = cross_section & (((x >= 8) & (x <= 29)) | ((x >= 33) & (x <= 55)))
    if include_island:
        mask[5:7, 5:7, 5:7] = True
    return mask


def run_demo(outdir: str | Path = DEFAULT_OUTDIR,
             parameters: RepairParameters | None = None) -> dict:
    """Write, reload, repair and validate the demo, raising on failed assertions."""
    outdir = Path(outdir)
    p = parameters or RepairParameters(cleanup=True, min_component_size=10)
    broken_path, repaired_path = outdir / "broken.nii.gz", outdir / "repaired.nii.gz"
    mask = synthetic_mask()
    reference = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
    reference.header.set_xyzt_units("mm")
    save_binary(mask, reference, broken_path)
    loaded, reference = load_binary(broken_path)
    result = repair_mask(loaded, affine_in_mm(reference), p)
    save_binary(result.mask, reference, repaired_path)
    saved, saved_reference = load_binary(repaired_path)
    assert result.after_components < result.before_components, result.summary()
    assert np.array_equal(saved, result.mask)
    assert np.array_equal(saved_reference.affine, reference.affine)
    assert result.voxels_added <= p.max_added_fg_fraction * result.original_foreground + 1e-9
    if p.cleanup and p.min_component_size == 10 and p.max_gap_mm == 20:
        assert result.after_components == 1, result.summary()
    summary = {"input_path": str(broken_path.resolve()),
               "output_path": str(repaired_path.resolve()), **result.summary()}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
