"""Regression tests for physical geometry, growth limits, and NIfTI preservation."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from vessel_repair.bridges import rasterize_tube
from vessel_repair.components import filter_distant_islands
from vessel_repair.demo import synthetic_mask
from vessel_repair.io_nifti import affine_in_mm, load_binary, save_binary
from vessel_repair.repair import RepairParameters, repair_mask


class RepairTests(unittest.TestCase):
    def test_demo_and_no_mutation(self):
        mask = synthetic_mask()
        saved = mask.copy()
        result = repair_mask(mask, np.eye(4), RepairParameters(cleanup=True, min_component_size=10))
        self.assertEqual((result.before_components, result.after_components), (3, 1))
        self.assertEqual(result.voxels_removed, 8)
        self.assertLessEqual(result.voxels_added, result.original_foreground * 0.05)
        np.testing.assert_array_equal(mask, saved)

    def test_cumulative_budget(self):
        mask = np.zeros((20, 3, 3), dtype=bool)
        mask[[1, 2, 4, 5, 7, 8], 1, 1] = True
        result = repair_mask(mask, np.eye(4), RepairParameters(tube_radius_mm=0.1, max_added_fg_fraction=0.2))
        self.assertEqual(result.voxels_added, 1)
        self.assertEqual(result.after_components, 2)
        zero = repair_mask(mask, np.eye(4), RepairParameters(max_added_fg_fraction=0))
        np.testing.assert_array_equal(zero.mask, mask)

    def test_physical_gap_and_island_filter(self):
        mask = np.zeros((12, 8, 8), dtype=bool)
        mask[1:3, 2, 2] = True
        mask[5, 2, 2] = True
        affine = np.diag([3.0, 1, 1, 1])
        result = repair_mask(mask, affine, RepairParameters(max_gap_mm=8, max_added_fg_fraction=10))
        self.assertEqual(result.after_components, 2)
        self.assertEqual(np.count_nonzero(filter_distant_islands(mask, affine, 8)), 2)
        self.assertEqual(np.count_nonzero(filter_distant_islands(mask, affine, 9)), 3)

    def test_capsule_in_sheared_geometry(self):
        affine = np.array([[0, -2, 0.4, 40], [1, 0, 0, -8], [0, 0, 3, 12], [0, 0, 0, 1.]])
        start, end = (2, 3, 3), (7, 3, 3)
        coords = rasterize_tube((10, 8, 8), affine, start, end, 1.1)
        self.assertTrue(np.all(coords[:, 1] == 3))
        self.assertTrue(np.all(coords[:, 2] == 3))
        self.assertTrue(any(np.all(row == start) for row in coords))
        self.assertTrue(any(np.all(row == end) for row in coords))

    def test_empty_connected_and_26_connectivity(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        self.assertEqual(repair_mask(mask, np.eye(4)).after_components, 0)
        mask[1, 1, 1] = mask[2, 2, 2] = True
        result = repair_mask(mask, np.eye(4))
        self.assertEqual(result.before_components, 1)
        self.assertEqual(result.voxels_added, 0)
        with self.assertRaises(ValueError):
            repair_mask(mask, np.eye(4), RepairParameters(tube_radius_mm=-1))

    def test_nifti_roundtrip_and_units(self):
        affine = np.array([[0, -2, 0.1, 40], [1, 0, 0, -8], [0, 0, 3, 12], [0, 0, 0, 1.]])
        mask = synthetic_mask(False)
        reference = nib.Nifti1Image(mask.astype(np.uint8), affine)
        reference.set_qform(np.diag([1., 2, 3, 1]), 1)
        reference.set_sform(affine, 2)
        reference.header.set_xyzt_units("micron")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.nii.gz"
            save_binary(mask, reference, path)
            loaded, image = load_binary(path)
            np.testing.assert_array_equal(loaded, mask)
            np.testing.assert_allclose(image.affine, reference.affine)
            np.testing.assert_allclose(image.get_qform(), reference.get_qform())
            self.assertEqual(int(image.header["sform_code"]), 2)
            self.assertEqual(int(image.header["qform_code"]), 1)
            self.assertEqual(image.get_data_dtype(), np.dtype("uint8"))
            np.testing.assert_allclose(affine_in_mm(image)[:3], image.affine[:3] * 0.001)


if __name__ == "__main__":
    unittest.main()
