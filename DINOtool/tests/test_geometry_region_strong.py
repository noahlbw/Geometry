import unittest
from unittest.mock import patch

import torch

from dinotool.geometry_region_strong import (
    FrozenStrongRegionObserver, StrongRegionConfig,
    footprint_region_crops, project_patch_support)
from dinotool.region_semantic_readout import FrozenRegionObserver


class StrongRegionTests(unittest.TestCase):
    def test_actual_patch_grid_and_uniform_footprint(self):
        values = project_patch_support(torch.ones(2, 32, 32), image_shape=(512, 512))
        self.assertEqual(values.shape, (2, 27*27))
        torch.testing.assert_close(values, torch.ones_like(values))

    def test_unseen_six_pixel_border_is_not_stretched_into_tokens(self):
        support = torch.zeros(1, 384, 384)
        support[:, 378:, :] = 1
        values = project_patch_support(support, image_shape=(384, 384))
        self.assertTrue(bool((values == 0).all()))

    def test_actual_first_patch_footprint(self):
        support = torch.zeros(1, 384, 384)
        support[:, :14, :14] = 1
        values = project_patch_support(support, image_shape=(384, 384)).reshape(27, 27)
        expected = torch.zeros_like(values); expected[0, 0] = 1
        torch.testing.assert_close(values, expected, atol=2e-6, rtol=0)

    def test_crop_padding_has_zero_evidence_mass(self):
        values = project_patch_support(torch.ones(1, 32, 32), image_shape=(512, 512),
                                       top=-64, left=-64, extent=64)
        self.assertTrue(bool((values == 0).all()))

    def test_small_corner_supports_remain_readable(self):
        masks = torch.zeros(3, 32, 32, dtype=torch.bool)
        for index, position in enumerate((0, 15, 31)):
            masks[index, position, position] = True
        crops, weights = footprint_region_crops(torch.rand(1, 3, 512, 512),
            masks.flatten(1).float(), masks, StrongRegionConfig())
        self.assertEqual(crops.shape, (3, 3, 384, 384))
        self.assertEqual(weights.shape, (3, 729))
        self.assertTrue(bool((weights.sum(-1) > 0).all()))

    def test_new_primary_mapping_does_not_mutate_base_method_names(self):
        observer = object.__new__(FrozenStrongRegionObserver)
        output = {"P": {"Geometry": torch.zeros(1), "RegionJoint": torch.ones(1)}}
        with patch.object(FrozenRegionObserver, "__call__", return_value=(output, {})):
            scores, _ = observer()
        self.assertIn("Geometry_RegionStrong", scores["P"])
        self.assertNotIn("RegionJoint", scores["P"])
        self.assertEqual(FrozenRegionObserver.encoder_size, 256)

    def test_invalid_patch_geometry_rejected(self):
        with self.assertRaises(ValueError):
            project_patch_support(torch.ones(1, 4, 4), image_shape=(512, 512), patch_size=0)


if __name__ == "__main__":
    unittest.main()
