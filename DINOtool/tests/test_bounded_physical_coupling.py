from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dinotool.bounded_physical_coupling import geometry_windows, predict_image, resize_geometry
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator


class BoundedPhysicalTests(unittest.TestCase):
    def test_four_crop_cap_and_no_native_layout(self):
        for shape in ((896, 896), (672, 896), (473, 896), (7, 896)):
            windows = geometry_windows(*shape)
            coverage = np.zeros(shape, dtype=int)
            for top, left in windows:
                coverage[top:top + 512, left:left + 512] += 1
            self.assertLessEqual(len(windows), 4)
            self.assertTrue((coverage > 0).all())
        with self.assertRaises(ValueError):
            geometry_windows(3000, 4000)

    def test_uint8_resize_and_aspect_ratio(self):
        image = torch.arange(3 * 13 * 29).reshape(3, 13, 29).float() / (3 * 13 * 29)
        actual = resize_geometry(image)
        array = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        expected = cv2.resize(array, (896, int(13 * 896 / 29 + .5)), interpolation=cv2.INTER_LINEAR)
        self.assertTrue(torch.equal(actual, torch.from_numpy(expected).permute(2, 0, 1).float() / 255))

    def test_restore_probabilities_before_argmax(self):
        accumulator = DeviceProbabilityAccumulator(2, 2, 3, 'cpu')
        probabilities = torch.tensor([[[.9, .2, .6], [.1, .8, .4]], [[.1, .8, .4], [.9, .2, .6]]])
        accumulator.add(probabilities, torch.ones(2, 3), 0, 0)
        expected = F.interpolate(probabilities[None], (7, 11), mode='bilinear', align_corners=False)[0].argmax(0)
        self.assertTrue(np.array_equal(accumulator.finalize_resized((7, 11)), expected.numpy()))
        historical = accumulator.finalize_outputs(None, [0., 0.], False)[0]
        self.assertTrue(np.array_equal(historical, probabilities.argmax(0).numpy()))

    def test_actual_4k_branch_budget_and_original_wide_source(self):
        class Branch:
            device = 'cpu'
            def prepare_image(self, image):
                return None
            def crop_patch_features(self, image):
                return None

        original = torch.zeros(3, 3000, 4000)
        def fake(resized, geometry, banks, vip, queries, work, **kwargs):
            self.assertIs(kwargs['wide_image'], original)
            self.assertEqual(kwargs['output_size'], (3000, 4000))
            self.assertEqual(tuple(resized.shape[-2:]), (672, 896))
            for _ in geometry_windows(*resized.shape[-2:]):
                geometry.prepare_image(torch.zeros(1, 3, 512, 512))
            for _ in range(2):
                vip.crop_patch_features(torch.zeros(3, 336, 336))
            return {'test': {'SharedRivalSoft': np.zeros((3000, 4000), np.uint8)}}, {'test': {'tiles': 4}}
        _, diagnostic = predict_image(original, Branch(), {}, Branch(), {}, local_predictor=fake)
        self.assertEqual(diagnostic['test']['geometry_encodings'], 4)
        self.assertEqual(diagnostic['test']['wide_encodings'], 2)
        self.assertEqual(diagnostic['test']['native_resolution_encodings'], 0)

    def test_actual_over_budget_is_rejected(self):
        branch = SimpleNamespace(device='cpu', prepare_image=lambda image: None)
        def fake(image, geometry, *args, **kwargs):
            for _ in range(5):
                geometry.prepare_image(torch.zeros(1, 3, 512, 512))
        with self.assertRaises(RuntimeError):
            predict_image(torch.zeros(3, 64, 64), branch, {}, branch, {}, local_predictor=fake)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
