from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dinotool.vip_resolution_coupling import (METHODS, PRIMARY, crop_windows,
    padded_crop, patch_coordinates, predict_image, resize_for_vip)


class FakeGeometry:
    device = torch.device('cpu')

    def __init__(self):
        self.inputs = []

    def prepare_image(self, image):
        self.inputs.append(tuple(image.shape))
        features = F.adaptive_avg_pool2d(image, (21, 21)).flatten(2).transpose(1, 2)
        features = F.normalize(torch.cat((features, torch.ones_like(features[..., :1])), -1), dim=-1)
        return SimpleNamespace(grid_height=21, grid_width=21, native_projected=features,
            geometry_projected=features, geometry_patch_conditional=torch.eye(441)[None])


class FakeVIP:
    def __init__(self):
        self.inputs = []

    def crop_patch_features(self, image):
        self.inputs.append(tuple(image.shape))
        features = F.adaptive_avg_pool2d(image[None], (21, 21)).flatten(2).transpose(1, 2)
        return F.normalize(torch.cat((features, torch.ones_like(features[..., :1])), -1), dim=-1)


def banks():
    torch.manual_seed(15)
    aliases = tuple(['a'] + [f'a{i}' for i in range(1, 20)] + ['b'] + [f'b{i}' for i in range(1, 20)])
    parents = torch.arange(2).repeat_interleave(20)
    features = F.normalize(torch.randn(40, 4), dim=-1)
    bank = SimpleNamespace(alias_names=aliases, parent_indices=parents, class_count=2, features=features)
    query = SimpleNamespace(aliases=aliases, parents=parents, class_names=('a', 'b'),
                            features=features[:, None].repeat(1, 2, 1))
    return {'test': bank}, {'test': query}


class VIPResolutionTests(unittest.TestCase):
    def test_layout_matches_literal_vip_and_covers_boundaries(self):
        for height, width in ((336, 448), (448, 448), (236, 448), (448, 252), (7, 448)):
            actual = crop_windows(height, width)
            expected = []
            coverage = np.zeros((height, width), dtype=int)
            rows = max(height - 336 + 112 - 1, 0) // 112 + 1
            columns = max(width - 336 + 112 - 1, 0) // 112 + 1
            for y in range(rows):
                for x in range(columns):
                    bottom, right = min(y * 112 + 336, height), min(x * 112 + 336, width)
                    top, left = max(bottom - 336, 0), max(right - 336, 0)
                    expected.append((top, left, bottom - top, right - left))
            self.assertEqual([(w.top, w.left, w.height, w.width) for w in actual], expected)
            self.assertLessEqual(len(actual), 4)
            for w in actual:
                coverage[w.top:w.top + w.height, w.left:w.left + w.width] += 1
            self.assertTrue((coverage > 0).all())
        with self.assertRaises(ValueError):
            crop_windows(3000, 4000)

    def test_resize_matches_vip_uint8_cv2_protocol(self):
        image = torch.arange(3 * 13 * 29).reshape(3, 13, 29).float() / (3 * 13 * 29)
        actual = resize_for_vip(image)
        array = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        expected = cv2.resize(array, (448, int(13 * 448 / 29 + .5)), interpolation=cv2.INTER_LINEAR)
        self.assertTrue(torch.equal(actual, torch.from_numpy(expected).permute(2, 0, 1).float() / 255))

    def test_real_336_padding_and_21_grid(self):
        image = torch.rand(3, 236, 448)
        w = crop_windows(236, 448)[1]
        crop = padded_crop(image, w)
        self.assertEqual(tuple(crop.shape), (3, 336, 336))
        self.assertTrue(torch.equal(crop[:, :236], image[:, :, 112:448]))
        self.assertFalse(bool(crop[:, 236:].any()))
        coordinates, valid = patch_coordinates(w, 'cpu')
        self.assertEqual(tuple(coordinates.shape), (441, 2))
        self.assertEqual(int(valid.sum()), 15 * 21)
        self.assertEqual(coordinates[0].tolist(), [8., 120.])

    def test_full_image_prediction_and_singleton_match(self):
        image = torch.rand(3, 128, 128)
        bank, query = banks()
        geometry, vip = FakeGeometry(), FakeVIP()
        actual, diagnostic = predict_image(image, geometry, bank, vip, query)
        self.assertEqual(geometry.inputs, [(1, 3, 336, 336)] * 4)
        self.assertEqual(vip.inputs, [(3, 336, 336)] * 4)
        self.assertEqual(set(actual['test']), set(METHODS))
        for value in actual['test'].values():
            self.assertEqual(value.shape, (128, 128))
            self.assertEqual(value.dtype, np.uint8)
        singleton, _ = predict_image(image, FakeGeometry(), bank, FakeVIP(), query, methods=(PRIMARY,))
        self.assertTrue(np.array_equal(actual['test'][PRIMARY], singleton['test'][PRIMARY]))
        self.assertEqual(diagnostic['test']['native_resolution_encodings'], 0)
        self.assertEqual(diagnostic['test']['fine_forwards'], 0)
        self.assertEqual(diagnostic['test']['canonical_weight_max_error'], 0)

    def test_4k_input_never_triggers_native_geometry_windows(self):
        bank, query = banks()
        geometry, vip = FakeGeometry(), FakeVIP()
        image = torch.zeros(3, 3000, 4000)
        output, diagnostic = predict_image(image, geometry, bank, vip, query, methods=('NoAdmission_Exact',))
        self.assertEqual(len(geometry.inputs), 2)
        self.assertEqual(len(vip.inputs), 2)
        self.assertEqual(diagnostic['test']['resized_size'], [336, 448])
        self.assertEqual(output['test']['NoAdmission_Exact'].shape, (3000, 4000))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
