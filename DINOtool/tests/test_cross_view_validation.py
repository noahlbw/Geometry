import unittest

import torch

from dinotool.cross_view_validation import (
    PRIMARY, padded_crop, shifted_starts, validated_scores, validation_coefficient,
)


class CrossViewValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.g = torch.randn(1, 7, 3)
        self.r = torch.rand(1, 7, 7).softmax(-1)
        self.valid = torch.ones(1, 7, dtype=torch.bool)

    def test_closed_form_matches_bounded_search(self):
        d, target = torch.randn_like(self.g), torch.randn_like(self.g)
        coefficient, before, after = validation_coefficient(d, target, self.r)
        grid = torch.linspace(0, 1, 10001)
        losses = torch.stack([((self.r @ (a*d-target)).double().square().sum()) for a in grid])
        self.assertAlmostEqual(float(coefficient), float(grid[losses.argmin()]), places=3)
        self.assertLessEqual(float(after), float(before)+1e-6)

    def test_endpoints_and_zero_direction(self):
        d = torch.randn_like(self.g)
        for target, expected in ((-d, 0.), (2*d, 1.), (.3*d, .3)):
            coefficient, _, _ = validation_coefficient(d, target, self.r)
            self.assertAlmostEqual(float(coefficient), expected, places=6)
        coefficient, _, _ = validation_coefficient(torch.zeros_like(d), d, self.r)
        self.assertEqual(float(coefficient), 0.)

    def test_no_innovation_recovers_original(self):
        output, _ = validated_scores(self.g, self.g, self.g, self.r, self.valid)
        self.assertTrue(torch.equal(output[PRIMARY], self.g))

    def test_view_exchange_and_class_offset_invariance(self):
        first, second = torch.randn_like(self.g), torch.randn_like(self.g)
        original, _ = validated_scores(self.g, first, second, self.r, self.valid)
        swapped, _ = validated_scores(self.g, second, first, self.r, self.valid)
        shifted, _ = validated_scores(self.g, first+torch.randn(1, 7, 1),
                                     second+torch.randn(1, 7, 1), self.r, self.valid)
        torch.testing.assert_close(original[PRIMARY], swapped[PRIMARY])
        torch.testing.assert_close(original[PRIMARY], shifted[PRIMARY], atol=2e-6, rtol=2e-6)

    def test_invalid_queries_receive_no_update(self):
        self.valid[:, -2:] = False
        output, _ = validated_scores(self.g, torch.randn_like(self.g), torch.randn_like(self.g), self.r, self.valid)
        self.assertTrue(torch.equal(output[PRIMARY][:, -2:], self.g[:, -2:]))

    def test_shifted_crop_coordinates_and_coverage(self):
        image = torch.arange(3*448*240).reshape(3, 448, 240).float()
        count = torch.zeros(448, 240)
        for top in shifted_starts(448):
            for left in shifted_starts(240):
                crop, (y0, y1, x0, x1, cy, cx) = padded_crop(image, top, left)
                self.assertEqual(tuple(crop.shape), (3, 336, 336))
                self.assertTrue(torch.equal(crop[:, cy:cy+y1-y0, cx:cx+x1-x0], image[:, y0:y1, x0:x1]))
                count[y0:y1, x0:x1] += 1
        self.assertTrue(bool((count > 0).all()))


if __name__ == "__main__":
    unittest.main()
