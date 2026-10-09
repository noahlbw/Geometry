import unittest

import torch

from dinotool.bounded_contrast_coupling import center_valid, contrast_coupling


class ContrastCouplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.local = torch.randn(9, 4, dtype=torch.float64)
        self.broad = torch.randn_like(self.local)
        self.valid = torch.tensor([True] * 7 + [False] * 2)
        self.operator = torch.randn(9, 9, dtype=torch.float64)

    def test_valid_mean_and_padding(self):
        centered = center_valid(self.broad, self.valid)
        self.assertLess(float(centered[self.valid].mean(0).abs().max()), 1e-14)
        self.assertTrue(torch.equal(centered[~self.valid], torch.zeros(2, 4, dtype=torch.float64)))

    def test_uniform_class_offset_has_no_effect(self):
        offset = torch.tensor([[3., -8., 2., 4.]], dtype=torch.float64)
        a = contrast_coupling(self.local, self.broad, self.operator, self.valid)
        b = contrast_coupling(self.local, self.broad + offset, self.operator, self.valid)
        torch.testing.assert_close(a, b, atol=1e-13, rtol=0)
        neutral = contrast_coupling(self.local, self.local + offset, self.operator, self.valid)
        torch.testing.assert_close(neutral, self.local, atol=1e-13, rtol=0)

    def test_correction_preserves_valid_local_mean_and_padding(self):
        result = contrast_coupling(self.local, self.broad, self.operator, self.valid)
        torch.testing.assert_close(result[self.valid].mean(0), self.local[self.valid].mean(0), atol=1e-13, rtol=0)
        self.assertTrue(torch.equal(result[~self.valid], self.local[~self.valid]))

    def test_spatial_innovation_remains(self):
        result = contrast_coupling(self.local, self.broad, torch.eye(9, dtype=torch.float64), self.valid)
        expected = self.local + center_valid(self.broad - self.local, self.valid)
        torch.testing.assert_close(result, expected, atol=1e-13, rtol=0)
        self.assertGreater(float((result - self.local).abs().max()), .1)

    def test_empty_validity_is_exact_identity(self):
        result = contrast_coupling(self.local, self.broad, self.operator, torch.zeros(9, dtype=torch.bool))
        self.assertTrue(torch.equal(result, self.local))


if __name__ == '__main__':
    unittest.main()
