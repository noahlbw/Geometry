import unittest

import torch

from dinotool.bounded_coupling_controls import local_only, mean_logits, shuffled_operator, wide_only


class SameFieldControlsTest(unittest.TestCase):
    def setUp(self):
        self.local = torch.arange(12, dtype=torch.float64).reshape(4, 3)
        self.broad = self.local.flip(0)
        self.operator = torch.eye(4, dtype=torch.float64) * .5
        self.valid = torch.tensor([True, True, True, False])

    def test_identity_ridge_is_equal_mean(self):
        expected = self.local + self.operator @ (self.broad - self.local)
        torch.testing.assert_close(mean_logits(self.local, self.broad, self.operator, self.valid), expected)

    def test_endpoints_and_duplicate_sources(self):
        self.assertIs(local_only(self.local, self.broad, self.operator, self.valid), self.local)
        self.assertIs(wide_only(self.local, self.broad, self.operator, self.valid), self.broad)
        torch.testing.assert_close(mean_logits(self.local, self.local, self.operator, self.valid), self.local)

    def test_permutation_keeps_padding_and_spectrum(self):
        weights = torch.arange(9, dtype=torch.float64).reshape(3, 3) / 10
        operator = torch.zeros(4, 4, dtype=torch.float64)
        operator[:3, :3] = weights.T @ weights
        shuffled = shuffled_operator(operator, self.valid)
        torch.testing.assert_close(torch.linalg.eigvalsh(shuffled), torch.linalg.eigvalsh(operator))
        self.assertEqual(int(torch.count_nonzero(shuffled[3])), 0)
        self.assertEqual(int(torch.count_nonzero(shuffled[:, 3])), 0)
        torch.testing.assert_close(shuffled, shuffled_operator(operator, self.valid))


if __name__ == '__main__':
    unittest.main()
