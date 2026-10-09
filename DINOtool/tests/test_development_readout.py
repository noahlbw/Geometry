import unittest
import numpy as np
import torch

from dinotool.development_readout import (Profile, profiles, development_keys,
    coupled, threshold_histogram, threshold_matrices, biased_probabilities, score)


class DevelopmentReadoutTests(unittest.TestCase):
    def test_fixed_grid_contains_unique_original(self):
        grid = profiles(['original', 'focused'])
        self.assertEqual(grid[0], Profile())
        self.assertEqual(len(grid), len(set(grid)))

    def test_disjoint_split_preserves_order(self):
        keys = [str(i) for i in range(80)]
        dev = development_keys(keys, 16)
        self.assertEqual(len(dev), 16)
        self.assertEqual(dev, [k for k in keys if k in set(dev)])
        self.assertEqual(len(set(keys) - set(dev)), 64)
        self.assertEqual(dev, development_keys(keys, 16))

    def test_exact_threshold_histograms_and_ties(self):
        p = torch.tensor([[[.5, .4, .2]], [[.5, .6, .8]]])
        y = torch.tensor([[0, 1, 0]])
        cuts = (0., .5, .6, .9)
        hist = threshold_histogram(p, y, 0, thresholds=cuts)
        for t, matrix in threshold_matrices(hist, 0, cuts).items():
            confidence, prediction = p.max(0)
            prediction = prediction.masked_fill(confidence < t, 0)
            expected = torch.bincount((y * 2 + prediction).flatten(), minlength=4).reshape(2, 2)
            np.testing.assert_array_equal(matrix, expected)

    def test_ignore_and_foreground_protocol(self):
        p = torch.tensor([[[.8, .1]], [[.2, .9]]])
        hist = threshold_histogram(p, torch.tensor([[-1, 1]]), None)
        matrix = threshold_matrices(hist, None)[0.]
        self.assertEqual(int(matrix.sum()), 1)
        self.assertEqual(score(matrix), 100.)

    def test_bias_keeps_normalization_and_foreground_ratios(self):
        p = torch.tensor([[[.2]], [[.3]], [[.5]]])
        q = biased_probabilities(p, 2., 0)
        torch.testing.assert_close(q.sum(0), torch.ones(1, 1))
        torch.testing.assert_close(q[1] / q[2], p[1] / p[2])
        self.assertGreater(float(q[0]), float(p[0]))

    def test_original_coupling_and_limits(self):
        local, broad, h = torch.randn(4, 3), torch.randn(4, 3), torch.eye(4) * .5
        torch.testing.assert_close(coupled(local, broad, h, 1.),
            local.double() + h.double() @ (broad.double() - local.double()), rtol=0, atol=0)
        torch.testing.assert_close(coupled(local, broad, h, 2.), broad.double(), rtol=0, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
