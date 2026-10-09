import unittest

import numpy as np

from dinotool.geometry_density_calibration import between_mode_boundary, calibrate_margins, margin_mode_shift


class DensityCalibrationTests(unittest.TestCase):
    def test_equal_variance_boundary_is_midpoint(self):
        self.assertAlmostEqual(between_mode_boundary([-2., 4.], [1., 1.], [.5, .5]), 1.)

    def test_component_permutation_invariance(self):
        first = between_mode_boundary([-2., 4.], [1., 1.], [.7, .3])
        second = between_mode_boundary([4., -2.], [1., 1.], [.3, .7])
        self.assertAlmostEqual(first, second)

    def test_constant_scores_are_identity(self):
        scores = np.full((8, 3), .2)
        output, result = calibrate_margins(scores, np.ones(8, bool))
        np.testing.assert_array_equal(output, scores)
        self.assertEqual(result["shifts"], [0., 0., 0.])

    def test_clear_modes_are_resolved_without_labels(self):
        rng = np.random.default_rng(27)
        shift, result = margin_mode_shift(np.concatenate((rng.normal(-2, .1, 256), rng.normal(3, .1, 256))))
        self.assertTrue(result["resolved"])
        self.assertTrue(-2 < shift < 3)

    def test_no_valid_support_falls_back(self):
        scores = np.arange(24).reshape(8, 3)
        output, _ = calibrate_margins(scores, np.zeros(8, bool))
        np.testing.assert_array_equal(output, scores)

    def test_invalid_geometry_is_rejected(self):
        scores = np.random.default_rng(2).normal(size=(8, 3))
        with self.assertRaises(ValueError):
            calibrate_margins(scores, np.ones(8, bool), -np.eye(8))


if __name__ == "__main__":
    unittest.main()
