import unittest

import numpy as np

from dinotool.alias_spatial_allocation import (ROUNDING_GUARD, capacity_balanced_mean,
    feasible_spatial_shuffle, matched_controls, validated_magnitudes)


class AliasSpatialAllocationTest(unittest.TestCase):
    def test_shuffle_preserves_exact_spectrum_and_caps(self):
        cap = np.array([[.1, .2], [.8, .5], [.9, .7], [.3, .4]])
        delta = -np.array([[.1, .1], [.6, .2], [.8, .6], [.2, .3]])
        valid = np.ones(4, bool)
        for seed in (3, 4, 5):
            changed = feasible_spatial_shuffle(delta, cap, valid, seed)
            np.testing.assert_array_equal(np.sort(changed, axis=0), np.sort(delta, axis=0))
            self.assertTrue((changed >= -cap).all())

    def test_random_assignment_is_reproducible_and_nontrivial(self):
        delta = -np.arange(16).reshape(8, 2)/20
        cap = np.ones_like(delta)
        valid = np.ones(8, bool)
        a = feasible_spatial_shuffle(delta, cap, valid, 8)
        b = feasible_spatial_shuffle(delta, cap, valid, 8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, delta))

    def test_mean_preserves_budget_with_saturation(self):
        cap = np.array([[.1], [.8], [.9]])
        delta = np.array([[-.1], [-.2], [-.9]])
        changed = capacity_balanced_mean(delta, cap, np.ones(3, bool))
        np.testing.assert_allclose(changed[:, 0], [-.1, -.55, -.55], atol=1e-12)
        np.testing.assert_allclose(changed.sum(0), delta.sum(0), atol=1e-12)

    def test_mean_is_constant_if_caps_allow_it(self):
        delta = -np.array([[.1, .2], [.3, .4]])
        changed = capacity_balanced_mean(delta, np.ones_like(delta), np.ones(2, bool))
        np.testing.assert_allclose(changed, [[-.2, -.3], [-.2, -.3]])

    def test_invalid_donors_stay_inert(self):
        delta = np.array([[-.1, -.2], [0., 0.], [-.3, -.4]])
        for changed in matched_controls(delta, np.ones_like(delta), np.array([True, False, True])).values():
            np.testing.assert_array_equal(changed[1], [0., 0.])
            np.testing.assert_allclose(changed.sum(0), delta.sum(0), atol=1e-12)

    def test_zero_or_empty_action_is_identity(self):
        for valid in (np.ones(3, bool), np.zeros(3, bool)):
            for changed in matched_controls(np.zeros((3, 2)), np.zeros((3, 2)), valid).values():
                np.testing.assert_array_equal(changed, np.zeros((3, 2)))

    def test_raw_mean_can_violate_capacity_but_balanced_does_not(self):
        delta, cap, valid = np.array([[0.], [-.8]]), np.array([[0.], [1.]]), np.ones(2, bool)
        controls = matched_controls(delta, cap, valid)
        self.assertLess(controls['RawMean_Exact'][0, 0], -cap[0, 0])
        self.assertTrue((controls['CapacityMean_Exact'] >= -cap).all())

    def test_numerical_slack_is_measured_not_free_budget(self):
        cap, delta = np.array([[.1]]), np.array([[-.1-ROUNDING_GUARD/2]])
        magnitude, effective = validated_magnitudes(delta, cap, np.ones(1, bool))
        np.testing.assert_array_equal(magnitude, effective)
        with self.assertRaises(ValueError):
            validated_magnitudes(delta-ROUNDING_GUARD, cap, np.ones(1, bool))

    def test_greedy_feasible_assignment_for_random_fields(self):
        generator = np.random.default_rng(4)
        for _ in range(12):
            cap = generator.uniform(0, 1, (40, 4))
            delta = -cap*generator.uniform(0, 1, cap.shape)
            valid = generator.uniform(0, 1, 40) > .2
            delta[~valid] = 0
            for changed in matched_controls(delta, cap, valid).values():
                np.testing.assert_allclose(changed.sum(0), delta.sum(0), atol=1e-12)
            for name, changed in matched_controls(delta, cap, valid).items():
                if name != 'RawMean_Exact':
                    self.assertTrue((changed >= -cap-1e-12).all())


if __name__ == '__main__':
    unittest.main()
