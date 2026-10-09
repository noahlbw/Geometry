import unittest

import numpy as np
import torch

from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.native_alias_noise import hard_pair_observation
from dinotool.survival_alias import soft_survival_observation, signed_allocation_controls, SPATIAL
from test_native_alias_noise import NativeAliasNoiseTest


class SurvivalAliasTest(unittest.TestCase):
    def source(self):
        return list(NativeAliasNoiseTest().source())

    def test_zero_risk_exact_identity(self):
        self.assertEqual(float(soft_survival_observation(*self.source()).abs().max()), 0.)

    def test_binary_risk_matches_hard(self):
        args = self.source()
        args[5][0, 1, 1] = 1
        args[5][0, 4, 0] = 1
        torch.testing.assert_close(soft_survival_observation(*args), hard_pair_observation(*args), atol=0, rtol=0)

    def test_fractional_weight_matches_direct_aggregation(self):
        args = self.source()
        args[5][0, 1, 1] = .4
        actual = soft_survival_observation(*args)[0, 0, 1]
        evidence = profiled_logits(args[0][0], args[4])[0, 0].double()
        weight = 1-args[5][0, args[4][0], 1].double()
        expected = (evidence+weight.log()).logsumexp(0)-evidence.logsumexp(0)+(3/weight.sum()).log()
        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_uniform_reliability_does_not_recalibrate_class(self):
        args = self.source()
        args[5][:, :3, 1] = .6
        self.assertLess(float(soft_survival_observation(*args).abs().max()), 1e-12)

    def test_signed_action_can_promote_or_suppress(self):
        high, low = self.source(), self.source()
        high[5][0, 0, 1], low[5][0, 2, 1] = .5, .5
        self.assertLess(float(soft_survival_observation(*high)[0, 0, 1]), 0.)
        self.assertGreater(float(soft_survival_observation(*low)[0, 0, 1]), 0.)

    def test_empty_mass_fails(self):
        args = self.source()
        args[5][0, :3, 1] = 1
        with self.assertRaises(ValueError):
            soft_survival_observation(*args)

    def test_chunking_invariant(self):
        args = self.source()
        args[5][0, 1, 1] = .3
        torch.testing.assert_close(soft_survival_observation(*args, chunk=1), soft_survival_observation(*args, chunk=2), atol=0, rtol=0)

    def test_signed_controls_preserve_budget_and_shuffle_spectrum(self):
        values = np.array([[[0., .3], [-.2, 0.]], [[0., -.1], [.4, 0.]], [[0., 0.], [0., 0.]]])
        valid = np.array([True, True, False])
        for name, result in signed_allocation_controls(values, valid).items():
            np.testing.assert_allclose(result.sum(0), values.sum(0), atol=1e-15)
            self.assertFalse(result[~valid].any())
            self.assertFalse(np.diagonal(result, axis1=1, axis2=2).any())
            if name in SPATIAL:
                np.testing.assert_array_equal(np.sort(result[valid], axis=0), np.sort(values[valid], axis=0))

