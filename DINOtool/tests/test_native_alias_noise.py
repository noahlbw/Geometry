import unittest

import torch

from dinotool.native_alias_noise import hard_pair_observation, signed_potential
from dinotool.rival_preserving_alias import competitive_potential
from dinotool.stratified_soft_alias import WideCrop


class NativeAliasNoiseTest(unittest.TestCase):
    def source(self):
        crop = WideCrop(torch.tensor([[4., 2., 0., -1., 1., 3.]]).expand(441, -1), torch.zeros(6), 0, 0, 336, 336)
        coords, members = torch.tensor([[8., 8.], [24., 24.]]), torch.tensor([[0, 1, 2], [3, 4, 5]])
        valid = torch.tensor([True, False])
        return [crop], torch.ones(336, 336), coords, (336, 336), members, torch.zeros(2, 6, 2), valid

    def test_signed_potential_matches_soft_solver(self):
        values = -torch.rand(3, 4, 4)
        values.diagonal(dim1=-2, dim2=-1).zero_()
        valid = torch.ones(3, dtype=torch.bool)
        torch.testing.assert_close(signed_potential(values, valid)[0], competitive_potential(values, valid)[0], atol=0, rtol=0)

    def test_signed_hard_actions_have_consistent_gauge(self):
        directed = torch.tensor([[[0., .7, -.2], [.1, 0., .3], [-.4, -.1, 0.]]], dtype=torch.float64)
        v, stats = signed_potential(directed, torch.ones(1, dtype=torch.bool))
        self.assertLess(abs(float(v.sum())), 1e-15)
        self.assertLess(stats['normal_equation_max_error'], 1e-15)

    def test_hard_zero_risk_is_exact_identity(self):
        self.assertEqual(float(hard_pair_observation(*self.source()).abs().max()), 0.)

    def test_hard_removal_matches_normalized_logmeanexp(self):
        args = list(self.source())
        args[5][0, 1, 1] = .2
        result = hard_pair_observation(*args)
        evidence = torch.tensor([4., 2., 0.], dtype=torch.float64)
        expected = evidence[[0, 2]].logsumexp(0)-evidence.logsumexp(0)+torch.tensor(1.5, dtype=torch.float64).log()
        self.assertAlmostEqual(float(result[0, 0, 1]), float(expected), places=12)
        self.assertEqual(float(result[1].abs().max()), 0.)
        self.assertEqual(float(result.diagonal(dim1=-2, dim2=-1).abs().max()), 0.)

    def test_count_renormalization_can_increase_class_score(self):
        args = list(self.source())
        args[5][0, 2, 1] = .1
        self.assertGreater(float(hard_pair_observation(*args)[0, 0, 1]), 0.)

    def test_no_survivors_fail(self):
        args = list(self.source())
        args[5][0, :3, 1] = 1
        with self.assertRaises(ValueError):
            hard_pair_observation(*args)

    def test_signed_potential_rejects_invalid_self_actions(self):
        with self.assertRaises(ValueError):
            signed_potential(torch.ones(1, 2, 2), torch.ones(1, dtype=torch.bool))
