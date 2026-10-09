import unittest

import torch

from dinotool.rival_alias_influence_soft import influence_weights, influence_directed
from dinotool.rival_alias_fast import cache_observations
from dinotool.stratified_soft_alias import WideCrop


class RivalAliasInfluenceSoftTest(unittest.TestCase):
    def test_identity_protection_and_monotonicity(self):
        pi = torch.tensor([[[[.1, .3, .6]]]])
        risk = torch.tensor([[[[0., .2], [0., .2], [0., .2]]]])
        weights = influence_weights(pi, risk)
        self.assertTrue(torch.equal(weights[..., 0], torch.ones_like(weights[..., 0])))
        self.assertTrue(bool((weights > 0).all() and (weights <= 1).all()))
        self.assertTrue(bool((weights[0, 0, 0, 1:, 1] < weights[0, 0, 0, :-1, 1]).all()))
        self.assertTrue(bool((influence_weights(pi, risk*.5) >= weights).all()))

    def test_saturated_responsibility_is_finite_and_canonical_has_unit_weight(self):
        pi = torch.tensor([[[[1., 0.]]]])
        risk = torch.tensor([[[[0., .1], [0., 0.]]]])
        weights = influence_weights(pi, risk)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertEqual(float(weights[0, 0, 0, 0, 0]), 1.)
        self.assertGreater(float(weights[0, 0, 0, 0, 1]), 0.)

    def test_zero_risk_writer_identity_invalid_and_self_actions(self):
        torch.manual_seed(105)
        members = torch.arange(40).reshape(2, 20)
        coords = torch.tensor([[8., 8.], [128., 128.], [520., 520.]])
        valid = torch.tensor([True, True, False])
        crop = WideCrop(torch.randn(441, 40), torch.randn(40), 0, 0, 336, 336)
        observations = cache_observations([crop], torch.ones(336, 336), coords, (512, 512), members)
        risk = torch.zeros(3, 40, 2)
        self.assertEqual(float(influence_directed(observations, members, risk, valid).abs().max()), 0.)
        risk[:, 1:20, 1] = .5
        result = influence_directed(observations, members, risk, valid)
        self.assertEqual(float(result[~valid].abs().max()), 0.)
        self.assertEqual(float(result.diagonal(dim1=-1, dim2=-2).abs().max()), 0.)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_rejects_invalid_risk(self):
        with self.assertRaises(ValueError):
            influence_weights(torch.ones(1, 1, 1, 1), torch.full((1, 1, 1, 2), 2.))


if __name__ == '__main__':
    unittest.main()
