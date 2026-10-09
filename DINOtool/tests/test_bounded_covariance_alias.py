import unittest

import torch

from dinotool.bounded_covariance_alias import covariance_weights, fixed_slot_potential, neighbourhood, text_weights
from dinotool.bounded_canonical_pair_alias import weight_controls
from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.bounded_patch_only import readout
from dinotool.geometry_execution import prepare_image_pruned
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from test_geometry_execution import fixture


class CovarianceAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)

    def test_neighbours_exclude_self_and_invalid_and_have_bounded_normalized_mass(self):
        valid = torch.arange(40) < 35
        for global_support in (False, True):
            ids, mass = neighbourhood(torch.rand(40, 40), valid, global_support=global_support)
            self.assertLessEqual(ids.shape[1], 32)
            self.assertFalse(bool((mass[(ids == torch.arange(40)[:, None]) | ~valid[ids]] != 0).any()))
            torch.testing.assert_close(mass[:35].sum(-1), torch.ones(35, dtype=torch.float64), atol=1e-14, rtol=0)
            self.assertTrue(torch.equal(mass[35:], torch.zeros_like(mass[35:])))

    def test_native_head_batch_axis_is_removed_before_response_covariance(self):
        model = fixture()
        prepared = prepare_image_pruned(model, torch.rand(1, 3, 32, 32))
        descriptors = readout(model.backbone.model.visual_model.head, prepared, 2.)[0]
        self.assertEqual(tuple(descriptors.shape[:2]), (1, 4))
        responses = (descriptors@torch.randn(descriptors.shape[-1], 6))[0]
        self.assertEqual(tuple(responses.shape), (4, 6))
        pairs = torch.tensor([[0, 1]]*4)
        valid = torch.ones(4, dtype=torch.bool)
        weights, _ = covariance_weights(responses.reshape(4, 3, 2), responses[:, ::2], pairs,
            torch.zeros(3, dtype=torch.long), neighbourhood(prepared.geometry_patch_conditional[0], valid), valid)
        self.assertEqual(tuple(weights.shape), (4, 2, 2))

    def test_weighted_correlation_matches_direct_computation(self):
        evidence, anchors = torch.randn(9, 3, 4, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64)
        valid = torch.ones(9, dtype=torch.bool)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(9)])
        canonical = torch.tensor([0, 1, 2])
        neighbours = neighbourhood(torch.rand(9, 9), valid)
        actual, known = covariance_weights(evidence, anchors, pairs, canonical, neighbours, valid)
        expected = torch.ones_like(actual)
        for q in range(9):
            ids, mass = neighbours[0][q], neighbours[1][q]
            for side in range(2):
                own, rival = pairs[q, side], pairs[q, 1-side]
                direction = (anchors[ids, own]-anchors[ids, rival]).double()
                direction -= (direction*mass).sum()
                for alias in range(4):
                    values = evidence[ids, own, alias].double()
                    values -= (values*mass).sum()
                    rho = (values*direction*mass).sum()/((values.square()*mass).sum()*(direction.square()*mass).sum()).sqrt()
                    expected[q, side, alias] = (1+rho.clamp(-1, 1))*.5
                expected[q, side, canonical[own]] = 1.
        self.assertTrue(bool(known.all()))
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_offsets_and_positive_scales_do_not_change_weights(self):
        evidence, anchors = torch.randn(12, 3, 4), torch.randn(12, 3)
        valid = torch.ones(12, dtype=torch.bool)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(12)])
        neighbours = neighbourhood(torch.rand(12, 12), valid)
        args = (pairs, torch.zeros(3, dtype=torch.long), neighbours, valid)
        a = covariance_weights(evidence, anchors, *args)[0]
        b = covariance_weights(evidence*3+7, anchors*2+5, *args)[0]
        torch.testing.assert_close(a, b, atol=3e-7, rtol=0)

    def test_same_alias_can_discriminate_one_pair_but_not_another(self):
        anchors = torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 1., 0.],
                                [0., -1., 0.], [-1., -1., 0.], [0., 1., 0.]])
        evidence = torch.zeros(6, 3, 3)
        evidence[:, 0, 1] = anchors[:, 0]
        pairs = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 1], [0, 1], [0, 1]])
        valid = torch.ones(6, dtype=torch.bool)
        weights, _ = covariance_weights(evidence, anchors, pairs, torch.zeros(3, dtype=torch.long),
                                        neighbourhood(torch.ones(6, 6), valid), valid)
        self.assertEqual(float(weights[0, 0, 1]), .5)
        self.assertGreater(float(weights[1, 0, 1]), .999999)
        self.assertEqual(float(weights[..., 0].min()), 1.)

    def test_missing_variation_padding_and_equal_pair_have_identity_weights(self):
        evidence, anchors = torch.ones(6, 3, 4), torch.randn(6, 3)
        valid = torch.tensor([True]*5+[False])
        pairs = torch.tensor([[0, 1], [0, 0], [1, 2], [0, 2], [1, 0], [0, 1]])
        weights, known = covariance_weights(evidence, anchors, pairs, torch.zeros(3, dtype=torch.long),
                                            neighbourhood(torch.ones(6, 6), valid), valid)
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertFalse(bool(known.any()))

    def test_self_response_does_not_enter_its_own_weight_estimate(self):
        evidence, anchors = torch.randn(7, 3, 4), torch.randn(7, 3)
        valid = torch.ones(7, dtype=torch.bool)
        pairs = torch.tensor([[0, 1]]*7)
        args = (pairs, torch.zeros(3, dtype=torch.long), neighbourhood(torch.ones(7, 7), valid), valid)
        expected = covariance_weights(evidence, anchors, *args)[0]
        evidence[0] += 100.
        anchors[0] -= 100.
        actual = covariance_weights(evidence, anchors, *args)[0]
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)

    def test_controls_preserve_weight_mass_spectrum_and_canonical(self):
        weights = torch.rand(8, 2, 4, dtype=torch.float64)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(8)])
        canonical = torch.tensor([0, 1, 2])
        weights.scatter_(-1, canonical[pairs][..., None], 1.)
        mean, shuffled = weight_controls(weights, pairs, canonical)
        torch.testing.assert_close(mean.sum(-1), weights.sum(-1), atol=1e-14, rtol=0)
        torch.testing.assert_close(shuffled.sort(-1).values, weights.sort(-1).values, atol=0, rtol=0)
        self.assertTrue(torch.equal(shuffled.gather(-1, canonical[pairs][..., None]), torch.ones(8, 2, 1, dtype=torch.float64)))

    def test_fixed_slot_writer_matches_explicit_lse_without_mass_compensation(self):
        members = torch.arange(9).reshape(3, 3)
        crop = WideCrop(torch.randn(4, 9), torch.randn(9), 0, 0, 2, 2, 2, 2)
        coordinates = torch.tensor([[.5, .5], [1.5, 1.5]])
        pairs, valid = torch.tensor([[0, 1], [1, 2]]), torch.ones(2, dtype=torch.bool)
        weights = torch.rand(2, 2, 3, dtype=torch.float64)*.8+.2
        ids, coefficients = crop_stencil(crop, torch.ones(2, 2), coordinates, (2, 2))
        profiled = profiled_logits(crop, members).double()
        for normalized in (False, True):
            actual = fixed_slot_potential([crop], torch.ones(2, 2), coordinates, (2, 2), members,
                                          pairs, weights, valid, normalized=normalized)
            expected = torch.zeros_like(actual)
            for q in range(2):
                delta = []
                for side in range(2):
                    scores = profiled[ids[q], pairs[q, side]]
                    change = (scores+weights[q, side].log()).logsumexp(-1)-scores.logsumexp(-1)
                    self.assertTrue(bool((change <= 1e-12).all()))
                    if normalized:
                        change -= weights[q, side].mean().log()
                    delta.append((change*coefficients[q].double()).sum())
                value = .5*(delta[0]-delta[1])
                expected[q, pairs[q, 0]], expected[q, pairs[q, 1]] = value, -value
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(actual.sum(-1), torch.zeros(2, dtype=torch.float64)))
        identity = fixed_slot_potential([crop], torch.ones(2, 2), coordinates, (2, 2), members,
                                        pairs, torch.ones_like(weights), valid)
        self.assertTrue(torch.equal(identity, torch.zeros_like(identity)))

    def test_text_control_is_pair_conditioned_and_bounded(self):
        features = torch.randn(3, 4, 5)
        pairs = torch.tensor([[0, 1], [0, 2], [0, 0]])
        canonical = torch.tensor([0, 1, 2])
        actual = text_weights(features, features[torch.arange(3), canonical], pairs, canonical, torch.ones(3, dtype=torch.bool))
        self.assertFalse(torch.equal(actual[0, 0, 1:], actual[1, 0, 1:]))
        self.assertTrue(bool(((actual > 0) & (actual <= 1)).all()))
        self.assertTrue(torch.equal(actual[2], torch.ones_like(actual[2])))


if __name__ == '__main__':
    unittest.main()
