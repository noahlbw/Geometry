import unittest
from unittest.mock import patch

import torch

from dinotool import response_collision_alias as current
from dinotool.bounded_covariance_alias import neighbourhood
from dinotool.stratified_soft_alias import WideCrop


class ResponseCollisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6627)
        self.valid = torch.tensor([True]*7+[False])
        self.pairs = torch.tensor([[0, 1]]*8)
        self.canonical = torch.tensor([0, 1, 2])
        self.evidence = torch.randn(8, 3, 4, dtype=torch.float64)
        self.relation = torch.rand(8, 8)
        self.neighbours = neighbourhood(self.relation, self.valid)

    def weights(self, evidence=None, pairs=None, **kwargs):
        return current.collision_weights(self.evidence if evidence is None else evidence,
            self.pairs if pairs is None else pairs, self.canonical, self.neighbours, self.valid, **kwargs)

    def test_matches_explicit_weighted_cross_word_correlations(self):
        actual, _ = self.weights()
        expected = torch.ones_like(actual)
        for q in range(7):
            ids, mass = self.neighbours[0][q], self.neighbours[1][q]
            for side in range(2):
                own, rival = self.pairs[q, side], self.pairs[q, 1-side]
                x, y = self.evidence[ids, own].clone(), self.evidence[ids, rival].clone()
                means = (y*mass[:, None]).sum(0)
                x -= (x*mass[:, None]).sum(0)
                y -= means
                rho = (x.T@(y*mass[:, None]))/torch.outer(
                    (x.square()*mass[:, None]).sum(0).sqrt(), (y.square()*mass[:, None]).sum(0).sqrt())
                expected[q, side] = (1-rho.clamp(0, 1).square()@means.softmax(-1)).clamp_min(1e-6)
                expected[q, side, self.canonical[own]] = 1.
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_same_word_can_collide_with_one_rival_not_another(self):
        t = torch.tensor([-1., -1., 1., 1., -1., -1., 1., 1.])
        u = torch.tensor([-1., 1., -1., 1., -1., 1., -1., 1.])
        evidence = torch.stack((t[:, None].expand(-1, 4), t[:, None].expand(-1, 4),
                                -t[:, None].expand(-1, 4)), 1).double()
        evidence[:, 0, 2] = u
        mass = torch.ones(8, 4, dtype=torch.float64)/4
        neighbours = (torch.tensor([[0, 1, 2, 3]]*8), mass)
        valid = torch.ones(8, dtype=torch.bool)
        ab = current.collision_weights(evidence, self.pairs, self.canonical, neighbours, valid)[0]
        ac = current.collision_weights(evidence, torch.tensor([[0, 2]]*8), self.canonical, neighbours, valid)[0]
        self.assertAlmostEqual(float(ab[0, 0, 1]), 1e-6, places=12)
        self.assertAlmostEqual(float(ac[0, 0, 1]), 1., places=12)
        self.assertAlmostEqual(float(ab[0, 0, 2]), 1., places=12)

    def test_word_information_is_not_algebraically_class_average(self):
        evidence = self.evidence.clone()
        evidence[:, :, 1] = -evidence[:, :, 0]
        evidence[:, :, 3] = -evidence[:, :, 2]
        word, _ = self.weights(evidence)
        pooled, _ = self.weights(evidence, pooled=True)
        self.assertTrue(torch.equal(pooled, torch.ones_like(pooled)))
        self.assertGreater(float((word-pooled).abs().max()), .1)

    def test_canonical_padding_equal_pair_and_constant_patterns_neutral(self):
        weights, _ = self.weights()
        self.assertTrue(torch.equal(weights.gather(-1, self.canonical[self.pairs][..., None]), torch.ones(8, 2, 1, dtype=torch.float64)))
        self.assertTrue(torch.equal(weights[~self.valid], torch.ones_like(weights[~self.valid])))
        for evidence, pairs in ((torch.ones_like(self.evidence), self.pairs),
                                (self.evidence, torch.zeros_like(self.pairs))):
            weights, known = self.weights(evidence, pairs)
            self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
            self.assertFalse(bool(known.any()))

    def test_query_does_not_enter_pattern_or_responsibility(self):
        expected, _ = self.weights()
        evidence = self.evidence.clone()
        evidence[0] += 1000
        actual, _ = self.weights(evidence)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)

    def test_chunking_and_common_class_offsets_are_exact(self):
        a, _ = self.weights(chunk=1)
        b, _ = self.weights(chunk=64)
        torch.testing.assert_close(a, b, atol=1e-14, rtol=0)
        shifted = self.evidence+torch.tensor([2., -4., 3.])[None, :, None]
        c, _ = self.weights(shifted)
        torch.testing.assert_close(a, c, atol=1e-12, rtol=0)

    def fixture(self):
        members = torch.arange(12).reshape(3, 4)
        crop = WideCrop(torch.randn(4, 12), torch.randn(12), 0, 0, 2, 2, 2, 2)
        coordinates = torch.tensor([[.25, .25], [.75, .25], [1.25, .25], [1.75, .25],
                                    [.25, 1.25], [.75, 1.25], [1.25, 1.25], [1.75, 1.25]])
        return (torch.randn(8, 3), torch.diag(self.valid.double()*.5), self.relation,
                self.evidence, [crop], torch.ones(2, 2), coordinates, (2, 2), members,
                self.pairs, self.canonical, self.valid)

    def test_singletons_match_all_control_execution(self):
        args = self.fixture()
        combined, _ = current.scores(*args)
        for name in current.METHODS[3:]:
            single, _ = current.scores(*args, methods=(name,))
            self.assertTrue(torch.equal(combined[name], single[name]), name)

    def test_matched_controls_share_written_norm_and_preserve_padding_gauge(self):
        args = self.fixture()
        values, info = current.scores(*args)
        base = args[0].double()
        primary = values[current.PRIMARY]-base
        for name in (current.MATCHED_POOLED, current.MATCHED_CLASS_MEAN, *current.MATCHED_SHUFFLES,
                     current.SHUFFLED_WRITE, current.DIRECT):
            delta = values[name]-base
            torch.testing.assert_close(delta[self.valid].norm(), primary[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(8, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertEqual(float(delta[~self.valid].abs().max()), 0.)
        self.assertEqual(info['matched_unmatchable'], 0.)

    def test_zero_collision_recovers_baseline_exactly(self):
        args = self.fixture()
        with patch('dinotool.response_collision_alias.collision_weights', return_value=(
                torch.ones(8, 2, 4, dtype=torch.float64), torch.zeros(8, 2, 4, dtype=torch.bool))):
            values, _ = current.scores(*args, methods=(current.PRIMARY,))
        self.assertTrue(torch.equal(values[current.PRIMARY], args[0].double()))

    def test_primary_skips_diagnostic_sources(self):
        with patch('dinotool.response_collision_alias.weight_controls', side_effect=AssertionError), \
             patch('dinotool.response_collision_alias.permuted_operator', side_effect=AssertionError), \
             patch('dinotool.response_collision_alias.match_previous', side_effect=AssertionError):
            current.scores(*self.fixture(), methods=(current.PRIMARY,))

    def test_invalid_source_is_rejected(self):
        evidence = self.evidence.clone()
        evidence[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            self.weights(evidence)
        with self.assertRaises(ValueError):
            current.scores(*self.fixture(), methods=('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
