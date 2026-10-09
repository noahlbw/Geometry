import unittest

import torch

from dinotool.bounded_canonical_pair_alias import (alias_weights, canonical_references,
    reference_contrast, rival_support, sparse_pair_potential, weight_controls)
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from dinotool.calibrated_competitive_alias import profiled_logits


class CanonicalPairAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)

    def test_references_use_canonical_agreement_and_ignore_invalid_queries(self):
        local = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.], [3., 0., 0.]])
        wide = local.clone()
        wide[1] = torch.tensor([5., 0., 0.])
        actual = canonical_references(local, wide, torch.tensor([True, True, False, True]))
        self.assertGreater(float(actual[0, 0]), 0.)
        self.assertEqual(float(actual[1].sum()), 0.)
        self.assertEqual(float(actual[2].sum()), 0.)
        self.assertTrue(torch.equal(actual.argmax(-1)[[0, 3]], torch.tensor([0, 0])))

    def test_compact_leave_query_out_moments_match_direct_cohorts(self):
        evidence = torch.randn(9, 3, 4, dtype=torch.float64)
        references = torch.rand(9, 3, dtype=torch.float64)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(9)])
        actual, known = reference_contrast(evidence, references, pairs)
        expected = torch.empty_like(actual)
        for i in range(9):
            for side in range(2):
                own, rival = pairs[i, side], pairs[i, 1-side]
                means, variances = [], []
                for cohort in (own, rival):
                    refs = references[:, cohort].clone()
                    refs[i] = 0.
                    mean = (refs[:, None]*evidence[:, own]).sum(0)/refs.sum()
                    variance = (refs[:, None]*(evidence[:, own]-mean).square()).sum(0)/refs.sum()
                    means.append(mean)
                    variances.append(variance)
                expected[i, side] = (means[0]-means[1])/(variances[0]+variances[1]).sqrt().clamp_min(1e-6)
        self.assertTrue(bool(known.all()))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        changed = evidence.clone()
        changed[0] += 50.
        after, _ = reference_contrast(changed, references, pairs)
        torch.testing.assert_close(actual[0], after[0], atol=1e-11, rtol=1e-11)

    def test_same_alias_can_be_ambiguous_for_one_rival_not_another(self):
        evidence = torch.zeros(8, 3, 3)
        evidence[2:6, 0, 1] = 3.
        evidence[6:, 0, 1] = -3.
        refs = torch.zeros(8, 3)
        refs[2:4, 0], refs[4:6, 1], refs[6:, 2] = 1., 1., 1.
        pairs = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 1], [1, 0], [1, 0], [2, 0], [2, 0]])
        weights, _, _ = alias_weights(evidence, refs, torch.ones(8, 8), pairs,
                                      torch.zeros(3, dtype=torch.long), torch.ones(8, dtype=torch.bool))
        self.assertLess(float(weights[0, 0, 1]), float(weights[1, 0, 1]))
        torch.testing.assert_close(weights[..., 0], torch.ones(8, 2, dtype=torch.float64), atol=0, rtol=0)

    def test_missing_cohort_agreement_and_invalid_have_identity_weights(self):
        evidence, refs = torch.randn(6, 3, 4), torch.zeros(6, 3)
        refs[:3, 0], refs[3:, 1] = 1., 1.
        pairs = torch.tensor([[0, 2], [0, 0], [0, 1], [1, 0], [1, 0], [1, 0]])
        valid = torch.tensor([True, True, False, True, True, True])
        weights, known, _ = alias_weights(evidence, refs, torch.ones(6, 6), pairs,
                                        torch.zeros(3, dtype=torch.long), valid)
        self.assertTrue(torch.equal(weights[:3], torch.ones_like(weights[:3])))
        self.assertFalse(bool(known[:3].any()))

    def test_geometry_support_excludes_self_and_padding(self):
        refs = torch.tensor([[1., 0.], [0., 1.], [100., 0.]])
        pairs = torch.tensor([[0, 1], [1, 0], [0, 1]])
        valid = torch.tensor([True, True, False])
        fractions, _ = rival_support(torch.ones(3, 3), refs, pairs, valid)
        torch.testing.assert_close(fractions[:2], torch.tensor([[1., 0.], [1., 0.]]), atol=0, rtol=0)

    def test_controls_preserve_canonical_mass_and_weight_spectrum(self):
        weights = torch.rand(8, 2, 4, dtype=torch.float64)*.7+.3
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(8)])
        canonical = torch.tensor([0, 2, 1])
        weights.scatter_(-1, canonical[pairs][..., None], 1.)
        shared, shuffled = weight_controls(weights, pairs, canonical)
        torch.testing.assert_close(shared.sum(-1), weights.sum(-1), atol=1e-14, rtol=0)
        torch.testing.assert_close(shuffled.sort(-1).values, weights.sort(-1).values, atol=0, rtol=0)
        self.assertTrue(torch.equal(shuffled.gather(-1, canonical[pairs][..., None]), torch.ones(8, 2, 1, dtype=torch.float64)))

    def test_stencil_writer_matches_weighted_lme_and_has_zero_pair_sum(self):
        members = torch.arange(9).reshape(3, 3)
        crop = WideCrop(torch.randn(4, 9), torch.randn(9), 0, 0, 2, 2, 2, 2)
        count, coordinates = torch.ones(2, 2), torch.tensor([[.5, .5], [.5, 1.5], [1.5, .5], [1.5, 1.5]])
        pairs = torch.tensor([[0, 1], [1, 2], [1, 1], [2, 0]])
        valid, weights = torch.tensor([True, True, True, False]), torch.rand(4, 2, 3, dtype=torch.float64)*.5+.5
        actual = sparse_pair_potential([crop], count, coordinates, (2, 2), members, pairs, weights, valid)
        ids, coeff = crop_stencil(crop, count, coordinates, (2, 2))
        profiled = profiled_logits(crop, members).double()
        expected = torch.zeros_like(actual)
        for q in range(2):
            delta = []
            for side in range(2):
                value = profiled[ids[q], pairs[q, side]]
                change = ((value+weights[q, side].log()).logsumexp(-1)-value.logsumexp(-1)
                          -weights[q, side].mean().log())
                delta.append((change*coeff[q].double()).sum())
            contrast = .5*(delta[0]-delta[1])
            expected[q, pairs[q, 0]], expected[q, pairs[q, 1]] = contrast, -contrast
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual.sum(-1), torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)
        identity = sparse_pair_potential([crop], count, coordinates, (2, 2), members, pairs, torch.ones_like(weights), valid)
        self.assertTrue(torch.equal(identity, torch.zeros_like(identity)))
        crop = WideCrop(torch.ones(4, 9)*3., torch.zeros(9), 0, 0, 2, 2, 2, 2)
        constant = sparse_pair_potential([crop], count, coordinates, (2, 2), members, pairs, weights, valid)
        torch.testing.assert_close(constant, torch.zeros_like(constant), atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
