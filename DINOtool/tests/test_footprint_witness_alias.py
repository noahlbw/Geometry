import unittest

import torch

from dinotool.footprint_witness_alias import (CLASS_MEAN, GUARD_ONLY, MATCHED_OLD, METHODS,
    OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, footprint_interval, footprint_risks, footprint_scores)
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop, sampled_cached


class FootprintWitnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3606)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.parents = torch.arange(3).repeat_interleave(4)
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            coefficients = torch.tensor([[.3, .7]]*4)
            target.append(CachedCrop(evidence, contribution_margins(evidence), ids, coefficients))

    def scores(self, methods=METHODS[3:]):
        return footprint_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_interval_bounds_original_weighted_mean(self):
        lower, upper, complete = footprint_interval(self.fine, self.valid)
        mean = sampled_cached(self.fine, self.coordinates)
        self.assertTrue(torch.equal(complete, self.valid))
        self.assertTrue(bool((mean[complete] >= lower[complete]-1e-6).all()))
        self.assertTrue(bool((mean[complete] <= upper[complete]+1e-6).all()))

    def test_negative_mean_with_positive_contributor_is_not_a_witness(self):
        wide = torch.ones(4, 12, 3)
        mean, upper = -wide, wide.clone()
        actual, previous, guard = footprint_risks(wide, mean, upper, self.valid,
            self.parents, self.canonical, self.valid)
        self.assertGreater(float(previous.max()), 0)
        self.assertEqual(float(actual.abs().max()), 0)
        self.assertEqual(float(guard.abs().max()), 0)

    def test_constant_footprint_recovers_previous_attenuation(self):
        wide, mean = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        actual, previous, guard = footprint_risks(wide, mean, mean, self.valid,
            self.parents, self.canonical, self.valid)
        self.assertTrue(torch.equal(actual, previous))
        self.assertTrue(torch.equal(guard, previous))

    def test_upper_negative_bound_is_no_stronger_than_mean_negative(self):
        wide = torch.ones(4, 12, 3)
        actual, previous, guard = footprint_risks(wide, -wide*2, -wide*.25, self.valid,
            self.parents, self.canonical, self.valid)
        self.assertTrue(bool((actual <= previous).all()))
        self.assertTrue(torch.equal(guard, previous))
        self.assertGreater(float((guard-actual).max()), .4)

    def test_zero_weight_positive_token_cannot_veto(self):
        crop = self.fine[0]
        margins = torch.full_like(crop.margins, -1)
        margins[1] = 100
        indices = torch.tensor([[0, 1]]*4)
        changed = CachedCrop(crop.evidence, margins, indices, torch.tensor([[1., 0.]]*4))
        lower, upper, complete = footprint_interval((changed,), self.valid)
        self.assertTrue(torch.equal(upper[complete], torch.full_like(upper[complete], -1)))
        self.assertTrue(torch.equal(lower[complete], upper[complete]))

    def test_multiple_crops_contribute_to_one_interval(self):
        first = self.fine[0]
        a = CachedCrop(first.evidence, torch.full_like(first.margins, -2), first.indices, first.coefficients*.5)
        b = CachedCrop(first.evidence, torch.full_like(first.margins, 3), first.indices, first.coefficients*.5)
        lower, upper, complete = footprint_interval((a, b), self.valid)
        self.assertTrue(torch.equal(lower[complete], torch.full_like(lower[complete], -2)))
        self.assertTrue(torch.equal(upper[complete], torch.full_like(upper[complete], 3)))

    def test_incomplete_source_is_neutral(self):
        first = self.fine[0]
        partial = CachedCrop(first.evidence, first.margins, first.indices, first.coefficients*.5)
        lower, upper, complete = footprint_interval((partial,), self.valid)
        self.assertFalse(bool(complete.any()))
        wide, mean = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        risk, _, _ = footprint_risks(wide, mean, upper, complete, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk.abs().max()), 0)
        self.assertEqual(float(lower.abs().max()), 0)

    def test_canonical_self_invalid_and_rival_specific_protection(self):
        wide, mean = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        upper = mean.clone()
        upper[0, 1, 2] = 1
        risk, _, _ = footprint_risks(wide, mean, upper, self.valid, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)
        self.assertGreater(float(risk[0, 1, 1]), 0)
        self.assertEqual(float(risk[0, 1, 2]), 0)

    def test_original_positive_soft_and_hard_replay(self):
        actual, _ = self.scores()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(PREVIOUS_MEAN, PREVIOUS_SOFT, PREVIOUS_HARD))
        for new, old in ((OBSERVATION_MEAN, PREVIOUS_MEAN), (OLD_SOFT, PREVIOUS_SOFT), (OLD_HARD, PREVIOUS_HARD)):
            self.assertTrue(torch.equal(actual[new], previous[old]), new)

    def test_singleton_matches_combined_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            single, _ = self.scores((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_no_change_for_equal_observations(self):
        self.fine = self.wide
        scores, _ = self.scores()
        self.assertTrue(torch.equal(scores[PRIMARY], scores[OBSERVATION_MEAN]))

    def test_geometry_does_not_certify_negative_semantics(self):
        a, _ = self.scores((PRIMARY,))
        b, _ = footprint_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, methods=(PRIMARY,))
        self.assertTrue(torch.equal(a[PRIMARY], b[PRIMARY]))

    def test_budget_gauge_support_and_matched_power(self):
        scores, diag = self.scores()
        for field in ('risk_support_expansion_fraction', 'unknown_source_risk_max',
                      'fully_negative_implication_violation', 'matched_previous_unmatchable'):
            self.assertEqual(diag[field], 0)
        self.assertLessEqual(diag['risk_increase_max'], 1e-5)
        self.assertLessEqual(diag['class_budget_max_error'], 1e-10)
        self.assertLessEqual(diag['matched_previous_norm_relative_error'], 1e-10)
        for name in (PRIMARY, GUARD_ONLY, CLASS_MEAN, MATCHED_OLD):
            delta = scores[name]-scores[OBSERVATION_MEAN]
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))

    def test_invalid_stencils_and_methods_rejected(self):
        crop = self.fine[0]
        bad = CachedCrop(crop.evidence, crop.margins, crop.indices, -crop.coefficients)
        with self.assertRaises(ValueError):
            footprint_interval((bad,), self.valid)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
