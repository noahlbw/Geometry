import math
import unittest

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop
from dinotool.rival_omission_alias import (CLASS_MEAN, GUARD_ONLY, MATCHED_OLD, METHODS,
    OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, omission_deltas, omission_risks,
    omission_scores, rival_omission_shift)


class RivalOmissionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3706)
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
        return omission_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_omission_matches_each_literal_remaining_mean(self):
        evidence = torch.randn(5, 3, 20, dtype=torch.float64)
        full = evidence.logsumexp(-1)-math.log(20)
        expected = torch.stack([torch.cat((evidence[..., :k], evidence[..., k+1:]), -1).logsumexp(-1)
            -math.log(19)-full for k in range(20)], -1)
        torch.testing.assert_close(omission_deltas(evidence), expected, atol=1e-12, rtol=0)

    def test_extreme_dominant_word_remains_finite(self):
        evidence = torch.tensor([[[1000., -1000., -999.], [-1000., 1000., 1000.]]])
        result = omission_deltas(evidence)
        self.assertTrue(bool(torch.isfinite(result).all()))
        expected = torch.logsumexp(torch.tensor([-1000., -999.], dtype=torch.float64), 0)
        expected -= math.log(2)+1000.-math.log(3)
        self.assertAlmostEqual(float(result[0, 0, 0]), float(expected), places=10)

    def test_uniform_words_have_zero_counterfactual_change(self):
        evidence = torch.zeros(5, 3, 20)
        torch.testing.assert_close(omission_deltas(evidence), torch.zeros_like(evidence).double(), atol=1e-12, rtol=0)

    def test_constant_class_offsets_do_not_change_omission(self):
        evidence = torch.randn(5, 3, 4, dtype=torch.float64)
        torch.testing.assert_close(omission_deltas(evidence+torch.randn(5, 3, 1)), omission_deltas(evidence),
            atol=1e-12, rtol=0)

    def test_one_word_identity_is_shared_across_the_whole_stencil(self):
        evidence = torch.tensor([[[10., 0.]], [[0., 10.]]])
        ids, coefficients = torch.tensor([[0, 1]]), torch.tensor([[.5, .5]])
        crop = CachedCrop(evidence, torch.zeros(2, 2, 1), ids, coefficients)
        shift, complete = rival_omission_shift((crop,), torch.tensor([True]))
        delta = omission_deltas(evidence)
        expected = delta.mean(0).amin(-1).clamp_max(0.)
        torch.testing.assert_close(shift[0], expected, atol=1e-12, rtol=0)
        self.assertGreater(float(shift[0, 0]), float(delta.amin(-1).mean()))
        self.assertTrue(bool(complete.all()))

    def test_multiple_crops_share_the_same_omitted_word(self):
        first = self.fine[0]
        second = CachedCrop(first.evidence.flip(-1), first.margins, first.indices, first.coefficients*.5)
        first = CachedCrop(first.evidence, first.margins, first.indices, first.coefficients*.5)
        shift, complete = rival_omission_shift((first, second), self.valid)
        expected = sum((omission_deltas(c.evidence)[c.indices]*c.coefficients.double()[..., None, None]).sum(1)
            for c in (first, second)).amin(-1).clamp_max(0.)
        torch.testing.assert_close(shift[complete], expected[complete], atol=1e-12, rtol=0)

    def test_missing_coverage_is_neutral(self):
        crop = self.fine[0]
        partial = CachedCrop(crop.evidence, crop.margins, crop.indices, crop.coefficients*.5)
        shift, complete = rival_omission_shift((partial,), self.valid)
        self.assertFalse(bool(complete.any()))
        wide, fine = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        risk, _, _ = omission_risks(wide, fine, shift, complete, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk.abs().max()), 0)

    def test_zero_weight_contributor_does_not_affect_probe(self):
        evidence = torch.tensor([[[1., 1.]], [[1000., -1000.]]])
        crop = CachedCrop(evidence, torch.zeros(2, 2, 1), torch.tensor([[0, 1]]), torch.tensor([[1., 0.]]))
        shift, _ = rival_omission_shift((crop,), torch.tensor([True]))
        self.assertEqual(float(shift.abs().max()), 0)

    def test_zero_shift_recovers_previous_risk(self):
        wide, fine = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        risk, previous, guard = omission_risks(wide, fine, torch.zeros(4, 3), self.valid,
            self.parents, self.canonical, self.valid)
        torch.testing.assert_close(risk, previous.double(), atol=0, rtol=0)
        self.assertTrue(torch.equal(guard, previous))

    def test_dominant_rival_support_can_withdraw_only_that_rival(self):
        wide, fine = torch.ones(4, 12, 3), -torch.ones(4, 12, 3)
        shift = torch.zeros(4, 3)
        shift[:, 2] = -2
        risk, previous, _ = omission_risks(wide, fine, shift, self.valid,
            self.parents, self.canonical, self.valid)
        self.assertGreater(float(risk[0, 1, 1]), 0)
        self.assertEqual(float(risk[0, 1, 2]), 0)
        self.assertTrue(bool((risk <= previous.double()).all()))
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)

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

    def test_equal_views_receive_no_action(self):
        self.fine = self.wide
        values, _ = self.scores()
        self.assertTrue(torch.equal(values[PRIMARY], values[OBSERVATION_MEAN]))

    def test_geometry_is_not_a_negative_semantic_teacher(self):
        values, _ = self.scores((PRIMARY,))
        alternate, _ = omission_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, methods=(PRIMARY,))
        self.assertTrue(torch.equal(values[PRIMARY], alternate[PRIMARY]))

    def test_budget_gauge_subset_and_matched_norm(self):
        values, stats = self.scores()
        for field in ('risk_support_expansion_fraction', 'unknown_source_risk_max', 'matched_previous_unmatchable'):
            self.assertEqual(stats[field], 0)
        self.assertLessEqual(stats['risk_increase_max'], 1e-5)
        self.assertLessEqual(stats['class_budget_max_error'], 1e-10)
        self.assertLessEqual(stats['matched_previous_norm_relative_error'], 1e-10)
        for name in (PRIMARY, GUARD_ONLY, CLASS_MEAN, MATCHED_OLD):
            delta = values[name]-values[OBSERVATION_MEAN]
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))

    def test_invalid_evidence_stencils_and_methods_are_rejected(self):
        with self.assertRaises(ValueError):
            omission_deltas(torch.ones(3, 4, 1))
        with self.assertRaises(ValueError):
            omission_deltas(torch.full((3, 4, 2), torch.nan))
        crop = self.fine[0]
        with self.assertRaises(ValueError):
            rival_omission_shift((CachedCrop(crop.evidence, crop.margins, crop.indices, -crop.coefficients),), self.valid)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
