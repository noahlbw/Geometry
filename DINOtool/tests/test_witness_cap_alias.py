import unittest

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as OLD_MEAN,
    ONE_SIDED_HARD as OLD_HARD, ONE_SIDED_SOFT as OLD_SOFT, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop
from dinotool.witness_cap_alias import (CLASS_MEAN, MATCHED_OLD, METHODS,
    OBSERVATION_MEAN, OLD_HARD as REPLAY_HARD, OLD_SOFT as REPLAY_SOFT,
    PRIMARY, match_previous, witness_risk, witness_scores)


class WitnessCapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2606)
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
        return witness_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_exact_likelihood_cap_on_existing_support(self):
        wide = torch.ones(4, 12, 3)
        fine = -torch.arange(144).reshape(4, 12, 3).float()/20
        risk, previous = witness_risk(wide, fine, self.parents, self.canonical, self.valid)
        expected = torch.where(previous > 0, -torch.expm1(fine.double()), 0.)
        self.assertTrue(torch.equal(risk, expected))
        self.assertTrue(torch.equal(risk > 0, previous > 0))
        self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))

    def test_wrong_wide_amplitude_cannot_protect_an_eligible_word(self):
        wide = torch.ones(4, 12, 3)
        fine = -torch.ones_like(wide)
        low, old_low = witness_risk(wide, fine, self.parents, self.canonical, self.valid)
        high, old_high = witness_risk(wide*100, fine, self.parents, self.canonical, self.valid)
        self.assertTrue(torch.equal(low, high))
        self.assertGreater(float((old_low-old_high).max()), .4)

    def test_stronger_negative_witness_increases_risk(self):
        wide = torch.ones(4, 12, 3)
        weak, _ = witness_risk(wide, -wide*.1, self.parents, self.canonical, self.valid)
        strong, _ = witness_risk(wide, -wide*2, self.parents, self.canonical, self.valid)
        self.assertTrue(bool((strong >= weak).all()))
        self.assertGreater(float((strong-weak).max()), .7)

    def test_canonical_self_invalid_and_agreement_protected(self):
        wide = torch.ones(4, 12, 3)
        fine = -wide
        fine[0, 1, 2] = 1
        risk, _ = witness_risk(wide, fine, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)
        self.assertEqual(float(risk[0, 1, 2]), 0)
        self.assertGreater(float(risk[0, 1, 1]), .6)

    def test_previous_positive_soft_and_hard_replay_bitwise(self):
        current, _ = self.scores()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(OLD_MEAN, OLD_SOFT, OLD_HARD))
        for new, old in ((OBSERVATION_MEAN, OLD_MEAN), (REPLAY_SOFT, OLD_SOFT), (REPLAY_HARD, OLD_HARD)):
            self.assertTrue(torch.equal(current[new], previous[old]), new)

    def test_singleton_matches_all_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            single, _ = self.scores((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_equal_observations_produce_no_intervention(self):
        self.fine = self.wide
        scores, _ = self.scores()
        self.assertTrue(torch.equal(scores[PRIMARY], scores[OBSERVATION_MEAN]))

    def test_geometry_is_not_used_as_semantic_truth(self):
        original, _ = self.scores((PRIMARY,))
        changed, _ = witness_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, methods=(PRIMARY,))
        self.assertTrue(torch.equal(original[PRIMARY], changed[PRIMARY]))

    def test_class_gauge_mass_invalid_and_matched_power(self):
        scores, diag = self.scores()
        self.assertLessEqual(diag['class_budget_max_error'], 1e-10)
        self.assertEqual(diag['risk_support_change_fraction'], 0)
        self.assertEqual(diag['matched_previous_unmatchable'], 0)
        self.assertLessEqual(diag['matched_previous_norm_relative_error'], 1e-12)
        for name in (PRIMARY, CLASS_MEAN, MATCHED_OLD):
            delta = scores[name]-scores[OBSERVATION_MEAN]
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))

    def test_empty_matching_is_identity_and_impossible_matching_is_reported(self):
        zero = torch.zeros(4, 3, dtype=torch.float64)
        actual, diag = match_previous(zero, zero, self.valid)
        self.assertTrue(torch.equal(actual, zero))
        self.assertEqual(diag['matched_previous_unmatchable'], 0)
        actual, diag = match_previous(zero, torch.ones_like(zero), self.valid)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(diag['matched_previous_unmatchable'], 1)

    def test_tiny_and_extreme_negative_margins_are_finite(self):
        wide = torch.ones(4, 12, 3)
        fine = -wide*1e-10
        tiny, _ = witness_risk(wide, fine, self.parents, self.canonical, self.valid)
        huge, _ = witness_risk(wide, -wide*1000, self.parents, self.canonical, self.valid)
        self.assertGreater(float(tiny.max()), 0)
        self.assertTrue(bool(torch.isfinite(huge).all()))
        self.assertEqual(float(huge.max()), 1)

    def test_invalid_inputs_and_methods_rejected(self):
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))
        with self.assertRaises(ValueError):
            witness_risk(torch.ones(4, 12, 3), torch.full((4, 12, 3), torch.nan),
                self.parents, self.canonical, self.valid)


if __name__ == '__main__':
    unittest.main()
