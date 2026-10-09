import unittest

import torch

from dinotool.reciprocal_alias_admission import (CLASS_MEAN, METHODS, OBSERVATION_MEAN,
    ONE_SIDED_HARD, PRIMARY, SHUFFLES, continuous_controls, fixed_slot_action,
    reciprocal_risks, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop
from dinotool.supported_positive_alias import (NO_GEOMETRY_RISK as OLD_HARD,
    OBSERVATION_MEAN as OLD_MEAN, joint_scores)


class ReciprocalAdmissionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6106)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.parents = torch.arange(3).repeat_interleave(4)
        self.valid = torch.tensor([True, True, True, False])
        self.coords = torch.randn(4, 2)
        self.relation = torch.eye(4)
        self.operator = torch.diag(self.valid.double()*.5)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wm = torch.randn(4, 12, 3)
        self.fm = torch.randn(4, 12, 3)
        ids = torch.arange(4)[:, None]
        coeff = torch.ones(4, 1)
        self.wide = [CachedCrop(torch.randn(4, 3, 4), self.wm, ids, coeff)]
        self.fine = [CachedCrop(torch.randn(4, 3, 4), self.fm, ids, coeff)]

    def risks(self):
        return reciprocal_risks(self.wm, self.fm, self.parents, self.canonical, self.valid)

    def scores(self, methods=METHODS[3:]):
        return reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_risk_protects_canonical_self_and_padding(self):
        for risk in self.risks():
            self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
            self.assertEqual(float(risk[~self.valid].abs().max()), 0)
            self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)

    def test_reciprocity_is_source_swap_equivariant(self):
        first, second = self.risks()
        reverse = reciprocal_risks(self.fm, self.wm, self.parents, self.canonical, self.valid)
        self.assertTrue(torch.equal(first, reverse[1]))
        self.assertTrue(torch.equal(second, reverse[0]))
        self.assertEqual(float((first*second).max()), 0)

    def test_alias_can_be_attenuated_against_only_one_rival(self):
        self.wm[0, 1] = torch.tensor([1., 2., 2.])
        self.fm[0, 1] = torch.tensor([1., -2., 2.])
        first, _ = self.risks()
        self.assertEqual(float(first[0, 1, 1]), .5)
        self.assertEqual(float(first[0, 1, 2]), 0)

    def test_fixed_slots_matches_independent_weighted_logsumexp(self):
        risk = self.risks()[0]
        actual, _ = fixed_slot_action(self.wide, self.members, risk, self.valid, chunk=2)
        e = self.wide[0].evidence.double()
        weights = 1-risk[:, self.members].double()
        expected = (e[..., None]+weights.log()).logsumexp(2)-e.logsumexp(-1)[..., None]
        expected.masked_fill_((weights == 1).all(2), 0)
        expected.masked_fill_(~self.valid[:, None, None], 0)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_removing_low_response_never_rewards_class(self):
        self.wide[0].evidence.fill_(0)
        self.wide[0].evidence[0, 0, 1] = -10
        risk = torch.zeros(4, 12, 3)
        risk[0, 1, 1] = 1
        delta, _ = fixed_slot_action(self.wide, self.members, risk, self.valid)
        self.assertLess(float(delta[0, 0, 1]), 0)

    def test_extreme_logits_use_stable_log_fallback_not_silent_clamp(self):
        self.wide[0].evidence[0, 0] = torch.tensor([-1500., 1500., 1500., 1500.])
        risk = torch.zeros(4, 12, 3)
        risk[0, 1:4, 1] = 1
        delta, diag = fixed_slot_action(self.wide, self.members, risk, self.valid)
        expected = -1500-torch.logsumexp(self.wide[0].evidence[0, 0].double(), 0)
        self.assertAlmostEqual(float(delta[0, 0, 1]), float(expected), places=10)
        self.assertGreater(diag['mass_log_fallbacks'], 0)

    def test_zero_risk_is_bitwise_zero(self):
        delta, diag = fixed_slot_action(self.wide, self.members, torch.zeros_like(self.wm), self.valid)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertEqual(diag['mass_log_fallbacks'], 0)

    def test_continuous_class_and_shuffle_controls_preserve_pair_budget(self):
        for risk in self.risks():
            controls, error = continuous_controls(risk, self.members, self.canonical)
            self.assertLess(error, 1e-12)
            for value in controls.values():
                self.assertEqual(float(value[:, self.canonical].abs().max()), 0)
                torch.testing.assert_close(value[:, self.members].double().sum(2),
                    risk[:, self.members].double().sum(2), atol=1e-12, rtol=0)
            for name in SHUFFLES:
                torch.testing.assert_close(controls[name][:, self.members].sort(2).values,
                    risk[:, self.members].sort(2).values, atol=0, rtol=0)

    def test_strong_previous_endpoints_replay_bitwise(self):
        current, _ = self.scores()
        previous, _ = joint_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid, self.members,
            self.canonical, self.parents, methods=(OLD_MEAN, OLD_HARD))
        self.assertTrue(torch.equal(current[OBSERVATION_MEAN], previous[OLD_MEAN]))
        self.assertTrue(torch.equal(current[ONE_SIDED_HARD], previous[OLD_HARD]))

    def test_no_contradiction_is_same_information_identity(self):
        self.fm.copy_(self.wm)
        actual, _ = self.scores()
        self.assertTrue(torch.equal(actual[PRIMARY], actual[OBSERVATION_MEAN]))

    def test_primary_is_source_swap_equivariant(self):
        actual, _ = self.scores((PRIMARY,))
        reverse, _ = reciprocal_scores(self.local, self.operator, self.field, self.broad,
            self.fine, self.wide, self.coords, self.relation, self.valid, self.members,
            self.canonical, self.parents, methods=(PRIMARY,))
        torch.testing.assert_close(actual[PRIMARY], reverse[PRIMARY], atol=1e-12, rtol=0)

    def test_singleton_matches_simultaneous_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            actual, _ = self.scores((name,))
            self.assertTrue(torch.equal(actual[name], combined[name]), name)

    def test_alias_action_preserves_class_gauge_and_padding(self):
        scores, diag = self.scores()
        torch.testing.assert_close((scores[PRIMARY]-scores[OBSERVATION_MEAN]).sum(-1),
            torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(scores[PRIMARY][~self.valid], scores[OBSERVATION_MEAN][~self.valid]))
        self.assertLessEqual(diag['positive_directed_delta_max'], 1e-12)

    def test_missing_survivor_and_invalid_method_rejected(self):
        with self.assertRaises(ValueError):
            fixed_slot_action(self.wide, self.members, torch.ones_like(self.wm), self.valid)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
