import unittest
from unittest.mock import patch

import torch

from dinotool import alias_budget_attribution as budget
from dinotool import competitive_excess_alias as current
from dinotool import one_sided_alias_audit as audit
from dinotool import own_peer_alias as peer
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.rival_alias_fast import CachedCrop


class CompetitiveExcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6626)
        self.members = torch.arange(12).reshape(3, 4)
        self.parents = torch.arange(3).repeat_interleave(4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wide, self.fine = [], []
        for observations in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            observations.append(CachedCrop(evidence, contribution_margins(evidence),
                ids, torch.tensor([[.3, .7]]*4)))

    def args(self):
        return (self.local, self.operator, self.broad, self.field, self.wide, self.fine,
            self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents)

    def test_jointly_positive_overstatement_is_actionable(self):
        wide = torch.full((4, 12, 3), 2.)
        fine = torch.ones_like(wide)
        actual = current.excess_risk(wide, fine, self.parents, self.canonical, self.valid)
        self.assertAlmostEqual(float(actual[0, 1, 1]), 1-torch.exp(torch.tensor(-1.)).item(), places=7)
        old = audit.native_risk(wide, fine, fine, self.parents, self.canonical, self.valid, audit.CONFIG)
        self.assertEqual(float(old.sum()), 0.)

    def test_exact_log_advantage_cap_before_protection(self):
        wide = torch.tensor([[[2., -1., 3., 1.]]], dtype=torch.float64)
        fine = torch.tensor([[[1., -2., 4., -2.]]], dtype=torch.float64)
        risk = current.competitive_excess(wide, fine)
        changed = wide+torch.log1p(-risk)
        expected = torch.where(wide > 0, torch.minimum(wide, fine), wide)
        torch.testing.assert_close(changed, expected, atol=1e-12, rtol=0)

    def test_equal_or_stronger_fine_is_neutral(self):
        wide = torch.randn(4, 12, 3)
        for fine in (wide, wide+1):
            self.assertEqual(float(current.competitive_excess(wide, fine).sum()), 0.)

    def test_same_alias_can_differ_by_current_rival(self):
        wide, fine = torch.zeros(4, 12, 3), torch.zeros(4, 12, 3)
        wide[:, 1, 1:] = 2.
        fine[:, 1, 1], fine[:, 1, 2] = 1., 3.
        risk = current.excess_risk(wide, fine, self.parents, self.canonical, self.valid)
        self.assertGreater(float(risk[0, 1, 1]), 0.)
        self.assertEqual(float(risk[0, 1, 2]), 0.)

    def test_canonical_self_and_padding_protected(self):
        risk = current.excess_risk(torch.ones(4, 12, 3), -torch.ones(4, 12, 3),
            self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].sum()), 0.)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).sum()), 0.)
        self.assertEqual(float(risk[~self.valid].sum()), 0.)

    def test_uniform_words_reduce_to_same_formula_class_control(self):
        broad, fine = torch.tensor([[4., 2., 0.]]), torch.tensor([[2., 1., 0.]])
        margins = lambda x: x[:, self.parents, None]-x[:, None]
        valid = torch.tensor([True])
        risk = current.excess_risk(margins(broad), margins(fine), self.parents, self.canonical, valid)
        expected = current.class_excess_risk(broad, fine, valid)[:, self.parents].clone()
        expected[:, self.canonical] = 0.
        self.assertTrue(torch.equal(risk, expected))

    def test_previous_controls_replay_exactly(self):
        actual, _ = current.scores(*self.args())
        old, _ = audit.scores(*self.args(), methods=(audit.PRIMARY, audit.HARD, audit.OBSERVATION_MEAN))
        prior, _ = peer.scores(*self.args(), methods=(peer.PRIMARY,))
        pooled, _ = budget.scores(*self.args(), methods=(budget.POOLED,))
        for new, original in ((current.PREVIOUS_SOFT, audit.PRIMARY),
                              (current.PREVIOUS_HARD, audit.HARD),
                              (current.OBSERVATION_MEAN, audit.OBSERVATION_MEAN)):
            self.assertTrue(torch.equal(actual[new], old[original]))
        self.assertTrue(torch.equal(actual[current.PREVIOUS_PEER], prior[peer.PRIMARY]))
        self.assertTrue(torch.equal(actual[current.POOLED], pooled[budget.POOLED]))

    def test_all_singletons_match_and_zero_risk_recovers_observation(self):
        actual, _ = current.scores(*self.args())
        for name in current.METHODS[3:]:
            single, _ = current.scores(*self.args(), methods=(name,))
            self.assertTrue(torch.equal(actual[name], single[name]), name)
        with patch('dinotool.competitive_excess_alias.excess_risk',
                   return_value=torch.zeros(4, 12, 3, dtype=torch.float64)):
            neutral, _ = current.scores(*self.args(), methods=(current.PRIMARY, current.OBSERVATION_MEAN))
        self.assertTrue(torch.equal(neutral[current.PRIMARY], neutral[current.OBSERVATION_MEAN]))

    def test_matched_controls_preserve_norm_gauge_and_padding(self):
        values, diagnostics = current.scores(*self.args())
        positive = values[current.OBSERVATION_MEAN]
        primary = values[current.PRIMARY]-positive
        for name in (current.MATCHED_POOLED, current.MATCHED_EXCESS,
                     current.OLD_NAMES[audit.MATCHED_CLASS_MEAN],
                     *(current.OLD_NAMES[m] for m in audit.MATCHED_SHUFFLES)):
            delta = values[name]-positive
            torch.testing.assert_close(delta[self.valid].norm(), primary[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertEqual(float(delta[~self.valid].abs().max()), 0.)
        self.assertEqual(diagnostics['calibration_matched_unmatchable'], 0.)

    def test_primary_skips_previous_sources_and_class_controls(self):
        with patch('dinotool.competitive_excess_alias.peer.scores', side_effect=AssertionError), \
             patch('dinotool.competitive_excess_alias.class_excess_risk', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.native_risk', side_effect=AssertionError):
            current.scores(*self.args(), methods=(current.PRIMARY,))

    def test_extreme_finite_source_and_invalid_input(self):
        risk = current.excess_risk(torch.full((4, 12, 3), 1000.), torch.full((4, 12, 3), -1000.),
            self.parents, self.canonical, self.valid)
        directed, _ = audit.fixed_slot_action(self.wide, self.members, risk, self.valid)
        self.assertTrue(bool(torch.isfinite(directed).all()))
        with self.assertRaises(ValueError):
            current.competitive_excess(torch.full((1, 2, 2), float('nan')), torch.zeros(1, 2, 2))
        with self.assertRaises(ValueError):
            current.scores(*self.args(), methods=('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
