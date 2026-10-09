import unittest
from unittest.mock import patch

import torch

from dinotool import alias_budget_attribution as current
from dinotool import one_sided_alias_audit as audit
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.reciprocal_alias_admission import fixed_slot_action
from dinotool.rival_alias_fast import CachedCrop


class AliasBudgetAttributionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6316)
        self.members = torch.arange(12).reshape(3, 4)
        self.parents = torch.arange(3).repeat_interleave(4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            target.append(CachedCrop(evidence, contribution_margins(evidence), ids,
                torch.tensor([[.3, .7]]*4)))

    def args(self):
        return (self.local, self.operator, self.broad, self.field, self.wide, self.fine,
            self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents)

    def score(self, methods=current.METHODS[3:]):
        return current.scores(*self.args(), methods=methods)

    def test_all_previous_endpoints_replay_exactly(self):
        actual, _ = self.score()
        old, _ = audit.scores(*self.args())
        for old_name, new_name in current.OLD_NAMES.items():
            self.assertTrue(torch.equal(actual[new_name], old[old_name]), new_name)

    def test_all_singletons_match_combined(self):
        combined, _ = self.score()
        for name in current.METHODS[3:]:
            singleton, _ = self.score((name,))
            self.assertTrue(torch.equal(combined[name], singleton[name]), name)

    def test_class_risk_only_uses_pooled_fields(self):
        broad = torch.tensor([[2., 0.], [0., 2.], [1., 0.]])
        fine = torch.tensor([[0., 4.], [0., 3.], [0., 3.]])
        risk = current.pooled_class_risk(broad, fine, torch.tensor([True, True, False]))
        self.assertAlmostEqual(float(risk[0, 0, 1]), 4/6)
        self.assertEqual(float(risk[0, 1, 0]), 0.)
        self.assertTrue(torch.equal(risk[1:], torch.zeros_like(risk[1:])))
        self.assertTrue(torch.equal(risk.diagonal(dim1=1, dim2=2), torch.zeros(3, 2, dtype=torch.float64)))

    def test_deployable_control_never_reads_primary_word_risk_or_dose(self):
        with patch('dinotool.alias_budget_attribution.audit.scores', side_effect=AssertionError), \
             patch('dinotool.alias_budget_attribution.match_previous', side_effect=AssertionError):
            self.score((current.POOLED,))

    def test_analytic_action_equals_explicit_alias_weights(self):
        risks = current.pooled_class_risk(self.broad, self.field, self.valid)
        expected_risk = risks[:, self.parents].clone()
        expected_risk[:, self.canonical] = 0.
        expected, _ = fixed_slot_action(self.wide, self.members, expected_risk, self.valid)
        actual = current.uniform_class_action(self.wide, self.members, self.canonical, risks, self.valid)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_underflow_keeps_canonical_log_mass_finite(self):
        evidence = torch.tensor([[[-1000., 0.], [-1000., 0.]]], dtype=torch.float64)
        crop = CachedCrop(evidence, contribution_margins(evidence), torch.tensor([[0]]), torch.ones(1, 1))
        members, canonical = torch.arange(4).reshape(2, 2), torch.tensor([0, 2])
        risk = torch.tensor([[[0., 1.], [1., 0.]]])
        value = current.uniform_class_action([crop], members, canonical, risk, torch.tensor([True]))
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(float(value[0, 0, 1]), -1000.)
        self.assertEqual(float(value[0, 1, 0]), -1000.)
        self.assertEqual(float(value[0, 0, 0]), 0.)

    def test_nonfirst_canonical_slots_and_overlap_are_supported(self):
        canonical = self.members[:, 2]
        risk = current.pooled_class_risk(self.broad, self.field, self.valid)
        alias_risk = risk[:, self.parents].clone()
        alias_risk[:, canonical] = 0.
        expected, _ = fixed_slot_action(self.wide, self.members, alias_risk, self.valid)
        actual = current.uniform_class_action(self.wide, self.members, canonical, risk, self.valid)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_identical_class_fields_are_neutral(self):
        self.field = self.broad
        actual, _ = self.score((current.POOLED, current.OBSERVATION_MEAN))
        self.assertTrue(torch.equal(actual[current.POOLED], actual[current.OBSERVATION_MEAN]))

    def test_invalid_positions_and_class_gauge_are_protected(self):
        actual, _ = self.score()
        for name in (current.POOLED, current.MATCHED_POOLED):
            delta = actual[name]-actual[current.OBSERVATION_MEAN]
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_matched_control_preserves_primary_word_write_norm(self):
        actual, diagnostics = self.score()
        positive = actual[current.OBSERVATION_MEAN]
        self.assertEqual(diagnostics['pooled_matched_unmatchable'], 0.)
        torch.testing.assert_close((actual[current.MATCHED_POOLED]-positive)[self.valid].norm(),
            (actual[current.PRIMARY]-positive)[self.valid].norm(), atol=1e-12, rtol=0)

    def test_primary_does_not_run_class_controls(self):
        with patch('dinotool.alias_budget_attribution.pooled_class_risk', side_effect=AssertionError):
            self.score((current.PRIMARY,))

    def test_invalid_methods_and_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.score(('Unfrozen',))
        with self.assertRaises(ValueError):
            current.pooled_class_risk(self.broad, self.field[:, :1], self.valid)


if __name__ == '__main__':
    unittest.main()
