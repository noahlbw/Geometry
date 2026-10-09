import unittest
from unittest.mock import patch

import torch

from dinotool import one_sided_alias_audit as audit
from dinotool import witness_cap_alias as cap
from dinotool import witness_cap_attribution as current
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.rival_alias_fast import CachedCrop


class WitnessCapAttributionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6306)
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

    def test_existing_cap_endpoints_are_bitwise_exact(self):
        actual, _ = self.score()
        old, _ = cap.witness_scores(*self.args())
        pairs = ((current.PRIMARY, cap.PRIMARY), (current.OBSERVATION_MEAN, cap.OBSERVATION_MEAN),
            (current.PREVIOUS_SOFT, cap.OLD_SOFT), (current.OLD_NAMES[audit.HARD], cap.OLD_HARD),
            (current.OLD_NAMES[audit.CLASS_MEAN], cap.CLASS_MEAN),
            (current.OLD_NAMES[audit.SHUFFLED_WRITE], cap.SHUFFLED_WRITE),
            *zip((current.OLD_NAMES[m] for m in audit.SHUFFLES), cap.SHUFFLES))
        for new, previous in pairs:
            self.assertTrue(torch.equal(actual[new], old[previous]), new)

    def test_all_singletons_match_combined(self):
        actual, _ = self.score()
        for name in current.METHODS[3:]:
            single, _ = self.score((name,))
            self.assertTrue(torch.equal(actual[name], single[name]), name)

    def test_primary_skips_control_work(self):
        with patch('dinotool.one_sided_alias_audit.continuous_controls', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.match_previous', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.collapse_rivals', side_effect=AssertionError):
            self.score((current.PRIMARY,))

    def test_previous_soft_uses_default_source_not_cap(self):
        with patch('dinotool.witness_cap_attribution.cap_risk', side_effect=AssertionError):
            self.score((current.PREVIOUS_SOFT,))

    def test_default_audit_path_is_unchanged(self):
        baseline, _ = audit.scores(*self.args(), methods=(audit.PRIMARY,))
        explicit, _ = audit.scores(*self.args(), methods=(audit.PRIMARY,), risk_builder=None)
        self.assertTrue(torch.equal(baseline[audit.PRIMARY], explicit[audit.PRIMARY]))

    def test_matched_controls_preserve_word_norm_and_gauge(self):
        actual, diagnostics = self.score()
        reference = actual[current.PRIMARY]-actual[current.OBSERVATION_MEAN]
        names = (audit.MATCHED_CLASS_MEAN, *audit.MATCHED_SHUFFLES,
            audit.MATCHED_RIVAL_COLLAPSED, audit.MATCHED_SHUFFLED_WRITE, audit.DIRECT_MATCHED)
        for name in names:
            delta = actual[current.OLD_NAMES[name]]-actual[current.OBSERVATION_MEAN]
            torch.testing.assert_close(delta[self.valid].norm(), reference[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertEqual(diagnostics['matched_unmatchable'], 0)

    def test_identical_sources_have_no_attenuation(self):
        self.fine = self.wide
        actual, _ = self.score()
        self.assertTrue(torch.equal(actual[current.PRIMARY], actual[current.OBSERVATION_MEAN]))

    def test_protection_and_budget_bounds(self):
        actual, diagnostics = self.score()
        self.assertEqual(diagnostics['canonical_risk_max'], 0)
        self.assertLessEqual(diagnostics['positive_directed_delta_max'], 1e-12)
        self.assertLessEqual(diagnostics['class_budget_max_error'], 1e-10)
        self.assertLessEqual(diagnostics['rival_budget_max_error'], 1e-10)
        for name in (current.PRIMARY, current.OLD_NAMES[audit.MATCHED_CLASS_MEAN]):
            self.assertTrue(torch.equal(actual[name][~self.valid], actual[current.OBSERVATION_MEAN][~self.valid]))

    def test_half_identity_geometry_equals_equal_logit_mean(self):
        actual, _ = self.score()
        torch.testing.assert_close(actual[current.PRIMARY][self.valid],
            actual[current.OLD_NAMES[audit.EQUAL_MEAN]][self.valid], atol=1e-12, rtol=0)

    def test_invalid_method_fails(self):
        with self.assertRaises(ValueError):
            self.score(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
