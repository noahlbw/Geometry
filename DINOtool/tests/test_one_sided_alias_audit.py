import unittest
from unittest.mock import patch

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.one_sided_alias_audit import (CLASS_MEAN, DIRECT_MATCHED, EQUAL_MEAN,
    MATCHED_CLASS_MEAN, MATCHED_SHUFFLES, METHODS, OBSERVATION_MEAN, PRIMARY,
    collapse_rivals, predict_image, scores)
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, reciprocal_scores)
from dinotool.one_sided_alias_audit import HARD
from dinotool.rival_alias_fast import CachedCrop


class OneSidedAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4106)
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

    def score(self, methods=METHODS[3:]):
        return scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_original_positive_soft_and_hard_are_bitwise_exact(self):
        actual, _ = self.score()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(PREVIOUS_MEAN, PREVIOUS_SOFT, PREVIOUS_HARD))
        for new, old in ((OBSERVATION_MEAN, PREVIOUS_MEAN), (PRIMARY, PREVIOUS_SOFT), (HARD, PREVIOUS_HARD)):
            self.assertTrue(torch.equal(actual[new], previous[old]), new)

    def test_all_singletons_match_combined_controls(self):
        combined, _ = self.score()
        for name in METHODS[3:]:
            single, _ = self.score((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_primary_does_not_compute_counterfactual_controls(self):
        with patch('dinotool.one_sided_alias_audit.continuous_controls', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.match_previous', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.collapse_rivals', side_effect=AssertionError):
            self.score((PRIMARY,))

    def test_geometry_is_not_a_semantic_truth_gate(self):
        actual, _ = self.score((PRIMARY,))
        changed, _ = scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, methods=(PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], changed[PRIMARY]))

    def test_identical_sources_produce_no_word_action(self):
        self.fine = self.wide
        values, _ = self.score()
        self.assertTrue(torch.equal(values[PRIMARY], values[OBSERVATION_MEAN]))

    def test_matched_controls_preserve_class_field_norm_and_gauge(self):
        values, diagnostics = self.score()
        reference = values[PRIMARY]-values[OBSERVATION_MEAN]
        for name in (MATCHED_CLASS_MEAN, *MATCHED_SHUFFLES, DIRECT_MATCHED):
            delta = values[name]-values[OBSERVATION_MEAN]
            torch.testing.assert_close(delta[self.valid].norm(), reference[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertEqual(diagnostics['matched_unmatchable'], 0)
        self.assertLessEqual(diagnostics['matched_norm_relative_error'], 1e-10)

    def test_class_and_rival_risk_budgets_are_preserved(self):
        _, diagnostics = self.score()
        self.assertLessEqual(diagnostics['class_budget_max_error'], 1e-10)
        self.assertLessEqual(diagnostics['rival_budget_max_error'], 1e-10)

    def test_rival_collapse_removes_identity_without_self_actions(self):
        risk = torch.rand(4, 12, 3)
        risk.scatter_(-1, self.parents[None, :, None].expand(4, -1, 1), 0.)
        risk[:, self.canonical] = 0
        changed, error = collapse_rivals(risk, self.parents)
        self.assertLessEqual(error, 1e-10)
        self.assertEqual(float(changed[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(changed.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)
        for a, parent in enumerate(self.parents.tolist()):
            alternatives = [c for c in range(3) if c != parent]
            self.assertTrue(torch.equal(changed[:, a, alternatives[0]], changed[:, a, alternatives[1]]))

    def test_invalid_words_canonical_and_scores_are_protected(self):
        values, diagnostics = self.score()
        self.assertEqual(diagnostics['canonical_risk_max'], 0)
        self.assertLessEqual(diagnostics['positive_directed_delta_max'], 1e-12)
        for name in (PRIMARY, CLASS_MEAN, *MATCHED_SHUFFLES):
            self.assertTrue(torch.equal(values[name][~self.valid], values[OBSERVATION_MEAN][~self.valid]))

    def test_equal_fusion_uses_identical_corrected_observations(self):
        values, _ = self.score()
        # H=.5I on valid tokens: the exact reader becomes equal fusion there.
        torch.testing.assert_close(values[PRIMARY][self.valid], values[EQUAL_MEAN][self.valid], atol=1e-12, rtol=0)

    def test_prediction_only_delegates_to_existing_visual_source(self):
        sentinel = object()
        with patch('dinotool.one_sided_alias_audit.source_predict', return_value=sentinel) as source:
            actual = predict_image(None, None, {}, None, {}, methods=(PRIMARY,))
        self.assertIs(actual, sentinel)
        self.assertEqual(source.call_args.kwargs['methods'], (PRIMARY,))

    def test_invalid_methods_and_rival_layout_fail(self):
        with self.assertRaises(ValueError):
            self.score(('Unfrozen',))
        with self.assertRaises(ValueError):
            collapse_rivals(torch.zeros(4, 12, 1), self.parents)


if __name__ == '__main__':
    unittest.main()
