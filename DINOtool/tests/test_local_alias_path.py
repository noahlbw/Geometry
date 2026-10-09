import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dinotool.local_alias_path import (CLASS_MEAN, DIRECT_MATCHED, LOCAL_ONLY, MATCHED_OLD,
    METHODS, OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, CapturedExecution,
    local_action, local_observations, local_scores, predict_image)
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.native_query_alias import native_risk
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop, sampled_cached


def source_probe(image, geometry, banks, vip, queries, work=None, *, methods,
                 cache=None, execution=None, tile_scorer=None, allowed_methods=None,
                 observation_method=None, diagnostic_fields=()):
    readout(None, None, 2.)
    execution.reader(vip)(vip, (None,))
    return tile_scorer(*work, methods=methods)


class LocalAliasPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3806)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.parents = torch.arange(3).repeat_interleave(4)
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.logits = torch.randn(4, 3, 4)
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            coefficients = torch.tensor([[.3, .7]]*4)
            target.append(CachedCrop(evidence, contribution_margins(evidence), ids, coefficients))
        self.fine_local = self.fine

    def scores(self, methods=METHODS[3:]):
        return local_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, self.logits, self.fine_local, methods=methods)

    def test_local_fixed_slot_action_matches_literal_weighted_lse(self):
        risk = torch.rand(4, 12, 3)*.8
        risk[~self.valid] = 0
        risk[:, self.canonical] = 0
        risk.scatter_(-1, self.parents[None, :, None].expand(4, -1, 1), 0)
        weights = 1-risk[:, self.members].double()
        expected = (self.logits.double()[..., None]+weights.log()).logsumexp(2)
        expected -= self.logits.double().logsumexp(-1)[..., None]
        expected[~self.valid] = 0
        actual, _ = local_action(self.logits, risk, self.valid, self.members)
        torch.testing.assert_close(actual, (expected-expected.transpose(-1, -2)).mean(-1), atol=1e-12, rtol=0)

    def test_zero_risk_has_exactly_zero_action(self):
        potential, _ = local_action(self.logits, torch.zeros(4, 12, 3), self.valid, self.members)
        self.assertEqual(float(potential.abs().max()), 0)

    def test_local_anchor_change_recovers_original_normal_equations(self):
        weights = torch.randn(4, 4, dtype=torch.float64)
        weights[~self.valid] = 0
        weights[:, ~self.valid] = 0
        gram = weights.T@weights
        operator = torch.linalg.solve(torch.eye(4, dtype=torch.float64)+gram, gram)
        delta = torch.randn(4, 3, dtype=torch.float64)
        delta[~self.valid] = 0
        baseline = self.local.double()+operator@(self.broad.double()-self.local.double())
        expected = torch.linalg.solve(torch.eye(4, dtype=torch.float64)+gram,
            self.local.double()+delta+gram@self.broad.double())
        torch.testing.assert_close(baseline+(torch.eye(4)-operator)@delta, expected, atol=1e-12, rtol=0)

    def test_existing_positive_soft_and_hard_replay(self):
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

    def test_reference_reuses_features_and_stencils(self):
        features = (torch.randn(1, 4, 7),)
        texts = torch.nn.functional.normalize(torch.randn(12, 7), dim=-1)
        values = local_observations(features, texts, self.members, self.fine)
        expected = ((features[0]@texts.T)[0]/.07)[:, self.members]
        self.assertTrue(torch.equal(values[0].evidence, expected))
        self.assertIs(values[0].indices, self.fine[0].indices)
        self.assertIs(values[0].coefficients, self.fine[0].coefficients)

    def test_capture_does_not_encode_or_change_feature_objects(self):
        state, values = {}, (torch.randn(1, 4, 7),)
        class Reader:
            calls = 0
            def __call__(self, observer, crops):
                self.calls += len(crops)
                return values
        class Execution:
            def reader(self, observer):
                return Reader()
        reader = CapturedExecution(Execution(), state).reader(None)
        self.assertIs(reader(None, (None,)), values)
        self.assertIs(state['fine'], values)
        self.assertEqual(reader.calls, 1)

    def test_predictor_captures_readout_tuple_and_reuses_same_bank(self):
        features, fine_features = torch.randn(1, 4, 7), (torch.randn(1, 4, 7),)
        texts = torch.nn.functional.normalize(torch.randn(12, 7), dim=-1)
        aliases = tuple('word'+str(i) for i in range(12))
        bank = SimpleNamespace(features=texts, alias_names=aliases, parent_indices=self.parents)
        query = SimpleNamespace(aliases=aliases, parents=self.parents)
        class Reader:
            calls = 0
            def __call__(self, observer, crops):
                self.calls += 1
                return fine_features
        class Execution:
            def reader(self, observer):
                return Reader()
        work = (self.local, self.operator, self.broad, self.field, self.wide, self.fine,
                self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents)
        with (patch('dinotool.local_alias_path.original_readout', return_value=(features, {})) as read,
              patch('dinotool.local_alias_path.source_predict', source_probe)):
            actual, _ = predict_image(None, None, {'p': bank}, None, {'p': query}, work,
                                      methods=(PRIMARY,), execution=Execution())
        read.assert_called_once_with(None, None, 2.)
        normalized = torch.nn.functional.normalize(texts.float(), dim=-1)
        logits = ((features.float()@normalized.T)[0]/.07)[:, self.members]
        expected, _ = local_scores(*work, logits,
            local_observations(fine_features, normalized, self.members, self.fine), methods=(PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], expected[PRIMARY]))

    def test_no_local_contradiction_recovers_previous_soft(self):
        margins = contribution_margins(self.logits)
        ids = torch.arange(4)[:, None]
        self.fine_local = (CachedCrop(self.logits, margins, ids, torch.ones(4, 1)),)
        values, _ = self.scores()
        self.assertTrue(torch.equal(values[PRIMARY], values[OLD_SOFT]))

    def test_local_and_wide_factorial_reconstructs_primary(self):
        values, _ = self.scores()
        torch.testing.assert_close(values[PRIMARY]-values[OLD_SOFT],
            values[LOCAL_ONLY]-values[OBSERVATION_MEAN], atol=1e-12, rtol=0)

    def test_geometry_is_not_a_semantic_truth_gate(self):
        actual, _ = self.scores((PRIMARY,))
        alternate, _ = local_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, self.logits, self.fine_local, methods=(PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], alternate[PRIMARY]))

    def test_canonical_self_invalid_and_rival_specific_protection(self):
        margins = contribution_margins(self.logits)
        reference = sampled_cached(self.fine_local, self.coordinates)
        risk = native_risk(margins, reference, reference, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)

    def test_budget_gauge_invalid_and_matched_norm(self):
        values, stats = self.scores()
        self.assertEqual(stats['local_canonical_risk_max'], 0)
        self.assertEqual(stats['local_invalid_write_max'], 0)
        self.assertEqual(stats['matched_previous_unmatchable'], 0)
        self.assertEqual(stats['direct_match_unmatchable'], 0)
        self.assertLessEqual(stats['matched_previous_norm_relative_error'], 1e-10)
        self.assertLessEqual(stats['direct_match_norm_relative_error'], 1e-10)
        self.assertLessEqual(stats['class_budget_max_error'], 1e-10)
        for name in (PRIMARY, LOCAL_ONLY, CLASS_MEAN, DIRECT_MATCHED):
            delta = values[name]-values[OLD_SOFT if name != LOCAL_ONLY else OBSERVATION_MEAN]
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))

    def test_bad_feature_count_and_methods_are_rejected(self):
        with self.assertRaises(ValueError):
            local_observations((), torch.ones(12, 7), self.members, self.fine)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
