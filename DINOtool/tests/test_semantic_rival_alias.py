import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.native_query_alias import native_risk
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop, sampled_cached
from dinotool.semantic_rival_alias import (CLASS_MEAN, DIRECT_MATCHED, MATCHED_KEY_NULL,
    METHODS, OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, predict_image, rival_affinity,
    semantic_margins, semantic_observations, semantic_scores, shuffled_affinity)


class SemanticRivalTests(unittest.TestCase):
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
        self.texts = torch.randn(12, 7)
        self.affinity = rival_affinity(self.texts, self.members)
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            coefficients = torch.tensor([[.3, .7]]*4)
            target.append(CachedCrop(evidence, contribution_margins(evidence), ids, coefficients))

    def scores(self, methods=METHODS[3:]):
        return semantic_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, self.affinity, methods=methods)

    def test_reference_is_positive_row_stochastic(self):
        self.assertTrue(bool((self.affinity > 0).all()))
        torch.testing.assert_close(self.affinity.sum(-1), torch.ones(12, 3, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_reference_depends_on_alias_and_rival_identity(self):
        self.assertFalse(torch.equal(self.affinity[0, 1], self.affinity[1, 1]))
        self.assertFalse(torch.equal(self.affinity[0, 1], self.affinity[0, 2]))

    def test_weighted_log_reference_matches_literal_lse(self):
        evidence = self.wide[0].evidence
        reference = (evidence.double()[:, None]+self.affinity.log()[None]).logsumexp(-1)
        expected = (evidence.double().flatten(1)[..., None]-reference).float()
        torch.testing.assert_close(semantic_margins(evidence, self.members, self.affinity), expected, atol=0, rtol=0)

    def test_uniform_reference_recovers_pool_margins(self):
        uniform = torch.full_like(self.affinity, .25)
        actual = semantic_margins(self.wide[0].evidence, self.members, uniform)
        torch.testing.assert_close(actual, self.wide[0].margins, atol=3e-7, rtol=0)

    def test_noncontiguous_alias_layout_is_preserved(self):
        members = torch.tensor([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
        affinity = rival_affinity(self.texts, members)
        evidence = self.wide[0].evidence
        reference = (evidence.double()[:, None]+affinity.log()[None]).logsumexp(-1)
        expected = evidence.new_empty(4, 12, 3)
        expected[:, members.flatten()] = (evidence.double().flatten(1)[..., None]-reference[:, members.flatten()]).float()
        self.assertTrue(torch.equal(semantic_margins(evidence, members, affinity), expected))

    def test_joint_token_offset_cancels(self):
        evidence = self.wide[0].evidence.double()
        actual = semantic_margins(evidence+torch.arange(4)[:, None, None]*10, self.members, self.affinity)
        expected = semantic_margins(evidence, self.members, self.affinity)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)

    def test_extreme_sparse_reference_has_finite_log_domain_repair(self):
        evidence = torch.zeros(4, 3, 4, dtype=torch.float64)
        evidence[..., 0] = 10000
        evidence[..., 1] = -10000
        affinity = torch.zeros_like(self.affinity)
        affinity[..., 1] = 1
        actual = semantic_margins(evidence, self.members, affinity)
        expected = evidence.flatten(1)[..., None]+10000
        self.assertTrue(torch.equal(actual, expected.expand_as(actual)))

    def test_cached_scores_and_stencils_are_identical_objects(self):
        rows = semantic_observations(self.wide, self.members, self.affinity)
        self.assertIs(rows[0].evidence, self.wide[0].evidence)
        self.assertIs(rows[0].indices, self.wide[0].indices)
        self.assertIs(rows[0].coefficients, self.wide[0].coefficients)
        sampled = sampled_cached(rows, self.coordinates)
        expected = (rows[0].margins[rows[0].indices]*rows[0].coefficients[..., None, None]).sum(1)
        self.assertTrue(torch.equal(sampled, expected))

    def test_key_shuffle_preserves_every_reference_weight_spectrum(self):
        changed = shuffled_affinity(self.affinity)
        self.assertFalse(torch.equal(changed, self.affinity))
        self.assertTrue(torch.equal(changed.sort(-1).values, self.affinity.sort(-1).values))

    def test_previous_positive_soft_and_hard_replay_exactly(self):
        actual, _ = self.scores()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(PREVIOUS_MEAN, PREVIOUS_SOFT, PREVIOUS_HARD))
        for new, old in ((OBSERVATION_MEAN, PREVIOUS_MEAN), (OLD_SOFT, PREVIOUS_SOFT), (OLD_HARD, PREVIOUS_HARD)):
            self.assertTrue(torch.equal(actual[new], previous[old]), new)

    def test_singleton_matches_all_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            single, _ = self.scores((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_canonical_self_and_invalid_comparisons_are_protected(self):
        wm = sampled_cached(semantic_observations(self.wide, self.members, self.affinity), self.coordinates)
        fm = sampled_cached(semantic_observations(self.fine, self.members, self.affinity), self.coordinates)
        risk = native_risk(wm, fm, fm, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)

    def test_identical_source_has_no_negative_witness(self):
        self.fine = self.wide
        values, _ = self.scores()
        self.assertTrue(torch.equal(values[PRIMARY], values[OBSERVATION_MEAN]))

    def test_geometry_is_not_a_semantic_truth_gate(self):
        actual, _ = self.scores((PRIMARY,))
        alternate, _ = semantic_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, self.affinity, methods=(PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], alternate[PRIMARY]))

    def test_budgets_norms_gauge_and_invalid_writes(self):
        values, stats = self.scores()
        self.assertEqual(stats['canonical_risk_max'], 0)
        self.assertLessEqual(stats['class_budget_max_error'], 1e-10)
        for prefix in ('matched_previous', 'matched_key', 'direct_match'):
            self.assertEqual(stats[prefix+'_unmatchable'], 0)
            self.assertLessEqual(stats[prefix+'_norm_relative_error'], 1e-10)
        for name in (PRIMARY, CLASS_MEAN, DIRECT_MATCHED, MATCHED_KEY_NULL):
            delta = values[name]-values[OBSERVATION_MEAN]
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(delta[~self.valid], torch.zeros_like(delta[~self.valid])))

    def test_prediction_reuses_query_text_without_an_encoder(self):
        aliases = tuple('word'+str(i) for i in range(12))
        bank = SimpleNamespace(alias_names=aliases, parent_indices=self.parents)
        query = SimpleNamespace(aliases=aliases, parents=self.parents, features=self.texts[:, None], class_names=('a', 'b', 'c'))
        sentinel = object()
        with patch('dinotool.semantic_rival_alias.source_predict', return_value=sentinel) as source:
            result = predict_image(None, None, {'p': bank}, None, {'p': query}, methods=(PRIMARY,))
        self.assertIs(result, sentinel)
        source.assert_called_once()
        self.assertEqual(source.call_args.kwargs['methods'], (PRIMARY,))

    def test_invalid_reference_layout_and_methods_are_rejected(self):
        with self.assertRaises(ValueError):
            rival_affinity(self.texts, torch.zeros_like(self.members))
        with self.assertRaises(ValueError):
            semantic_margins(self.wide[0].evidence, self.members, self.affinity*.5)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
