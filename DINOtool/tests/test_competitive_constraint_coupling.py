import unittest

import torch

from dinotool.competitive_constraint_coupling import (METHODS, NORM_MATCHED, NO_ALIAS,
    OBSERVATION_MEAN, PRIMARY, UNIFORM, competitive_graph, constraint_scores,
    geometry_weights, shuffled_prior, solve_constraints)
from dinotool.native_alias_noise import signed_potential
from dinotool.rival_alias_fast import CachedCrop
from dinotool.supported_positive_alias import (NO_GEOMETRY_RISK as OLD_UNIFORM,
    OBSERVATION_MEAN as OLD_MEAN, joint_scores)


class CompetitiveConstraintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(606)
        self.valid = torch.tensor([True, True, True, True, False])
        self.relation = torch.rand(5, 5, dtype=torch.float64)
        self.weights = geometry_weights(self.relation, self.valid)
        self.local = torch.randn(5, 3, dtype=torch.float64)
        self.observation = torch.randn(5, 3, dtype=torch.float64)
        raw = torch.randn(5, 3, 3, dtype=torch.float64)
        self.margins = raw-raw.transpose(-1, -2)
        self.margins[~self.valid] = 0
        self.posterior = self.local.softmax(-1)
        gram = self.weights.T@self.weights
        self.operator = torch.linalg.solve(torch.eye(5, dtype=torch.float64)+gram, gram)

    def solve(self, margins=None, posterior=None, observation=None):
        return solve_constraints(self.local, self.observation if observation is None else observation,
            self.margins if margins is None else margins, self.weights,
            self.posterior if posterior is None else posterior, self.valid)

    def test_graph_budget_symmetry_gauge_and_positive_semidefiniteness(self):
        edges, laplacian, stats = competitive_graph(self.posterior, self.valid)
        self.assertLessEqual(stats['constraint_trace_error'], 1e-12)
        torch.testing.assert_close(edges, edges.transpose(-1, -2), atol=0, rtol=0)
        torch.testing.assert_close(laplacian.sum(-1), torch.zeros(5, 3, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(laplacian).min()), -1e-12)
        self.assertTrue(torch.equal(laplacian[~self.valid], torch.zeros(1, 3, 3, dtype=torch.float64)))

    def test_nearly_one_hot_prior_retains_stable_graph_budget(self):
        prior = torch.tensor([[1., 1e-25, 1e-26]]*5, dtype=torch.float64)
        _, laplacian, stats = competitive_graph(prior, self.valid)
        self.assertEqual(stats['empty_graph_rows'], 0)
        self.assertTrue(bool(torch.isfinite(laplacian).all()))
        self.assertLessEqual(stats['constraint_trace_error'], 1e-12)
        torch.testing.assert_close(laplacian.sum(-1), torch.zeros(5, 3, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_numerically_empty_graph_fallback_is_recorded(self):
        prior = torch.tensor([[1., 0., 0.]]*5, dtype=torch.float64)
        _, laplacian, stats = competitive_graph(prior, self.valid)
        self.assertEqual(stats['empty_graph_rows'], 4)
        uniform = torch.full_like(prior, 1/3)
        _, expected, _ = competitive_graph(uniform, self.valid)
        torch.testing.assert_close(laplacian, expected, atol=0, rtol=0)

    def test_matrix_free_solution_matches_independent_dense_system(self):
        actual, stats = self.solve()
        edges, laplacian, _ = competitive_graph(self.posterior, self.valid)
        dense = torch.eye(15, dtype=torch.float64)+torch.einsum('qi,qcd,qj->icjd',
            self.weights, laplacian, self.weights).reshape(15, 15)
        read = self.weights@self.observation
        transported = (self.weights@self.margins.flatten(1)).reshape_as(self.margins)
        target = read[:, :, None]-read[:, None]+.5*transported
        rhs = self.local+self.weights.T@(edges*target).sum(-1)
        expected = torch.linalg.solve(dense, rhs.flatten()).reshape_as(self.local)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=0)
        self.assertLessEqual(stats['cg_relative_residual'], 1e-7)
        self.assertGreater(float(torch.linalg.eigvalsh(dense).min()), .99)

    def test_uniform_prior_recovers_original_h_up_to_class_common_offset(self):
        actual, _ = self.solve(posterior=torch.full_like(self.posterior, 1/3))
        potential = self.margins.mean(-1)
        expected = self.local+self.operator@(self.observation-self.local+.5*potential)
        torch.testing.assert_close(actual-actual.mean(-1, keepdim=True),
            expected-expected.mean(-1, keepdim=True), atol=1e-8, rtol=0)
        torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), atol=1e-9, rtol=0)

    def test_joint_solution_preserves_local_class_common_mode(self):
        actual, stats = self.solve()
        torch.testing.assert_close(actual.sum(-1), self.local.sum(-1), atol=1e-12, rtol=0)
        self.assertLessEqual(stats['class_mean_preservation_error'], 1e-8)

    def test_class_common_offsets_do_not_change_decisions(self):
        actual, _ = self.solve()
        offset = torch.randn(5, 1, dtype=torch.float64)
        changed, _ = solve_constraints(self.local+offset, self.observation+offset,
            self.margins, self.weights, self.posterior, self.valid)
        torch.testing.assert_close(changed.softmax(-1), actual.softmax(-1), atol=1e-9, rtol=0)

    def test_zero_geometry_support_keeps_exact_local_scores(self):
        actual, _ = solve_constraints(self.local, self.observation, self.margins,
            torch.zeros_like(self.weights), self.posterior, self.valid)
        self.assertTrue(torch.equal(actual, self.local))

    def test_padded_donors_remain_local_and_do_not_affect_valid_scores(self):
        actual, _ = self.solve()
        changed_observation = self.observation.clone()
        changed_observation[~self.valid] = 1e6
        changed, _ = self.solve(observation=changed_observation)
        self.assertTrue(torch.equal(actual, changed))
        self.assertTrue(torch.equal(actual[~self.valid], self.local[~self.valid]))

    def test_prior_shuffle_preserves_distribution_and_is_reproducible(self):
        first = shuffled_prior(self.posterior, self.valid)
        second = shuffled_prior(self.posterior, self.valid)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[~self.valid], self.posterior[~self.valid]))
        self.assertTrue(torch.equal(first[self.valid].sort(0).values, self.posterior[self.valid].sort(0).values))

    def test_reject_nonantisymmetric_action_or_unnormalized_prior(self):
        with self.assertRaises(ValueError):
            self.solve(margins=torch.ones_like(self.margins))
        with self.assertRaises(ValueError):
            self.solve(posterior=torch.ones_like(self.posterior))

    def fixture(self):
        members = torch.arange(12).reshape(3, 4)
        parents = torch.arange(3).repeat_interleave(4)
        canonical = members[:, 0]
        coordinates = torch.randn(5, 2)
        indices = torch.arange(5)[:, None]
        coefficients = torch.ones(5, 1)
        wide = [CachedCrop(torch.randn(5, 3, 4), torch.randn(5, 12, 3), indices, coefficients)]
        fine = [CachedCrop(torch.randn(5, 3, 4), torch.randn(5, 12, 3), indices, coefficients)]
        return (self.local, self.operator, self.observation, self.local.float(), wide, fine,
                coordinates, self.relation, self.valid, members, canonical, parents)

    def test_uniform_and_positive_replay_previous_reader_exactly(self):
        args = self.fixture()
        actual, _ = constraint_scores(*args, methods=(UNIFORM, OBSERVATION_MEAN))
        expected, _ = joint_scores(*args, methods=(OLD_UNIFORM, OLD_MEAN))
        self.assertTrue(torch.equal(actual[UNIFORM], expected[OLD_UNIFORM]))
        self.assertTrue(torch.equal(actual[OBSERVATION_MEAN], expected[OLD_MEAN]))

    def test_singleton_matches_simultaneous_controls(self):
        args = self.fixture()
        combined, _ = constraint_scores(*args)
        for method in METHODS[3:]:
            single, _ = constraint_scores(*args, methods=(method,))
            self.assertTrue(torch.equal(combined[method], single[method]), method)

    def test_word_write_norm_control_shares_noalias_base(self):
        values, stats = constraint_scores(*self.fixture(), methods=(PRIMARY, NO_ALIAS, NORM_MATCHED))
        torch.testing.assert_close((values[PRIMARY]-values[NO_ALIAS]).norm(),
            (values[NORM_MATCHED]-values[NO_ALIAS]).norm(), atol=1e-8, rtol=0)
        self.assertLessEqual(stats['norm_matching_error'], 1e-7)

    def test_no_risk_is_exact_noalias_identity(self):
        args = list(self.fixture())
        args[5][0].margins.fill_(1)
        values, _ = constraint_scores(*args, methods=(PRIMARY, NO_ALIAS))
        self.assertTrue(torch.equal(values[PRIMARY], values[NO_ALIAS]))


if __name__ == '__main__':
    unittest.main()
