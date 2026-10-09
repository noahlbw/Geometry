import unittest
from unittest.mock import patch

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.one_sided_alias_audit import (HARD as OLD_HARD, PRIMARY as OLD_PRIMARY,
    OBSERVATION_MEAN as OLD_MEAN, scores as old_scores)
from dinotool.pair_likelihood_coupling import (CLASS_MEAN, HARD, MATCHED_OLD,
    MATCHED_PROJECTED, METHODS, NO_ALIAS,
    OBSERVATION_MEAN, OLD_SOFT, PRIMARY, SHUFFLES, geometry_weights, logistic_gradient,
    logistic_objective, reconstruct, scores)
from dinotool.rival_alias_fast import CachedCrop


class PairLikelihoodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6206)
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
        return scores(self.local, self.operator, self.broad, self.field, self.wide, self.fine,
            self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents, methods=methods)

    def test_gradient_matches_autograd(self):
        weights = geometry_weights(torch.rand(4, 4), self.valid)
        value = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        local = self.local.double()
        raw = torch.randn(4, 3, dtype=torch.float64)
        q = (raw[:, :, None]-raw[:, None]).sigmoid()
        expected, = torch.autograd.grad(logistic_objective(local, value, weights, q, self.valid), value)
        actual = logistic_gradient(local, value.detach(), weights, q)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_masked_geometry_matches_retained_shell(self):
        weights = geometry_weights(torch.ones(4, 4), self.valid)
        torch.testing.assert_close(weights[self.valid].sum(-1), torch.ones(3, dtype=torch.float64))
        self.assertEqual(float(weights[~self.valid].abs().max()), 0.)
        self.assertEqual(float(weights[:, ~self.valid].abs().max()), 0.)

    def test_local_target_is_fixed_point(self):
        weights = geometry_weights(self.relation, self.valid)
        observed = weights@self.local.double()
        target = observed[:, :, None]-observed[:, None]
        value, diagnostics = reconstruct(self.local, weights, target, self.valid)
        self.assertTrue(torch.equal(value, self.local.double()))
        self.assertEqual(diagnostics['solver_max_gradient'], 0.)

    def test_objective_improves_gauge_and_invalid_scores_preserved(self):
        values, diagnostics = self.score()
        self.assertLess(diagnostics['solver_final_objective'], diagnostics['solver_initial_objective'])
        self.assertLess(diagnostics['solver_gauge_max_error'], 1e-10)
        self.assertTrue(torch.equal(values[PRIMARY][~self.valid], self.local.double()[~self.valid]))

    def test_converges_on_small_well_conditioned_problem(self):
        weights = geometry_weights(torch.rand(4, 4), self.valid)
        raw = torch.randn(4, 3, dtype=torch.float64)
        target = raw[:, :, None]-raw[:, None]
        _, diagnostics = reconstruct(self.local, weights, target, self.valid)
        self.assertLess(diagnostics['solver_max_gradient'], 1e-6)

    def test_global_class_gauge_shift_does_not_change_corrections(self):
        weights = geometry_weights(self.relation, self.valid)
        raw = torch.randn(4, 3, dtype=torch.float64)
        target = raw[:, :, None]-raw[:, None]
        a, _ = reconstruct(self.local, weights, target, self.valid)
        b, _ = reconstruct(self.local.double()+7, weights, target, self.valid)
        torch.testing.assert_close(b, a+7, atol=1e-12, rtol=0)

    def test_cycle_information_can_change_read_under_asymmetric_base(self):
        valid = torch.ones(1, dtype=torch.bool)
        local = torch.tensor([[2., 0., -1.]], dtype=torch.float64)
        weights = torch.ones(1, 1, dtype=torch.float64)
        base = local[:, :, None]-local[:, None]
        cycle = torch.tensor([[[0., 2., -2.], [-2., 0., 2.], [2., -2., 0.]]], dtype=torch.float64)
        self.assertEqual(float(cycle.sum(-1).abs().max()), 0.)
        a, _ = reconstruct(local, weights, base, valid)
        b, _ = reconstruct(local, weights, base+cycle, valid)
        self.assertGreater(float((a-b).abs().max()), .01)

    def test_probability_row_sums_are_sufficient_for_reader(self):
        valid = torch.ones(1, dtype=torch.bool)
        local = torch.tensor([[2., 0., -1.]], dtype=torch.float64)
        weights = torch.ones(1, 1, dtype=torch.float64)
        probability = torch.tensor([[[.5, .7, .2], [.3, .5, .8], [.8, .2, .5]]], dtype=torch.float64)
        circulation = torch.tensor([[[0., .02, -.02], [-.02, 0., .02], [.02, -.02, 0.]]], dtype=torch.float64)
        shifted = probability+circulation
        torch.testing.assert_close(probability.sum(-1), shifted.sum(-1), atol=1e-12, rtol=0)
        a, _ = reconstruct(local, weights, probability.logit(), valid)
        b, _ = reconstruct(local, weights, shifted.logit(), valid)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)

    def test_old_positive_soft_and_hard_are_bitwise_exact(self):
        actual, _ = self.score()
        old, _ = old_scores(self.local, self.operator, self.broad, self.field, self.wide, self.fine,
            self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents,
            methods=(OLD_MEAN, OLD_PRIMARY, OLD_HARD))
        for new, previous in ((OBSERVATION_MEAN, OLD_MEAN), (OLD_SOFT, OLD_PRIMARY), (HARD, OLD_HARD)):
            self.assertTrue(torch.equal(actual[new], old[previous]), new)

    def test_primary_singleton_matches_combined_and_skips_controls(self):
        actual, _ = self.score()
        with patch('dinotool.pair_likelihood_coupling.continuous_controls', side_effect=AssertionError):
            single, _ = self.score((PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], single[PRIMARY]))

    def test_all_singletons_match_combined(self):
        actual, _ = self.score()
        for name in METHODS[3:]:
            single, _ = self.score((name,))
            self.assertTrue(torch.equal(actual[name], single[name]), name)

    def test_old_singleton_timing_skips_new_solver(self):
        with patch('dinotool.pair_likelihood_coupling.reconstruct', side_effect=AssertionError):
            for method in (OBSERVATION_MEAN, OLD_SOFT, HARD):
                values, _ = self.score((method,))
                self.assertEqual(set(values), {method})

    def test_matched_word_controls_preserve_intervention_norm(self):
        values, diagnostics = self.score()
        norm = (values[PRIMARY]-values[NO_ALIAS])[self.valid].norm()
        for name in (CLASS_MEAN, MATCHED_OLD, MATCHED_PROJECTED, *SHUFFLES):
            torch.testing.assert_close((values[name]-values[NO_ALIAS])[self.valid].norm(), norm, atol=1e-12, rtol=0)
        self.assertEqual(diagnostics['matched_unmatchable'], 0.)

    def test_identical_sources_produce_no_word_action(self):
        self.fine = self.wide
        values, _ = self.score((PRIMARY, NO_ALIAS))
        self.assertTrue(torch.equal(values[PRIMARY], values[NO_ALIAS]))

    def test_invalid_targets_and_methods_fail(self):
        with self.assertRaises(ValueError):
            self.score(('NotFrozen',))
        with self.assertRaises(ValueError):
            reconstruct(self.local, torch.eye(4), torch.ones(4, 3, 3), self.valid)


if __name__ == '__main__':
    unittest.main()
