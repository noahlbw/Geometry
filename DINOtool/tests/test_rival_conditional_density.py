import unittest

import torch

from dinotool.fine_alias_view import CONFIG
from dinotool.rival_conditional_density import (PRIMARY, REPLAY, FINE_ONLY, UNIFORM,
    response_densities, density_responsibility, density_scores)
from dinotool.rival_responsibility_intersection import intersection_scores
from test_rival_responsibility_intersection import ResponsibilityIntersectionTests


class RivalConditionalDensityTests(unittest.TestCase):
    def test_sorted_scan_matches_dense_reference_and_chunking(self):
        torch.manual_seed(303)
        alias, q = torch.randn(13, 9, dtype=torch.float64), torch.randn(13, 4, dtype=torch.float64).softmax(-1)
        alias[1] = alias[2]
        valid = torch.rand(13) > .2
        for holdout in (False, True):
            for balance in (False, True):
                got, known = response_densities(alias, q, valid, exclude_self=holdout, balance=balance, chunk=3)
                kernel = torch.exp(-(alias[:, None]-alias[None]).abs())
                kernel.masked_fill_(~valid[None, :, None], 0.)
                if holdout:
                    kernel[torch.arange(13), torch.arange(13)] = 0.
                expected = torch.einsum('ija,jc->iac', kernel, q)
                mass = q[valid].sum(0)-q if holdout else q[valid].sum(0).expand_as(q)
                if balance:
                    expected /= mass[:, None].clamp_min(CONFIG.epsilon)
                expected.masked_fill_(~valid[:, None, None], 0.)
                self.assertTrue(torch.allclose(got, expected, atol=1e-12, rtol=0))
                self.assertTrue(torch.equal(known[valid], mass[valid] > CONFIG.epsilon))
                other, _ = response_densities(alias, q, valid, exclude_self=holdout, balance=balance, chunk=1)
                self.assertTrue(torch.equal(got, other))

    def test_class_balance_and_query_self_exclusion_are_exact(self):
        alias = torch.tensor([[0.], [0.], [5.], [5.], [5.]], dtype=torch.float64)
        q = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]], dtype=torch.float64)
        valid = torch.ones(5, dtype=torch.bool)
        density, available = response_densities(alias, q, valid)
        source, _ = density_responsibility(density, available, torch.tensor([0]))
        self.assertGreater(float(source[0, 0, 1]), .9)
        self.assertLess(float(source[3, 0, 1]), .1)
        changed = q.clone()
        changed[0] = torch.tensor([0., 1.])
        other, _ = response_densities(alias, changed, valid)
        self.assertTrue(torch.equal(density[0], other[0]))

    def test_flat_response_has_no_discrimination_and_no_support_is_neutral(self):
        alias = torch.zeros(7, 5, dtype=torch.float64)
        q = torch.randn(7, 3, dtype=torch.float64).softmax(-1)
        density, available = response_densities(alias, q, torch.ones(7, dtype=torch.bool))
        source, known = density_responsibility(density, available, torch.arange(5)%3)
        self.assertTrue(torch.allclose(source, torch.full_like(source, .5), atol=1e-12, rtol=0))
        density, available = response_densities(alias, q, torch.zeros(7, dtype=torch.bool))
        source, known = density_responsibility(density, available, torch.arange(5)%3)
        self.assertTrue(torch.equal(source, torch.full_like(source, .5)))
        self.assertFalse(bool(known.any()))
        valid = torch.zeros(7, dtype=torch.bool)
        valid[0] = True
        self.assertFalse(bool(response_densities(alias, q, valid)[1].any()))

    def test_extreme_scores_stay_finite_and_invalid_sources_reject(self):
        alias = torch.tensor([[-1e4, 1e4], [1e4, -1e4], [0., 0.]], dtype=torch.float64)
        q, valid = torch.ones(3, 2)/2, torch.ones(3, dtype=torch.bool)
        density, available = response_densities(alias, q, valid)
        self.assertTrue(bool(torch.isfinite(density).all()))
        source, _ = density_responsibility(density, available, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(source, torch.full_like(source, .5)))
        with self.assertRaises(ValueError):
            response_densities(alias*torch.nan, q, valid)

    def test_original_scores_uniform_mass_own_nulls_and_singleton(self):
        for k in (20, 40):
            inputs, matches = ResponsibilityIntersectionTests().fixture(k)
            values, diagnostics = density_scores(*inputs)
            old = intersection_scores(*inputs, matches=matches, methods=(*REPLAY, FINE_ONLY))[0]
            for name in (*REPLAY, FINE_ONLY):
                self.assertTrue(torch.equal(values[name], old[name]), name)
            self.assertTrue(torch.equal(values[UNIFORM], old[FINE_ONLY]))
            self.assertTrue(torch.equal(values[PRIMARY], density_scores(*inputs, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertLess(max(diagnostics['all_mass_max_errors'].values()), 1e-10)
            self.assertTrue(all(x == 0 for x in diagnostics['control_spectrum_max_errors'].values()))
            self.assertEqual(diagnostics['canonical_weight_min'], 1.)
            self.assertEqual(diagnostics['old_rejected_weights_max'], 0.)


if __name__ == '__main__':
    unittest.main()
