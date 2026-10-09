import unittest
from types import SimpleNamespace

import torch

from dinotool.rival_competition_admission import (METHODS, posterior_potential, project_actions, manifest_samples,
    top_two_potential, retained_competition_scores)
from dinotool.rival_alias_fast import retained_scores_fast
from test_rival_alias_fast import RivalAliasFastTest


class RivalCompetitionAdmissionTest(unittest.TestCase):
    def test_manifest_preserves_declared_complete_image_order(self):
        samples = [SimpleNamespace(key=k) for k in ('a', 'b', 'c')]
        actual = manifest_samples(samples, {'dataset': 'vdd', 'sample_keys': ['c', 'a']}, 'vdd')
        self.assertEqual([s.key for s in actual], ['c', 'a'])

    def test_manifest_refuses_invalid_scope_or_keys(self):
        samples = [SimpleNamespace(key='a')]
        for dataset, keys in (('udd5', ['a']), ('vdd', []), ('vdd', ['a', 'a']), ('vdd', ['b'])):
            with self.assertRaises(ValueError):
                manifest_samples(samples, {'dataset': dataset, 'sample_keys': keys}, 'vdd')

    def test_uniform_posterior_recovers_original_reconciliation(self):
        torch.manual_seed(6)
        x = torch.randn(7, 5, 5, dtype=torch.float64)
        margins = x-x.transpose(-1, -2)
        valid = torch.ones(7, dtype=torch.bool)
        actual, stats = posterior_potential(margins, torch.full((7, 5), .2, dtype=torch.float64), valid)
        self.assertTrue(torch.allclose(actual, margins.mean(-1), atol=1e-15, rtol=0))
        self.assertLess(stats['weighted_normal_equation_max_error'], 1e-10)

    def test_weighted_solution_matches_constrained_linear_solver(self):
        torch.manual_seed(7)
        x = torch.randn(4, 4, 4, dtype=torch.float64)
        e = x-x.transpose(-1, -2)
        p = torch.randn(4, 4, dtype=torch.float64).softmax(-1)
        actual, _ = posterior_potential(e, p, torch.ones(4, dtype=torch.bool))
        w = p[:, :, None]*p[:, None]
        laplacian = torch.diag_embed(w.sum(-1))-w
        system = laplacian+torch.ones(4, 4, dtype=torch.float64)[None]/4
        expected = torch.linalg.solve(system, (w*e).sum(-1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))

    def test_class_permutation_equivariance(self):
        torch.manual_seed(8)
        a = torch.randn(3, 5, 5, dtype=torch.float64)
        e = a-a.transpose(-1, -2)
        p = torch.randn(3, 5, dtype=torch.float64).softmax(-1)
        valid = torch.ones(3, dtype=torch.bool)
        permutation = torch.tensor([3, 1, 4, 0, 2])
        first, _ = posterior_potential(e, p, valid)
        second, _ = posterior_potential(e[:, permutation][:, :, permutation], p[:, permutation], valid)
        self.assertTrue(torch.allclose(first[:, permutation], second, atol=1e-12, rtol=0))

    def test_projection_does_not_reverse_or_overshoot_fine_target(self):
        e = torch.tensor([[[0., -8., 2.], [8., 0., -3.], [-2., 3., 0.]]], dtype=torch.float64)
        t = torch.tensor([[[0., -3., -4.], [3., 0., -5.], [4., 5., 0.]]], dtype=torch.float64)
        actual = project_actions(e, t)
        expected = torch.tensor([[[0., -3., 0.], [3., 0., -3.], [0., 3., 0.]]], dtype=torch.float64)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, -actual.transpose(-1, -2)))

    def test_top_two_changes_only_current_competitors(self):
        e = torch.tensor([[[0., 2., 6.], [-2., 0., 4.], [-6., -4., 0.]]], dtype=torch.float64)
        actual = top_two_potential(e, torch.tensor([[8., 4., 9.]]), torch.tensor([True]))
        self.assertTrue(torch.equal(actual, torch.tensor([[3., 0., -3.]], dtype=torch.float64)))

    def test_original_scores_and_risks_are_bitwise_retained(self):
        args = RivalAliasFastTest().fixture()
        old, old_risk, old_stats = retained_scores_fast(*args)
        new, risk, stats = retained_competition_scores(*args)
        self.assertTrue(torch.equal(old_risk, risk))
        for m in old: self.assertTrue(torch.equal(old[m], new[m]), m)
        for k, v in old_stats.items(): self.assertEqual(stats[k], v)

    def test_every_independent_endpoint_and_zero_risk_identity(self):
        args = list(RivalAliasFastTest().fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        for method in METHODS:
            values, risk, stats = retained_competition_scores(*args, methods=(method,))
            self.assertEqual(set(values), {method})
            self.assertEqual(float(risk.abs().max()), 0.)
            if method != 'Geometry':
                baseline = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
                self.assertTrue(torch.equal(values[method], baseline), method)

    def test_invalid_rows_do_not_create_actions(self):
        e = torch.zeros(3, 3, 3, dtype=torch.float64)
        e[0, 0, 1], e[0, 1, 0] = 4., -4.
        p = torch.full((3, 3), 1/3, dtype=torch.float64)
        out, _ = posterior_potential(e, p, torch.tensor([False, True, True]))
        self.assertEqual(float(out.abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
