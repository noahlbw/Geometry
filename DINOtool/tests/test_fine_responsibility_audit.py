import unittest

import torch

from dinotool.fine_alias_view import CONFIG
from dinotool.fine_responsibility_audit import (PRIMARY, REPLAY, METHODS, CLASS_MEAN,
    SHUFFLES, audit_scores, factorized_logs, position_control)
from dinotool.rival_responsibility_intersection import intersection_scores
from dinotool.rival_survivor_redistribution import redistribute
from test_rival_responsibility_intersection import ResponsibilityIntersectionTests


class FineResponsibilityAuditTests(unittest.TestCase):
    def test_old_scores_fine_only_exact_uniform_mass_and_own_nulls(self):
        for k in (20, 40):
            inputs, matches = ResponsibilityIntersectionTests().fixture(k)
            values, diagnostics = audit_scores(*inputs)
            old, _ = intersection_scores(*inputs, matches=matches, methods=(*REPLAY, PRIMARY))
            for name in (*REPLAY, PRIMARY):
                self.assertTrue(torch.equal(values[name], old[name]), name)
            self.assertTrue(torch.equal(values[CLASS_MEAN], old[REPLAY[-1]]))
            self.assertTrue(torch.equal(values[PRIMARY], audit_scores(*inputs, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertLess(max(diagnostics['all_mass_max_errors'].values()), 1e-10)
            self.assertEqual(set(diagnostics['control_spectrum_max_errors']), set(SHUFFLES))
            self.assertTrue(all(x == 0 for x in diagnostics['control_spectrum_max_errors'].values()))

    def test_common_pair_factor_cancels_without_floor(self):
        members = torch.arange(12).reshape(3, 4)
        canonical = members[:, 0]
        within = torch.tensor([.15, .2, .25, .4], dtype=torch.float64).repeat(2, 3, 1)
        pair_mass = torch.tensor([[.5, .3, .7], [.7, .5, .8], [.3, .2, .5]], dtype=torch.float64).expand(2, -1, -1)
        pair = (within[..., None]*pair_mass[:, :, None]).reshape(2, 12, 3).log()
        logs = within.reshape(2, 12).log()
        factored = factorized_logs(pair, logs, members)
        self.assertTrue(torch.allclose(factored, pair, atol=1e-14, rtol=0))
        risk = torch.zeros_like(pair)
        first = redistribute(factored.exp(), risk, members, canonical)
        second = redistribute(logs.exp()[..., None].expand_as(pair), risk, members, canonical)
        self.assertTrue(torch.allclose(first, second, atol=1e-12, rtol=0))
        self.assertTrue(bool((factored.exp() > CONFIG.epsilon).all()))

    def test_product_of_mixtures_differs_from_mixture_of_products(self):
        members = torch.arange(4).reshape(2, 2)
        q = torch.tensor([[[.9, .1], [.5, .5]], [[.1, .9], [.5, .5]]], dtype=torch.float64)
        mass = torch.tensor([[[.5, .9], [.1, .5]], [[.5, .1], [.9, .5]]], dtype=torch.float64)
        pair = (q[..., None]*mass[:, :, None]).mean(0, keepdim=True).reshape(1, 4, 2)
        within = q.mean(0, keepdim=True).reshape(1, 4)
        self.assertGreater(float((factorized_logs(pair.log(), within.log(), members).exp()-pair).abs().max()), .1)

    def test_position_null_uses_only_valid_queries_and_preserves_source_rows(self):
        logs = -torch.arange(72).reshape(6, 4, 3).double()
        valid = torch.tensor([True, False, True, True, False, True])
        changed = position_control(logs, valid)
        self.assertTrue(torch.equal(changed[~valid], logs[~valid]))
        self.assertTrue(torch.equal(changed[valid].flatten().sort().values, logs[valid].flatten().sort().values))
        self.assertTrue(torch.equal(changed, position_control(logs, valid)))
        self.assertFalse(torch.equal(changed[valid], logs[valid]))

    def test_invalid_queries_have_identical_scores(self):
        inputs, _ = ResponsibilityIntersectionTests().fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = audit_scores(*inputs)
        self.assertTrue(all(torch.equal(v, values[REPLAY[0]]) for v in values.values()))
        with self.assertRaises(ValueError):
            audit_scores(*inputs, methods=('undeclared',))


if __name__ == '__main__':
    unittest.main()
