import unittest

import torch

from dinotool.rival_alias_fast import cache_observations
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_matched_support import semantic_rivals
from dinotool.rival_responsibility_intersection import (PRIMARY, CLASS_MEAN,
    sampled_log_responsibilities, intersection_log_weights, protected_weights, intersection_scores)
from dinotool.rival_survivor_redistribution import PRIMARY as SURVIVOR, redistribution_scores
from test_rival_alias_fast import RivalAliasFastTest


class ResponsibilityIntersectionTests(unittest.TestCase):
    def fixture(self, k=20):
        inputs = RivalAliasFastTest().fixture(k=k)
        torch.manual_seed(32)
        matches = semantic_rivals(torch.randn(inputs[10].numel(), 2, 12), inputs[10])
        return inputs, matches

    def test_pair_and_within_probabilities_have_actual_normalized_mass(self):
        inputs, _ = self.fixture()
        observed = cache_observations(inputs[3], inputs[4], inputs[7], inputs[-1], inputs[10])
        pair, within = sampled_log_responsibilities(observed, inputs[9], chunk=7)
        q = within[:, inputs[10]].exp()
        p = pair[:, inputs[10]].exp().sum(2)
        self.assertTrue(torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-12, rtol=0))
        self.assertTrue(torch.allclose(p+p.transpose(-1, -2), torch.ones_like(p), atol=1e-12, rtol=0))
        self.assertTrue(torch.equal(intersection_log_weights(pair, pair), torch.zeros_like(pair)))
        default = sampled_log_responsibilities(observed, inputs[9])
        self.assertTrue(torch.equal(pair, default[0]))
        self.assertTrue(torch.equal(within, default[1]))

    def test_intersection_identity_and_rival_conditionality_without_floor(self):
        wide = torch.tensor([[[-1., -2.], [-4., -5.]]], dtype=torch.float64)
        fine = torch.tensor([[[-3., -1.], [-4., -10.]]], dtype=torch.float64)
        logs = intersection_log_weights(wide, fine)
        self.assertTrue(torch.equal(logs+wide, torch.minimum(wide, fine)))
        self.assertNotEqual(float(logs[0, 0, 0]), float(logs[0, 0, 1]))
        self.assertEqual(float(logs[0, 0, 1]), 0.)

    def test_extreme_response_logs_are_finite_and_protection_holds(self):
        inputs, _ = self.fixture()
        inputs[3][0].alias_logits.mul_(1000)
        observed = cache_observations(inputs[3], inputs[4], inputs[7], inputs[-1], inputs[10])
        pair, _ = sampled_log_responsibilities(observed, inputs[9])
        self.assertTrue(bool(torch.isfinite(pair).all()))
        risk = torch.zeros_like(pair)
        risk[0, 1, 2] = 1
        weights = protected_weights(-torch.ones_like(pair), risk, inputs[12], inputs[11], inputs[9])
        self.assertEqual(float(weights[0, 1, 2]), 0.)
        self.assertTrue(bool((weights[:, inputs[11]] == 1).all()))
        self.assertTrue(bool((weights[~inputs[9]] == 1).all()))

    def test_historical_endpoints_controls_mass_and_singleton(self):
        for k in (20, 40):
            inputs, matches = self.fixture(k)
            values, diagnostics = intersection_scores(*inputs, matches=matches)
            old = retained_competition_scores(*inputs)[0]
            for name in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[name], old[name]))
            self.assertTrue(torch.equal(values[CLASS_MEAN], old['FineRivalProjected_Exact']))
            survivor = redistribution_scores(*inputs, matches=matches, methods=(SURVIVOR,))[0][SURVIVOR]
            self.assertTrue(torch.equal(values[SURVIVOR], survivor))
            self.assertTrue(torch.equal(values[PRIMARY], intersection_scores(*inputs, matches=matches, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertLess(max(diagnostics['all_mass_max_errors'].values()), 1e-10)
            self.assertTrue(all(v == 0 for v in diagnostics['control_spectrum_max_errors'].values()))
            self.assertEqual(diagnostics['canonical_weight_min'], 1.)

    def test_invalid_queries_are_identity_and_bad_logs_reject(self):
        inputs, matches = self.fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = intersection_scores(*inputs, matches=matches)
        self.assertTrue(all(torch.equal(v, values['NoAdmission_Exact']) for v in values.values()))
        with self.assertRaisesRegex(ValueError, 'finite'):
            intersection_log_weights(torch.zeros(2, 3), torch.full((2, 3), torch.nan))


if __name__ == '__main__':
    unittest.main()
