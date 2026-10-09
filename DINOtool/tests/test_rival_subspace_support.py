import unittest

import torch

from dinotool.fine_alias_view import CONFIG
from dinotool.rival_subspace_support import (PRIMARY, CLASS_MEAN, rival_subspaces,
    subspace_weights, subspace_scores)
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_survivor_redistribution import PRIMARY as SURVIVOR, redistribution_scores
from test_rival_alias_fast import RivalAliasFastTest


class RivalSubspaceTests(unittest.TestCase):
    def fixture(self, k=20):
        inputs = RivalAliasFastTest().fixture(k=k)
        torch.manual_seed(19)
        source = rival_subspaces(torch.randn(inputs[10].numel(), 2, 12), inputs[10])
        return inputs, source

    def test_ridge_residual_equals_explicit_visual_projection(self):
        torch.manual_seed(51)
        features = torch.randn(12, 3, 7, dtype=torch.float64)
        members = torch.arange(12).reshape(3, 4)
        source = rival_subspaces(features, members)
        text = features.mean(1)/source.template_mean_norms[:, None]
        visual = torch.randn(9, 7, dtype=torch.float64)
        raw = visual @ features.mean(1).T
        normalized = raw/source.template_mean_norms
        shared = torch.einsum('nck,ack->nac', normalized[:, members], source.coefficients)
        for c, group in enumerate(members):
            projection = source.coefficients[:, c] @ text[group]
            residual = text-projection
            self.assertTrue(torch.allclose(normalized-shared[:, :, c], visual @ residual.T, atol=1e-12, rtol=0))

    def test_shared_response_is_downweighted_but_exclusive_response_survives(self):
        features = torch.zeros(8, 1, 3, dtype=torch.float64)
        features[:4, 0, 0] = 1
        features[4:, 0, 1] = 1
        features[1, 0] = features[4, 0]
        members = torch.arange(8).reshape(2, 4)
        source = rival_subspaces(features, members)
        raw = torch.ones(1, 8, dtype=torch.float64)
        risk = torch.zeros(1, 8, 2)
        parents, canonical = torch.arange(2).repeat_interleave(4), members[:, 0]
        weights = subspace_weights(raw, source, risk, members, parents, canonical, torch.tensor([True]))
        self.assertAlmostEqual(float(weights[0, 1, 1]), 1/5, places=12)
        self.assertEqual(float(weights[0, 2, 1]), 1.)
        risk[0, 3, 1] = 1
        rejected = subspace_weights(raw, source, risk, members, parents, canonical, torch.tensor([True]))
        self.assertEqual(float(rejected[0, 3, 1]), 0.)
        negative = subspace_weights(-raw, source, risk, members, parents, canonical, torch.tensor([True]))
        self.assertEqual(float(negative[0, 1, 1]), CONFIG.epsilon)

    def test_historical_endpoints_mass_nulls_and_singleton(self):
        for k in (20, 40):
            inputs, source = self.fixture(k)
            values, diagnostics = subspace_scores(*inputs, source=source)
            old = retained_competition_scores(*inputs)[0]
            for name in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[name], old[name]))
            self.assertTrue(torch.equal(values[CLASS_MEAN], old['FineRivalProjected_Exact']))
            survivor = redistribution_scores(*inputs, matches=source.matches, methods=(SURVIVOR,))[0][SURVIVOR]
            self.assertTrue(torch.equal(values[SURVIVOR], survivor))
            self.assertTrue(torch.equal(values[PRIMARY], subspace_scores(*inputs, source=source, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertLess(max(diagnostics['all_mass_max_errors'].values()), 1e-10)
            self.assertTrue(all(v == 0 for v in diagnostics['control_spectrum_max_errors'].values()))
            self.assertEqual(diagnostics['canonical_weight_min'], 1.)

    def test_invalid_queries_are_identity_and_groups_validate(self):
        inputs, source = self.fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = subspace_scores(*inputs, source=source)
        self.assertTrue(all(torch.equal(v, values['NoAdmission_Exact']) for v in values.values()))
        with self.assertRaisesRegex(ValueError, 'unique'):
            rival_subspaces(torch.ones(6, 1, 2), torch.tensor([[0, 1, 2], [3, 4, 4]]))
        with self.assertRaisesRegex(ValueError, 'Nonzero'):
            rival_subspaces(torch.zeros(6, 1, 2), torch.arange(6).reshape(2, 3))


if __name__ == '__main__':
    unittest.main()
