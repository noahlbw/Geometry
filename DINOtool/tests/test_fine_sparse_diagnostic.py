import unittest

import torch

from dinotool.fine_responsibility_reader import responsibility_scores
from dinotool.fine_sparse_diagnostic import FULL, METHODS, PRIMARY, diagnostic_scores, restrict_actions
from dinotool.rival_competition_admission import posterior_potential
from test_rival_alias_fast import RivalAliasFastTest


class FineSparseDiagnosticTests(unittest.TestCase):
    def test_full_budget_is_identity_and_invalid_rows_zero(self):
        torch.manual_seed(4)
        raw = torch.randn(7, 5, 5, dtype=torch.float64)
        action = raw - raw.transpose(-1, -2)
        posterior = torch.randn(7, 5, dtype=torch.float64).softmax(-1)
        valid = torch.tensor([True, True, False, True, False, True, True])
        masked, _ = restrict_actions(action, posterior, valid, 5)
        self.assertTrue(torch.equal(masked[valid], action[valid]))
        self.assertEqual(float(masked[~valid].abs().max()), 0.)
        self.assertTrue(torch.equal(posterior_potential(masked, posterior, valid)[0],
                                    posterior_potential(action, posterior, valid)[0]))

    def test_top2_preserves_antisymmetry_and_posterior_reconciliation(self):
        action = torch.tensor([[[0., 3., 8.], [-3., 0., 2.], [-8., -2., 0.]]], dtype=torch.float64)
        posterior = torch.tensor([[.6, .3, .1]], dtype=torch.float64)
        valid = torch.ones(1, dtype=torch.bool)
        masked, _ = restrict_actions(action, posterior, valid)
        self.assertTrue(torch.equal(masked, -masked.transpose(-1, -2)))
        potential, _ = posterior_potential(masked, posterior, valid)
        expected = torch.tensor([[.9, -1.8, 0.]], dtype=torch.float64)
        expected -= expected.mean(-1, keepdim=True)
        self.assertTrue(torch.allclose(potential, expected, atol=1e-15, rtol=0))

    def test_two_class_reader_recovers_frozen_full_source_exactly(self):
        inputs = RivalAliasFastTest().fixture(2, 20)
        values, _, _ = diagnostic_scores(*inputs)
        self.assertTrue(torch.equal(values[PRIMARY], values[FULL]))

    def test_original_endpoints_risk_and_singleton_unchanged(self):
        inputs = RivalAliasFastTest().fixture(5, 20)
        values, risk, diagnostics = diagnostic_scores(*inputs)
        old, old_risk, _ = responsibility_scores(*inputs, methods=METHODS[:-1])
        self.assertTrue(torch.equal(risk, old_risk))
        for method in METHODS[:-1]:
            self.assertTrue(torch.equal(values[method], old[method]), method)
        single = diagnostic_scores(*inputs, methods=(PRIMARY,))[0][PRIMARY]
        self.assertTrue(torch.equal(values[PRIMARY], single))
        self.assertTrue(diagnostics['diagnostic_dense_reader_retained'])

    def test_ties_are_stable_and_invalid_budget_rejects(self):
        action = torch.ones(2, 3, 3, dtype=torch.float64)
        posterior = torch.ones(2, 3, dtype=torch.float64) / 3
        valid = torch.ones(2, dtype=torch.bool)
        _, mask = restrict_actions(action, posterior, valid)
        self.assertTrue(bool(mask[:, :2, :2].all()))
        self.assertFalse(bool(mask[:, 2].any()))
        for k in (1, 4):
            with self.assertRaises(ValueError):
                restrict_actions(action, posterior, valid, k)


if __name__ == '__main__':
    unittest.main()
