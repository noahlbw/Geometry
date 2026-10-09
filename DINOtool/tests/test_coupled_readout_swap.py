import unittest

import torch

from dinotool.coupled_readout_swap import replace_local


class CoupledReadoutSwapTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        self.geometry, self.local, self.broad, self.potential = (
            torch.randn(7, 3, dtype=torch.float64) for _ in range(4))
        relation = torch.randn(7, 7, dtype=torch.float64)
        gram = relation.T @ relation
        self.operator = torch.linalg.solve(torch.eye(7) + gram, gram)

    def reference(self, operator):
        baseline = self.geometry + operator @ (self.broad - self.geometry)
        return baseline, baseline + operator @ self.potential

    def test_matches_direct_fixed_observation_reconstruction(self):
        baseline, screened = self.reference(self.operator)
        unchanged = tuple(value.clone() for value in (self.geometry, self.local, self.operator, baseline, screened))
        actual, admitted = replace_local(self.local, self.geometry, self.operator, baseline, screened)
        direct = self.local + self.operator @ (self.broad - self.local)
        torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(admitted, direct + self.operator @ self.potential, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(admitted - actual, screened - baseline, atol=1e-12, rtol=1e-12)
        for original, value in zip(unchanged, (self.geometry, self.local, self.operator, baseline, screened)):
            self.assertTrue(torch.equal(original, value))

    def test_identity_local_exactly_replays_reference(self):
        baseline, screened = self.reference(self.operator)
        actual, admitted = replace_local(self.geometry, self.geometry, self.operator, baseline, screened)
        self.assertTrue(torch.equal(actual, baseline))
        self.assertTrue(torch.equal(admitted, screened))

    def test_full_reconstruction_erases_local_readout_difference(self):
        operator = torch.eye(7, dtype=torch.float64)
        baseline, screened = self.reference(operator)
        actual, admitted = replace_local(self.local, self.geometry, operator, baseline, screened)
        self.assertTrue(torch.equal(actual, baseline))
        self.assertTrue(torch.equal(admitted, screened))

    def test_zero_reconstruction_uses_only_local_scores(self):
        operator = torch.zeros(7, 7, dtype=torch.float64)
        baseline, screened = self.reference(operator)
        actual, admitted = replace_local(self.local, self.geometry, operator, baseline, screened)
        torch.testing.assert_close(actual, self.local, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(actual, admitted))

    def test_incompatible_scores_rejected(self):
        baseline, screened = self.reference(self.operator)
        with self.assertRaises(ValueError):
            replace_local(self.local[:2], self.geometry, self.operator, baseline, screened)


if __name__ == '__main__':
    unittest.main()
