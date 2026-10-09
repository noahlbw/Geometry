import unittest

import torch

from dinotool.geometry_semantic_innovation import anchored_innovation, semantic_innovation


class SemanticInnovationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.g = torch.randn(1, 4, 3)
        self.b = torch.randn(1, 4, 3)
        self.a = torch.rand(1, 4, 4).softmax(-1)
        self.valid = torch.ones(1, 4, dtype=torch.bool)

    def test_equal_observation_is_exact_identity(self):
        z, _ = semantic_innovation(self.g, self.g, self.a, self.valid)
        self.assertTrue(torch.equal(z, self.g))

    def test_uniform_support_preserves_local_differences(self):
        z, _ = semantic_innovation(self.g, self.b, torch.ones_like(self.a), self.valid)
        self.assertTrue(torch.allclose(z[:, 0]-z[:, 1], self.g[:, 0]-self.g[:, 1], atol=1e-6))
        self.assertTrue(torch.allclose(z.mean(1), self.b.mean(1), atol=1e-6))

    def test_identity_support_reads_observer(self):
        z, _ = semantic_innovation(self.g, self.b, torch.eye(4)[None], self.valid)
        self.assertTrue(torch.allclose(z, self.b, atol=1e-6))

    def test_class_permutation_equivariance(self):
        order = [2, 0, 1]
        z, _ = semantic_innovation(self.g, self.b, self.a, self.valid)
        swapped, _ = semantic_innovation(self.g[..., order], self.b[..., order], self.a, self.valid)
        self.assertTrue(torch.allclose(swapped, z[..., order]))

    def test_padding_donor_excluded_and_query_unchanged(self):
        self.valid[:, -1] = False
        z, _ = semantic_innovation(self.g, self.b, self.a, self.valid)
        self.b[:, -1] = 1e6
        other, _ = semantic_innovation(self.g, self.b, self.a, self.valid)
        self.assertTrue(torch.equal(z, other))
        self.assertTrue(torch.equal(z[:, -1], self.g[:, -1]))

    def test_no_donor_fallback(self):
        z, _ = semantic_innovation(self.g, self.b, self.a*0, self.valid)
        self.assertTrue(torch.equal(z, self.g))

    def test_inputs_not_mutated(self):
        before = [value.clone() for value in (self.g, self.b, self.a, self.valid)]
        semantic_innovation(self.g, self.b, self.a, self.valid)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (self.g, self.b, self.a, self.valid))))

    def test_invalid_relation_rejected(self):
        with self.assertRaises(ValueError):
            semantic_innovation(self.g, self.b, -self.a, self.valid)

    def test_anchored_identity_support_is_logit_average(self):
        z, _ = anchored_innovation(self.g, self.b, torch.eye(4)[None], self.valid)
        self.assertTrue(torch.allclose(z, .5*(self.g+self.b), atol=1e-6))

    def test_anchored_matches_direct_linear_solve(self):
        z, diagnostics = anchored_innovation(self.g, self.b, self.a, self.valid)
        at = self.a.transpose(-1, -2)
        expected = torch.linalg.solve(torch.eye(4)[None]+at@self.a, at@self.a@(self.b-self.g))
        self.assertTrue(torch.allclose(z, self.g+expected, atol=1e-6))
        self.assertLessEqual(diagnostics["energy_after"], diagnostics["energy_before"]+1e-6)
        self.assertLess(diagnostics["solver_relative_residual"], 1e-5)

    def test_anchored_identity_and_padding(self):
        self.valid[:, -1] = False
        z, _ = anchored_innovation(self.g, self.g, self.a, self.valid)
        self.assertTrue(torch.equal(z, self.g))
        z, _ = anchored_innovation(self.g, self.b, self.a, self.valid)
        self.b[:, -1] = 1e6
        other, _ = anchored_innovation(self.g, self.b, self.a, self.valid)
        self.assertTrue(torch.equal(z, other))
        self.assertTrue(torch.equal(z[:, -1], self.g[:, -1]))


if __name__ == "__main__":
    unittest.main()
