"""Meaningful invariants for conditional, fixed-slot positive-excess use."""
import unittest

import torch

from dinotool.semantic_contract_alias import attenuate_log_evidence, conditional_delta, Plan


class ContractTests(unittest.TestCase):
    def test_log_formula_and_no_inflation(self):
        x = torch.tensor([-1000., -3., 1., 5., 1000.], dtype=torch.float64)
        b = torch.zeros_like(x)
        for w in (0., .5, 1.):
            weight = torch.full_like(x, w)
            result = attenuate_log_evidence(x, b, weight.log(), (1-weight).log())
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result <= x+1e-12).all())
            self.assertTrue(torch.equal(result[x <= b], x[x <= b]))
            if w == 1:
                self.assertTrue(torch.equal(result, x))
            small = x.abs() < 100
            expected = torch.minimum(x[small].exp(), b[small].exp()) + w*(x[small].exp()-b[small].exp()).clamp_min(0)
            torch.testing.assert_close(result[small].exp(), expected)

    def test_rival_changes_word_action_unknown_abstains(self):
        lookup = torch.full((3, 3), -1, dtype=torch.long); lookup[0, 1] = 0
        plan = Plan(None, (), ((0, 1),), None, None, lookup, ())
        fields = torch.tensor([[-2.], [-2.], [-2.]])
        pairs = torch.tensor([[0, 1], [0, 2], [2, 1]])
        delta, active = conditional_delta(fields, pairs, plan, 3)
        torch.testing.assert_close(delta, torch.tensor([[-2., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
        self.assertEqual(active.sum().item(), 1)

    def test_no_contract_is_exact_zero(self):
        lookup = torch.full((3, 3), -1, dtype=torch.long)
        plan = Plan(None, (), (), None, None, lookup, ())
        delta, active = conditional_delta(torch.empty(2, 0), torch.tensor([[0, 1], [1, 2]]), plan, 3)
        self.assertTrue(torch.equal(delta, torch.zeros(2, 3)))
        self.assertFalse(active.any())


if __name__ == '__main__':
    unittest.main()
