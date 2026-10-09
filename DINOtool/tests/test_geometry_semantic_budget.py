import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.geometry_semantic_budget import project_relation, relation_for_head, shell_attended, read_head
from dinotool.tcpr import _intervened_head
from test_frozen_semantic_path import Block


class BudgetTests(unittest.TestCase):
    def test_binary_closed_form_and_inactive_identity(self):
        logits = torch.tensor([[.2, .8], [.8, .2]]).log()
        cost = torch.tensor([[0., 2.], [0., 2.]])
        result, stats = project_relation(logits, cost, torch.tensor([.6, .6]))
        torch.testing.assert_close(result[0], torch.tensor([.7, .3]), atol=2e-6, rtol=0)
        self.assertTrue(torch.equal(result[1], logits.softmax(-1)[1]))
        self.assertLess(stats["constraint_violation"], 2e-5)
        self.assertEqual(stats["active_fraction"], .5)

    def test_nonnegative_mass_and_constraint(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 3, 9)
        cost = torch.rand_like(logits)*5
        budget = cost.min(-1).values+.2
        relation, _ = project_relation(logits, cost, budget)
        torch.testing.assert_close(relation.sum(-1), torch.ones_like(budget))
        self.assertTrue(bool((relation >= 0).all()))
        self.assertLess(float(((relation*cost).sum(-1)-budget).max()), 2e-5)

    def test_masked_donors_remain_zero(self):
        result, _ = project_relation(torch.tensor([[0., -torch.inf, 0.]]),
                                    torch.tensor([[0., 0., 2.]]), torch.tensor([.4]))
        self.assertEqual(float(result[0, 1]), 0.)

    def test_infeasible_and_empty_rejected(self):
        for logits, cost, budget in (([[0., 0.]], [[1., 2.]], [.5]),
                                     ([[-torch.inf, -torch.inf]], [[0., 2.]], [.5])):
            with self.assertRaises(ValueError):
                project_relation(torch.tensor(logits), torch.tensor(cost), torch.tensor(budget))

    def test_valid_edges_only_and_prefix_shell(self):
        torch.manual_seed(11)
        q, k, v = (torch.randn(1, 2, 6, 3) for _ in range(3))
        g = torch.randn(1, 4, 4).softmax(-1)
        logits = torch.randn(1, 4, 4)
        valid = torch.tensor([[True, True, False, True]])
        relation, stats = relation_for_head(q, k, g, logits, valid, 2, .7, "budget")
        torch.testing.assert_close(relation.sum(-1), g[:, None].expand(1, 2, 4, 4).sum(-1))
        self.assertEqual(stats["invalid_edge_error"], 0.)
        native = (q @ k.transpose(-1, -2)).softmax(-1)
        attended = shell_attended(native, relation, v, 2)
        torch.testing.assert_close(attended[..., :2, :], (native @ v)[..., :2, :], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(relation[..., 2, :], g[:, None, 2, :].expand(1, 2, 4))
        self.assertFalse(torch.allclose(relation[:, 0], relation[:, 1]))

    def test_original_geometry_bit_exact_and_unmutated(self):
        torch.manual_seed(2)
        head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                               linear_projection=torch.nn.Identity())
        tokens = torch.randn(1, 6, 4)
        g = torch.randn(1, 4, 4).softmax(-1)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, geometry_patch_conditional=g,
                                   grid_height=2, grid_width=2, block_index=1)
        original = tokens.clone()
        expected = F.normalize(_intervened_head(head, tokens, g, 2, 1, 2, "preserve")[:, 2:].float(), dim=-1)
        result, _ = read_head(head, prepared, torch.ones(1, 4, dtype=torch.bool), "geometry")
        self.assertTrue(torch.equal(result, expected))
        read_head(head, prepared, torch.ones(1, 4, dtype=torch.bool), "budget")
        self.assertTrue(torch.equal(tokens, original))

    def test_bf16_proxy_logits_are_normalized_in_float32(self):
        torch.manual_seed(17)
        q, k = (torch.randn(1, 2, 66, 3) for _ in range(2))
        g = torch.randn(1, 64, 64).softmax(-1).bfloat16()
        logits = torch.randn(1, 64, 64).bfloat16()
        relation, stats = relation_for_head(q, k, g, logits, torch.ones(1, 64, dtype=torch.bool), 2, .7, "proxy")
        self.assertEqual(relation.dtype, torch.float32)
        self.assertLess(stats["row_mass_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
