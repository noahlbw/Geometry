import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.geometry_staged_readout import PRIMARY, POLICIES, attended_with_policy, read_head
from dinotool.matched_readout_controls import qkv
from dinotool.tcpr import _geometry_attended, _finish_attention_block, _finish_head, _intervened_head
from test_frozen_semantic_path import Block


class StagedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.tokens = torch.randn(1, 6, 4)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, geometry_patch_conditional=self.g,
                                        prefix_tokens=2, block_index=1)

    def test_original_and_blocked_heads_are_exact(self):
        for method, policy in (("Geometry", "preserve"), ("Geometry_BlockPrefix", "block")):
            expected = F.normalize(_intervened_head(self.head, self.tokens, self.g, 2, 1, 2, policy)[:, 2:].float(), dim=-1)
            actual, _ = read_head(self.head, self.prepared, method)
            self.assertTrue(torch.equal(actual, expected))

    def test_declared_stage_order_is_followed(self):
        for method, sequence in POLICIES.items():
            current = self.tokens
            for block, policy in zip(self.head.blocks, sequence):
                q, k, v = qkv(block, current)
                native = ((q @ k.transpose(-1, -2))*block.attn.scale).softmax(-1)
                current = _finish_attention_block(block, current, attended_with_policy(native, self.g, v, 2, policy))
            expected = F.normalize(_finish_head(self.head, current, 1)[:, 2:].float(), dim=-1)
            actual, _ = read_head(self.head, self.prepared, method)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_factorial_decomposes_attention_before_nonlinearity(self):
        native = torch.rand(1, 2, 6, 6).softmax(-1)
        value = torch.randn(1, 2, 6, 2)
        arms = {mode: attended_with_policy(native, self.g, value, 2, mode)
                for mode in ("preserve", "block", "prefix_unit", "patch_mass")}
        torch.testing.assert_close(arms["preserve"]-arms["patch_mass"], arms["prefix_unit"]-arms["block"])
        torch.testing.assert_close(arms["prefix_unit"]-arms["preserve"], arms["block"]-arms["patch_mass"])
        for result in arms.values():
            torch.testing.assert_close(result[..., :2, :], (native @ value)[..., :2, :])

    def test_patch_only_is_independent_of_special_values(self):
        native = torch.rand(1, 2, 6, 6).softmax(-1)
        value = torch.randn(1, 2, 6, 2)
        modified = value.clone(); modified[..., :2, :] += 100
        for mode in ("block", "patch_mass"):
            a = attended_with_policy(native, self.g, value, 2, mode)
            b = attended_with_policy(native, self.g, modified, 2, mode)
            self.assertTrue(torch.equal(a[..., 2:, :], b[..., 2:, :]))

    def test_input_parameters_and_relation_are_not_mutated(self):
        token, relation = self.tokens.clone(), self.g.clone()
        weights = [p.clone() for block in self.head.blocks for p in block.attn.qkv.parameters()]
        for method in POLICIES:
            read_head(self.head, self.prepared, method, trace=True)
        self.assertTrue(torch.equal(token, self.tokens))
        self.assertTrue(torch.equal(relation, self.g))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(weights, [p for block in self.head.blocks for p in block.attn.qkv.parameters()])))

    def test_first_stage_identity_and_nontrivial_order(self):
        _, plain = read_head(self.head, self.prepared, "Geometry", trace=True)
        primary, staged = read_head(self.head, self.prepared, PRIMARY, trace=True)
        for key in plain:
            if key.startswith("block0_"):
                self.assertEqual(plain[key], staged[key])
        reverse, _ = read_head(self.head, self.prepared, "Geometry_ReverseStage")
        self.assertFalse(torch.allclose(primary, reverse))

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, "unknown")
        self.prepared.block_index = 0
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, PRIMARY)


if __name__ == "__main__":
    unittest.main()
