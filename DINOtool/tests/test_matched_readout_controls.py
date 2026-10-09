import unittest
from types import SimpleNamespace

import torch

from dinotool.matched_readout_controls import METHODS, proxy_relation, run_head
from dinotool.tcpr import _geometry_attended, _intervened_head
from test_frozen_semantic_path import Block


class MatchedReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.tokens = torch.randn(1, 6, 4)
        self.raw = torch.randn(1, 4, 4)
        self.relation = torch.rand(1, 4, 4).softmax(-1)

    def test_geometry_is_original_intervention(self):
        actual, _ = run_head(self.head, self.tokens, self.raw, self.relation, 2, "Geometry")
        expected = torch.nn.functional.normalize(_intervened_head(
            self.head, self.tokens, self.relation, 2, 1, 2, "preserve")[:, 2:], dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_csa_keeps_author_sum_and_residual_mlp(self):
        from dinotool.matched_readout_controls import qkv
        from dinotool.tcpr import _finish_attention_block
        current = self.head.blocks[0](self.tokens)
        block = self.head.blocks[1]
        q, k, v = qkv(block, current)
        weights = (q @ q.transpose(-1, -2)*.7).softmax(-1)+(k @ k.transpose(-1, -2)*.7).softmax(-1)
        torch.testing.assert_close(weights.sum(-1), torch.full_like(weights.sum(-1), 2.))
        expected = torch.nn.functional.normalize(self.head.ln_final(
            _finish_attention_block(block, current, weights @ v))[:, 2:], dim=-1)
        actual, _ = run_head(self.head, self.tokens, self.raw, self.relation, 2, "SCLIP_Last")
        torch.testing.assert_close(actual, expected)

    def test_empty_proxy_support_reads_only_self(self):
        logits, empty, count = proxy_relation(torch.ones(1, 4, 4), 1.5, .5, strict_zero=True)
        self.assertEqual((empty, count), (4, 4))
        torch.testing.assert_close(logits.softmax(-1), torch.eye(4)[None])

    def test_conditional_replacement_preserves_native_mass_and_prefix(self):
        native = torch.rand(1, 2, 6, 6).softmax(-1)
        values = torch.randn(1, 2, 6, 2)
        actual = _geometry_attended(native, self.relation, values, 2)
        replacement = native.clone()
        replacement[..., 2:, 2:] = native[..., 2:, 2:].sum(-1, keepdim=True)*self.relation[:, None]
        torch.testing.assert_close(replacement.sum(-1), torch.ones(1, 2, 6))
        torch.testing.assert_close(actual, replacement @ values)

    def test_all_methods_preserve_weights_and_inputs(self):
        before = self.tokens.clone()
        weights = [block.attn.qkv.weight.clone() for block in self.head.blocks]
        for method in METHODS:
            result, _ = run_head(self.head, self.tokens, self.raw, self.relation, 2, method)
            self.assertEqual(result.shape, (1, 4, 4))
            self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(torch.equal(self.tokens, before))
        for block, weight in zip(self.head.blocks, weights):
            self.assertTrue(torch.equal(block.attn.qkv.weight, weight))

    def test_attention_displacement_is_invariant_to_common_patch_value(self):
        native = torch.randn(1, 2, 6, 6).softmax(-1)
        values = torch.randn(1, 2, 6, 2)
        shifted = values.clone()
        shifted[..., 2:, :] += torch.randn(1, 2, 1, 2)
        delta = _geometry_attended(native, self.relation, values, 2)-native @ values
        shifted_delta = _geometry_attended(native, self.relation, shifted, 2)-native @ shifted
        torch.testing.assert_close(delta, shifted_delta, rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(delta[..., :2, :], torch.zeros_like(delta[..., :2, :]))


if __name__ == "__main__":
    unittest.main()
