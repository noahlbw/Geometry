import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.geometry_proxy_factors import (METHODS, PRIMARY, read_head, relations,
                                           restrict_relation, vip_block)
from dinotool.matched_readout_controls import proxy_relation, qkv, run_head
from dinotool.tcpr import _geometry_attended
from test_frozen_semantic_path import Block


class ProxyFactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.tokens = torch.randn(1, 6, 4)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, geometry_patch_conditional=self.g,
                                        prefix_tokens=2, block_index=1)

    def test_full_support_and_identity_support(self):
        torch.testing.assert_close(restrict_relation(self.g, torch.ones_like(self.g, dtype=torch.bool)), self.g)
        eye = torch.eye(4, dtype=torch.bool)[None]
        torch.testing.assert_close(restrict_relation(self.g, eye), eye.float())

    def test_zero_support_has_finite_identity_fallback(self):
        actual = restrict_relation(self.g, torch.zeros_like(self.g, dtype=torch.bool))
        torch.testing.assert_close(actual, torch.eye(4)[None])
        self.assertTrue(torch.isfinite(actual).all())

    def test_support_is_exact_pinned_global_not_row_mean(self):
        cache, _ = relations(self.prepared)
        raw = self.tokens[:, 2:]
        logits, _, _ = proxy_relation(raw, 1.5, .5, strict_zero=True)
        torch.testing.assert_close(cache['vip'], logits.softmax(-1), atol=0, rtol=0)
        torch.testing.assert_close(cache['vip_support'], restrict_relation(self.g, torch.isfinite(logits)), atol=0, rtol=0)

    def test_all_masked_pinned_rows_read_self(self):
        self.prepared.backbone_tokens = torch.ones_like(self.tokens)
        cache, diagnostics = relations(self.prepared)
        self.assertEqual(diagnostics['vip_empty_row_fraction'], 1.)
        torch.testing.assert_close(cache['vip'], torch.eye(4)[None])
        torch.testing.assert_close(cache['vip_support'], torch.eye(4)[None])

    def test_native_special_and_patch_mass_are_preserved(self):
        native = torch.rand(1, 2, 6, 6).softmax(-1)
        value = torch.randn(1, 2, 6, 2)
        g = relations(self.prepared)[0]['vip_support']
        explicit = native.clone()
        explicit[..., 2:, 2:] = native[..., 2:, 2:].sum(-1, keepdim=True)*g[:, None]
        actual = _geometry_attended(native, g, value, 2)
        torch.testing.assert_close(actual, explicit @ value)
        torch.testing.assert_close(explicit.sum(-1), native.sum(-1))
        torch.testing.assert_close(actual[..., :2, :], (native @ value)[..., :2, :])

    def test_all_four_historical_readouts_are_exact(self):
        for method in METHODS[:4]:
            expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, method, 1)[0]
            actual = read_head(self.head, self.prepared, method)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_vip_path_is_exact_operator_when_given_vip_relation(self):
        relation = proxy_relation(self.tokens[:, 2:], 1.5, .5, strict_zero=True)[0].softmax(-1)
        current = self.tokens
        for block in self.head.blocks:
            _, _, value = qkv(block, current)
            current = vip_block(block, current, value, relation, 2)
        expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, 'VIPProxy_Two', 1)[0]
        actual = F.normalize(self.head.ln_final(current)[:, 2:].float(), dim=-1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_factor_arms_are_finite_and_leave_inputs_and_weights_unchanged(self):
        tokens, relation = self.tokens.clone(), self.g.clone()
        parameters = [p for block in self.head.blocks for p in block.parameters()]
        old = [p.clone() for p in parameters]
        outputs = [read_head(self.head, self.prepared, method, trace=True)[0] for method in METHODS]
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
        self.assertTrue(torch.equal(tokens, self.tokens) and torch.equal(relation, self.g))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, parameters)))

    def test_invalid_method_and_non_square_grid_fail(self):
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, 'unknown')
        self.prepared.backbone_tokens = self.tokens[:, :-1]
        with self.assertRaises(ValueError):
            relations(self.prepared)


if __name__ == '__main__':
    unittest.main()
