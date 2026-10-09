import unittest
from types import SimpleNamespace

import torch

from dinotool.geometry_semantic_response import (METHODS, CANDIDATES, read_head, semantic_block,
                                               self_attended, transport)
from dinotool.matched_readout_controls import run_head
from dinotool.tcpr import _geometry_attended, _intervened_head
from test_frozen_semantic_path import Block


class Zero(torch.nn.Module):
    def forward(self, value):
        return value*0


class SemanticResponseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        for block in self.head.blocks:
            block.mlp = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.GELU(), torch.nn.Linear(7, 4))
        self.tokens = torch.randn(1, 6, 4)
        self.g = torch.rand(1, 4, 4).softmax(-1)
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=2,
                                        geometry_patch_conditional=self.g, block_index=1)

    def test_conditional_self_read_preserves_mass_and_prefix(self):
        native = torch.randn(1, 2, 6, 6).softmax(-1)
        values = torch.randn(1, 2, 6, 2)
        actual = self_attended(native, values, 2)
        expected = _geometry_attended(native, torch.eye(4)[None], values, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_identity_relation_recovers_self_geometry(self):
        eye = torch.eye(4)[None]
        expected = _intervened_head(self.head, self.tokens, eye, 2, 1, 2, 'preserve')
        self.prepared.geometry_patch_conditional = eye
        expected = torch.nn.functional.normalize(expected[:, 2:].float(), dim=-1)
        for method in ('Geometry_ResponseTransport','Geometry_DonorBefore','Geometry_FullDonor',
                       'Geometry_QueryResponse','Geometry_SelfConditional'):
            actual = read_head(self.head, self.prepared, method)[0]
            torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)

    def test_affine_semantic_path_commutes_even_with_distinct_query_states(self):
        block = self.head.blocks[0]
        block.norm2 = torch.nn.Identity()
        block.mlp = torch.nn.Linear(4, 4)
        before = semantic_block(block, self.tokens, self.g, 2, 'before')[0]
        response, diagnostics = semantic_block(block, self.tokens, self.g, 2, 'response', trace=True)
        torch.testing.assert_close(before, response, atol=5e-7, rtol=5e-7)
        self.assertLess(diagnostics['nonlinear_order_gap_norm'], 5e-7)
        full = semantic_block(block, self.tokens, self.g, 2, 'full_donor')[0]
        self.assertFalse(torch.allclose(full[:, 2:], before[:, 2:]))

    def test_nonlinear_order_has_nonzero_effect(self):
        block = self.head.blocks[0]
        before = semantic_block(block, self.tokens, self.g, 2, 'before')[0]
        response, diagnostics = semantic_block(block, self.tokens, self.g, 2, 'response', trace=True)
        self.assertGreater(diagnostics['nonlinear_order_gap_norm'], 1e-4)
        self.assertFalse(torch.allclose(response[:, 2:], before[:, 2:]))

    def test_zero_attention_increment_has_zero_induced_response(self):
        block = self.head.blocks[0]
        block.ls1 = Zero()
        result, diagnostics = semantic_block(block, self.tokens, self.g, 2, 'response', trace=True)
        expected = self.tokens+block.ls2(block.mlp(block.norm2(self.tokens)))
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertEqual(diagnostics['response_norm'], 0.)

    def test_zero_mlp_reduces_to_attention_transport(self):
        block = self.head.blocks[0]
        block.ls2 = Zero()
        a = semantic_block(block, self.tokens, self.g, 2, 'response')[0]
        b = semantic_block(block, self.tokens, self.g, 2, 'before')[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_prefix_is_native_for_same_input_all_candidate_modes(self):
        block = self.head.blocks[0]
        native = block(self.tokens)
        for mode in ('response','before','full_donor','query_response','self'):
            result, _ = semantic_block(block, self.tokens, self.g, 2, mode)
            torch.testing.assert_close(result[:, :2], native[:, :2], atol=5e-7, rtol=5e-7)

    def test_original_four_methods_exact(self):
        for method in METHODS[:4]:
            expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, method, 1)[0]
            actual = read_head(self.head, self.prepared, method)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_finite_relation_controls_and_no_mutation(self):
        before, g = self.tokens.clone(), self.g.clone()
        parameters = [p for block in self.head.blocks for module in (block,block.attn.qkv,block.attn.proj)
                      for p in module.parameters()]
        copies = [p.clone() for p in parameters]
        for method in CANDIDATES:
            result, _ = read_head(self.head, self.prepared, method, trace=True)
            self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(before, self.tokens) and torch.equal(g, self.g))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(copies,parameters)))

    def test_transport_keeps_query_baseline_out_of_mixture(self):
        delta = torch.zeros(1, 4, 4)
        state = self.tokens[:, 2:]
        torch.testing.assert_close(state+transport(self.g,delta), state, atol=0, rtol=0)
        self.assertFalse(torch.allclose(transport(self.g,state), state))

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, 'unknown')


if __name__ == '__main__':
    unittest.main()
