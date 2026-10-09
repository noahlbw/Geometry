import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party" / "DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
if UPSTREAM.is_dir():
    sys.path.insert(0, str(UPSTREAM))

from dinov3.layers.block import SelfAttentionBlock
from dinotool.geometry_reacquisition import (
    GeometryReacquisition, NativeBlockRecord, ReacquisitionConfig,
    relation_bias, replay_block, split_qkv,
)
from dinotool.matched_readout_controls import run_head


def frozen_block():
    block = SelfAttentionBlock(16, 4, ffn_ratio=2., init_values=.2)
    block.ls1.reset_parameters()
    block.ls2.reset_parameters()
    return block.eval().requires_grad_(False)


class ReacquisitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261002)
        self.block = frozen_block()
        self.state = torch.randn(1, 7, 16)
        theta = torch.randn(4, 4)
        self.rope = (theta.sin(), theta.cos())
        self.record = NativeBlockRecord(self.state.clone(), self.rope)
        self.relation = torch.rand(1, 4, 4).softmax(-1)

    def test_uniform_relation_replays_actual_upstream_block_with_rope(self):
        actual, stats = replay_block(self.block, self.state[:, 3:], self.record,
                                    relation_bias(torch.ones(1, 4, 4), 4), 3)
        expected = self.block(self.state, self.rope)[:, 3:]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(stats["mean_abs_attention_displacement"], 0.)

    def test_fp32_matches_explicit_native_special_and_patch_mass(self):
        query_state = self.state[:, 3:] + .3 * torch.randn(1, 4, 16)
        actual, _ = replay_block(self.block, query_state, self.record,
                                 relation_bias(self.relation, 4), 3)
        q, k, v = split_qkv(self.block.attn, self.block.norm1(self.state))
        full = torch.cat((self.state[:, :3], query_state), dim=1)
        new_q, _, _ = split_qkv(self.block.attn, self.block.norm1(full))
        q, k = self.block.attn.apply_rope(q, k, self.rope)
        new_q, _ = self.block.attn.apply_rope(new_q, v, self.rope)
        native = ((q @ k.transpose(-1, -2)) * self.block.attn.scale).softmax(-1)
        patches = ((new_q[..., 3:, :] @ k[..., 3:, :].transpose(-1, -2)) *
                   self.block.attn.scale + self.relation.log()[:, None]).softmax(-1)
        read = native @ v
        read[..., 3:, :] = (native[..., 3:, :3] @ v[..., :3, :] +
                            native[..., 3:, 3:].sum(-1, keepdim=True) * (patches @ v[..., 3:, :]))
        attended = full + self.block.ls1(self.block.attn.proj(read.transpose(1, 2).reshape_as(full)))
        expected = attended + self.block.ls2(self.block.mlp(self.block.norm2(attended)))
        torch.testing.assert_close(actual, expected[:, 3:], atol=4e-7, rtol=1e-6)

    def test_rope_is_used_at_original_patch_coordinates(self):
        bias = relation_bias(self.relation, 4)
        actual, _ = replay_block(self.block, self.state[:, 3:], self.record, bias, 3)
        without, _ = replay_block(self.block, self.state[:, 3:],
                                  NativeBlockRecord(self.state, None), bias, 3)
        self.assertGreater(float((actual-without).abs().max()), 1e-5)

    def test_native_donors_positions_and_parameters_are_immutable(self):
        before = self.record.state.clone()
        parameters = [p.clone() for p in self.block.parameters()]
        positions = [part.clone() for part in self.rope]
        replay_block(self.block, self.state[:, 3:], self.record,
                     relation_bias(self.relation, 4), 3)
        self.assertTrue(torch.equal(before, self.record.state))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(parameters, self.block.parameters())))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(positions, self.rope)))

    def test_sparse_relation_is_finite_and_changes_actual_queries(self):
        actual, stats = replay_block(self.block, self.state[:, 3:], self.record,
                                    relation_bias(torch.eye(4)[None], 4), 3)
        native = self.block(self.state, self.rope)[:, 3:]
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreater(stats["mean_abs_attention_displacement"], 1e-5)
        self.assertFalse(torch.allclose(actual, native))

    def test_row_scaling_does_not_change_conditional_reading(self):
        actual, _ = replay_block(self.block, self.state[:, 3:], self.record,
                                 relation_bias(self.relation, 4), 3)
        scaled, _ = replay_block(self.block, self.state[:, 3:], self.record,
                                 relation_bias(self.relation * torch.tensor([1., 2., 7., .2])[None, :, None], 4), 3)
        torch.testing.assert_close(actual, scaled, atol=4e-7, rtol=1e-6)

    def test_invalid_relations_and_training_are_rejected(self):
        for bad in (-self.relation, torch.zeros_like(self.relation),
                    self.relation.clone().fill_(float("nan"))):
            with self.assertRaises(ValueError):
                relation_bias(bad, 4)
        self.block.train()
        with self.assertRaises(ValueError):
            replay_block(self.block, self.state[:, 3:], self.record, None, 3)

    def test_complete_replay_preserves_native_capture_and_final_prefix(self):
        blocks = torch.nn.ModuleList([frozen_block() for _ in range(3)])
        native_backbone = SimpleNamespace(blocks=blocks, norm=torch.nn.LayerNorm(16).requires_grad_(False), training=False)
        head = SimpleNamespace(blocks=torch.nn.ModuleList([frozen_block() for _ in range(2)]),
            ln_final=torch.nn.LayerNorm(16).requires_grad_(False), linear_projection=torch.nn.Identity())
        visual = SimpleNamespace(backbone=native_backbone, head=head, patch_token_layer=1)
        backbone = SimpleNamespace(model=SimpleNamespace(visual_model=visual), _autocast=nullcontext)
        reader = GeometryReacquisition(backbone, ReacquisitionConfig(last_blocks=2))
        try:
            native = self.state
            for block in blocks:
                native = block(native, self.rope)
            tokens = native_backbone.norm(native)
            geometry, _ = run_head(head, tokens, tokens[:, 3:], self.relation, 3, "Geometry", 1)
            prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=3,
                geometry_patch_conditional=self.relation, geometry_projected=geometry, block_index=1)
            original = [record.state.clone() for record in reader.records]
            uniform, _ = reader.replay(prepared, torch.ones_like(self.relation))
            torch.testing.assert_close(uniform, tokens, atol=0, rtol=0)
            uniform_geometry, _ = run_head(head, uniform, uniform[:, 3:], self.relation, 3, "Geometry", 1)
            torch.testing.assert_close(uniform_geometry, geometry, atol=0, rtol=0)
            changed, stats = reader.replay(prepared)
            self.assertGreater(stats["mean_abs_backbone_change"], 1e-5)
            self.assertEqual(stats["backbone_prefix_change"], 0.)
            self.assertTrue(all(torch.equal(old, record.state) for old, record in zip(original, reader.records)))
            self.assertTrue(torch.equal(changed[:, :3], tokens[:, :3]))
        finally:
            reader.close()
        self.assertFalse(reader.hooks)
        self.assertFalse(any(block._forward_pre_hooks for block in blocks))


if __name__ == "__main__":
    unittest.main()
