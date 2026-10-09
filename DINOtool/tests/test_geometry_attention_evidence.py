import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.geometry_attention_evidence import read_attention_evidence
from test_frozen_semantic_path import Block


class AttentionEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        self.head = SimpleNamespace(blocks=[Block(), Block()],
                                    ln_final=torch.nn.LayerNorm(4), linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=torch.randn(1, 6, 4), prefix_tokens=2,
                                        geometry_patch_conditional=torch.rand(1, 4, 4).softmax(-1),
                                        geometry_projected=F.normalize(torch.randn(1, 4, 4), dim=-1), block_index=1)

    def test_native_matches_original_head(self):
        current = self.prepared.backbone_tokens
        for block in self.head.blocks:
            current = block(current)
        expected = F.normalize(self.head.ln_final(current)[:, 2:], dim=-1)
        features, _ = read_attention_evidence(self.head, self.prepared)
        torch.testing.assert_close(features["Native"], expected)

    def test_relation_changes_evidence_not_native_donors(self):
        before, _ = read_attention_evidence(self.head, self.prepared)
        self.prepared.geometry_patch_conditional = torch.eye(4)[None]
        after, _ = read_attention_evidence(self.head, self.prepared)
        torch.testing.assert_close(before["Native"], after["Native"], rtol=0, atol=0)
        torch.testing.assert_close(before["EvidenceNative"], after["EvidenceNative"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before["EvidenceGeometry"], after["EvidenceGeometry"]))

    def test_no_explicit_residual_or_mlp_in_descriptor(self):
        state = self.prepared.backbone_tokens
        total = torch.zeros_like(state)
        for block in self.head.blocks:
            q, k, v = block.attn.qkv(block.norm1(state)).reshape(1, 6, 3, 2, 2).unbind(2)
            q, k, v = (part.transpose(1, 2) for part in (q, k, v))
            weights = ((q @ k.transpose(-1, -2))*.7).softmax(-1)
            increment = block.attn.proj((weights @ v).transpose(1, 2).reshape(1, 6, 4))
            total = total+increment
            state = block(state)
        expected = F.normalize(self.head.ln_final(total)[:, 2:], dim=-1)
        features, _ = read_attention_evidence(self.head, self.prepared)
        torch.testing.assert_close(features["EvidenceNative"], expected)
        self.assertFalse(torch.allclose(features["EvidenceNative"], features["Native"]))

    def test_uniform_matching_attention_has_equal_observations(self):
        for block in self.head.blocks:
            with torch.no_grad():
                block.attn.qkv.weight[:8].zero_()
                block.attn.qkv.bias[:8].zero_()
        self.prepared.geometry_patch_conditional.fill_(.25)
        features, _ = read_attention_evidence(self.head, self.prepared)
        torch.testing.assert_close(features["EvidenceNative"], features["EvidenceGeometry"], atol=1e-6, rtol=1e-6)

    def test_inputs_and_weights_are_unchanged(self):
        original = self.prepared.backbone_tokens.clone()
        weights = [block.attn.qkv.weight.clone() for block in self.head.blocks]
        read_attention_evidence(self.head, self.prepared)
        self.assertTrue(torch.equal(original, self.prepared.backbone_tokens))
        self.assertTrue(all(torch.equal(weight, block.attn.qkv.weight)
                            for weight, block in zip(weights, self.head.blocks)))

    def test_invalid_relation_rejected(self):
        self.prepared.geometry_patch_conditional[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            read_attention_evidence(self.head, self.prepared)


if __name__ == "__main__":
    unittest.main()
