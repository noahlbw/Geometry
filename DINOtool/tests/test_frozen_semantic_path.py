import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.frozen_semantic_path import read_frozen_path
from dinotool.tcpr import _finish_attention_block


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(qkv=torch.nn.Linear(4, 12), num_heads=2, scale=.7,
                                    proj=torch.nn.Linear(4, 4), proj_drop=torch.nn.Identity())
        self.norm1, self.norm2 = torch.nn.LayerNorm(4), torch.nn.LayerNorm(4)
        self.ls1, self.ls2, self.mlp = torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(4, 4)

    def forward(self, tokens):
        batch, count, channels = tokens.shape
        q, k, v = self.attn.qkv(self.norm1(tokens)).reshape(batch, count, 3, 2, 2).unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        weights = ((q.float() @ k.float().transpose(-1, -2))*.7).softmax(-1).to(v.dtype)
        return _finish_attention_block(self, tokens, weights @ v)


class FrozenPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.head = SimpleNamespace(blocks=[Block(), Block()],
                                    ln_final=torch.nn.LayerNorm(4), linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=torch.randn(1, 6, 4), prefix_tokens=2,
                                        geometry_patch_conditional=torch.rand(1, 4, 4).softmax(-1),
                                        geometry_projected=F.normalize(torch.randn(1, 4, 4), dim=-1), block_index=1)

    def test_native_and_mlp_follow_unedited_path(self):
        seen = []
        hooks = [block.mlp.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
                 for block in self.head.blocks]
        outputs, _ = read_frozen_path(self.head, self.prepared)
        for hook in hooks:
            hook.remove()
        current = self.prepared.backbone_tokens.clone()
        for index, block in enumerate(self.head.blocks):
            q, k, v = block.attn.qkv(block.norm1(current)).reshape(1, 6, 3, 2, 2).unbind(2)
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
            weights = ((q @ k.transpose(-1, -2))*.7).softmax(-1)
            increment = block.attn.proj((weights @ v).transpose(1, 2).reshape(1, 6, 4))
            attended = current+increment
            torch.testing.assert_close(seen[index], block.norm2(attended))
            current = attended+block.mlp(block.norm2(attended))
        expected = F.normalize(self.head.ln_final(current)[:, 2:], dim=-1)
        torch.testing.assert_close(outputs["Native"], expected)

    def test_prefix_is_unchanged_and_inputs_not_mutated(self):
        before = self.prepared.backbone_tokens.clone()
        weights = [b.attn.qkv.weight.clone() for b in self.head.blocks]
        _, diagnostics = read_frozen_path(self.head, self.prepared)
        self.assertEqual(diagnostics["prefix_transport_max"], 0.)
        self.assertTrue(torch.equal(before, self.prepared.backbone_tokens))
        self.assertTrue(all(torch.equal(w, b.attn.qkv.weight) for w, b in zip(weights, self.head.blocks)))

    def test_matching_relation_is_native_identity(self):
        for block in self.head.blocks:
            block.attn.qkv.weight.data[:8].zero_()
            block.attn.qkv.bias.data[:8].zero_()
        self.prepared.geometry_patch_conditional.fill_(.25)
        outputs, diagnostics = read_frozen_path(self.head, self.prepared)
        torch.testing.assert_close(outputs["FrozenPathGeometry"], outputs["Native"], atol=1e-6, rtol=1e-6)
        self.assertLess(diagnostics["mean_abs_state_transport"], 1e-7)

    def test_relation_really_changes_readout(self):
        outputs, _ = read_frozen_path(self.head, self.prepared)
        self.prepared.geometry_patch_conditional = torch.eye(4)[None]
        changed, _ = read_frozen_path(self.head, self.prepared)
        self.assertFalse(torch.allclose(outputs["FrozenPathGeometry"], changed["FrozenPathGeometry"]))
        torch.testing.assert_close(outputs["Native"], changed["Native"])

    def test_invalid_relation_rejected(self):
        self.prepared.geometry_patch_conditional.neg_()
        with self.assertRaises(ValueError):
            read_frozen_path(self.head, self.prepared)


if __name__ == "__main__":
    unittest.main()
