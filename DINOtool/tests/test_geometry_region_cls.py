import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.geometry_region_cls import region_descriptors
from test_frozen_semantic_path import Block


class RegionCLSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(137)
        self.head = SimpleNamespace(blocks=[Block()], ln_final=torch.nn.LayerNorm(4), linear_projection=torch.nn.Identity())
        tokens = torch.randn(1, 6, 4)
        output = self.head.ln_final(self.head.blocks[0](tokens))
        self.prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, block_index=0,
            geometry_patch_conditional=torch.rand(1, 4, 4).softmax(-1),
            native_global=F.normalize(torch.cat((output[:, 0], output[:, 2:].mean(1)), -1), dim=-1))
        self.valid = torch.ones(1, 4, dtype=torch.bool)

    def test_native_global_is_exact(self):
        outputs, usable, diagnostics = region_descriptors(self.head, self.prepared, self.valid)
        self.assertTrue(usable.all())
        self.assertEqual(diagnostics["native_global_replay_max_error"], 0.)
        torch.testing.assert_close(outputs["NativeGlobal"][:, 0], self.prepared.native_global)

    def test_support_changes_virtual_but_not_native(self):
        first, _, _ = region_descriptors(self.head, self.prepared, self.valid)
        self.prepared.geometry_patch_conditional = torch.eye(4)[None]
        second, _, _ = region_descriptors(self.head, self.prepared, self.valid)
        self.assertFalse(torch.allclose(first["RegionCLS"], second["RegionCLS"]))
        torch.testing.assert_close(first["NativeGlobal"], second["NativeGlobal"])

    def test_uniform_support_produces_equal_virtual_queries(self):
        self.prepared.geometry_patch_conditional.fill_(.25)
        outputs, _, _ = region_descriptors(self.head, self.prepared, self.valid)
        torch.testing.assert_close(outputs["RegionFull"][:, :1].expand_as(outputs["RegionFull"]), outputs["RegionFull"])

    def test_inputs_and_weights_are_not_mutated(self):
        tokens = self.prepared.backbone_tokens.clone()
        relation = self.prepared.geometry_patch_conditional.clone()
        weight = self.head.blocks[0].attn.qkv.weight.clone()
        region_descriptors(self.head, self.prepared, self.valid)
        self.assertTrue(torch.equal(tokens, self.prepared.backbone_tokens))
        self.assertTrue(torch.equal(relation, self.prepared.geometry_patch_conditional))
        self.assertTrue(torch.equal(weight, self.head.blocks[0].attn.qkv.weight))

    def test_empty_valid_support_is_finite_and_unusable(self):
        outputs, usable, _ = region_descriptors(self.head, self.prepared, self.valid*False)
        self.assertFalse(usable.any())
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_two_block_native_global_is_exact(self):
        self.head.blocks.append(Block())
        self.prepared.block_index = 1
        output = self.head.ln_final(self.head.blocks[1](self.head.blocks[0](self.prepared.backbone_tokens)))
        self.prepared.native_global = F.normalize(torch.cat((output[:, 0], output[:, 2:].mean(1)), -1), dim=-1)
        _, _, diagnostics = region_descriptors(self.head, self.prepared, self.valid)
        self.assertEqual(diagnostics["native_global_replay_max_error"], 0.)


if __name__ == "__main__":
    unittest.main()
