"""Compare the adaptations with isolated pinned author attention functions."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from dinotool.matched_neighborhood_controls import run_neighborhood
from dinotool.matched_readout_controls import run_head
from test_frozen_semantic_path import Block


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCES = WORKSPACE/"research/geometry_publication_20261001/public_sources"


def isolated_function(path, class_name, method):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch, "F": F, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method]


def mha_from(block):
    attention = torch.nn.MultiheadAttention(4, 2)
    with torch.no_grad():
        attention.in_proj_weight.copy_(block.attn.qkv.weight)
        attention.in_proj_bias.copy_(block.attn.qkv.bias)
        attention.out_proj.weight.copy_(block.attn.proj.weight)
        attention.out_proj.bias.copy_(block.attn.proj.bias)
    return attention


class PublicSourceFidelity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        for block in self.head.blocks:
            block.attn.scale = 2**-.5
        self.raw = torch.eye(4)[None]
        self.tokens = torch.randn(1, 5, 4)
        self.relation = torch.eye(4)[None]

    def test_sclip_author_csa(self):
        author = isolated_function(SOURCES/"sclip/clip/model.py", "VisionTransformer", "custom_attn")
        current = self.head.blocks[0](self.tokens)
        block = self.head.blocks[1]
        increment = author(None, mha_from(block), block.norm1(current).transpose(0, 1), csa=True).transpose(0, 1)
        current = current+block.ls1(increment)
        current = current+block.ls2(block.mlp(block.norm2(current)))
        expected = F.normalize(self.head.ln_final(current)[:, 1:], dim=-1)
        actual, _ = run_head(self.head, self.tokens, self.raw, self.relation, 1, "SCLIP_Last")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_proxyclip_author_attention_only(self):
        author = isolated_function(SOURCES/"proxyclip/open_clip/transformer.py", "VisionTransformer", "custom_attn")
        current = self.head.blocks[0](self.tokens)
        block = self.head.blocks[1]
        external = self.raw.transpose(1, 2).reshape(1, 4, 2, 2)
        observed = author(None, mha_from(block), block.norm1(current).transpose(0, 1),
                          ex_feats=external, beta=1.2, gamma=3., token_size=(2, 2)).transpose(0, 1)
        expected = F.normalize(self.head.ln_final(observed), dim=-1)
        actual, _ = run_head(self.head, self.tokens, self.raw, self.relation, 1, "ProxyCLIP_Last")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_naclip_author_gaussian_and_reduced_readout(self):
        source = SOURCES/"naclip/clip/model.py"
        window = isolated_function(source, "VisionTransformer", "gaussian_window")
        addition = isolated_function(source, "VisionTransformer", "get_attention_addition")
        author = isolated_function(source, "VisionTransformer", "custom_attn")
        holder = SimpleNamespace(gaussian_window=window, get_attention_addition=addition)
        author.__globals__["VisionTransformer"] = holder
        owner = SimpleNamespace(attn_strategy="naclip", gaussian_std=5., addition_cache={})
        current = self.head.blocks[0](self.tokens)
        block = self.head.blocks[1]
        observed = author(owner, mha_from(block), block.norm1(current).transpose(0, 1), (2, 2)).transpose(0, 1)
        expected = F.normalize(self.head.ln_final(observed)[:, 1:], dim=-1)
        prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=1, block_index=1,
            grid_height=2, grid_width=2, geometry_patch_conditional=self.relation)
        actual = run_neighborhood(self.head, prepared, "NACLIP_Last")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_vip_author_prefix_increment_and_two_blocks(self):
        source = WORKSPACE/"third_party/VIP_official/dinov3/eval/text/vision_tower.py"
        author = isolated_function(source, "VisionHead", "proxy_attn")
        tokens = torch.randn(1, 9, 4)
        current = tokens
        for block in self.head.blocks:
            increment = author(SimpleNamespace(patch_size=2), block.attn, block.norm1(current), self.raw)
            current = current+block.ls1(increment)
            current = current+block.ls2(block.mlp(block.norm2(current)))
        expected = F.normalize(self.head.ln_final(current)[:, 5:], dim=-1)
        actual, _ = run_head(self.head, tokens, self.raw, self.relation, 5, "VIPProxy_Two")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
