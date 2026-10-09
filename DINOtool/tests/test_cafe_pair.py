"""Numerical and CAFe interface checks; runnable with standard unittest."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(OFFICIAL / "CAFe_DINO")]
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_pca import CafePCA, PCADINOConfig, QueryAxisChannelAggregator
from dinotool.cafe_pair import CafePair, CafePairConfig, config_from_checkpoint
from dinotool.pair_fusion import PairComparisonFusion


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList(nn.Linear(1024, 1024) for _ in range(4))

    def encode_image_with_patch_tokens(self, images, normalize=False):
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            tokens = tokens + 0.05 * torch.tanh(block(tokens))
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official():
    return CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu",
                     aggregator_dim=16, aggregator_blocks=2).eval()


def fusion_inputs(queries=4, batch=2):
    return (torch.randn(batch, 8, queries, 3, 5), torch.randn(batch, 8, queries, 3, 5),
            torch.randn(batch, 12, 3, 5), torch.randn(queries, 10))


def fusion(**kwargs):
    return PairComparisonFusion(8, 12, 10, relation_dim=8, **kwargs)


class DeterministicTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260923)
        torch.set_num_threads(2)
        stack = ExitStack()
        stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
        stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))
        self.addCleanup(stack.close)


class FusionTests(DeterministicTest):
    def test_padding_and_coefficients(self):
        data = fusion_inputs()
        model = fusion().eval()
        out, trace = model(*data, return_trace=True)
        self.assertEqual(out.shape, data[0].shape)
        self.assertEqual(trace["weights"].shape, (2, 6, 15, 9))
        valid = F.unfold(torch.ones(2, 1, 3, 5), 3, padding=1).transpose(1, 2).bool()
        padded = trace["weights"].masked_select(~valid[:, None].expand_as(trace["weights"]))
        self.assertTrue(torch.equal(padded, torch.zeros_like(padded)))
        torch.testing.assert_close(trace["weights"].sum(-1), torch.ones(2, 6, 15))

    def test_manual_center_referenced_pair_read(self):
        data = fusion_inputs(batch=1, queries=3)
        model = fusion().eval()
        out, trace = model(*data, return_trace=True)
        semantic = data[1]
        expected = torch.zeros_like(semantic)
        for edge, (q, r) in enumerate(trace["pairs"].T.tolist()):
            for y in range(3):
                for x in range(5):
                    local = semantic[0, :, q, y, x] - semantic[0, :, r, y, x]
                    read = torch.zeros(8)
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if 0 <= y + dy < 3 and 0 <= x + dx < 5:
                                weight = trace["weights"][0, edge, y * 5 + x, (dy + 1) * 3 + dx + 1]
                                member = semantic[0, :, q, y + dy, x + dx] - semantic[0, :, r, y + dy, x + dx]
                                read = read + weight * (member - local)
                    message = model.message_projection(read) / 3
                    expected[0, :, q, y, x] += message
                    expected[0, :, r, y, x] -= message
        torch.testing.assert_close(trace["delta"], expected, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(out, semantic + expected)

    def test_query_permutation_equivariance(self):
        spatial, semantic, visual, text = fusion_inputs()
        model = fusion(pair_chunk=2).eval()
        order = torch.tensor([2, 0, 3, 1])
        reference = model(spatial, semantic, visual, text)
        permuted = model(spatial[:, :, order], semantic[:, :, order], visual, text[order])
        torch.testing.assert_close(permuted, reference[:, :, order], rtol=1e-5, atol=1e-6)

    def test_pair_reversal_preserves_support_and_reverses_message(self):
        spatial, semantic, visual, text = fusion_inputs(queries=2)
        model = fusion().eval()
        out, trace = model(spatial, semantic, visual, text, return_trace=True)
        other, reversed_trace = model(spatial.flip(2), semantic.flip(2), visual, text.flip(0), return_trace=True)
        torch.testing.assert_close(trace["weights"], reversed_trace["weights"])
        torch.testing.assert_close(trace["delta"][:, :, 0], -trace["delta"][:, :, 1])
        torch.testing.assert_close(other, out.flip(2))

    def test_uniform_semantic_regions_are_unchanged_including_corners(self):
        spatial, semantic, visual, text = fusion_inputs()
        semantic = semantic[..., :1, :1].expand_as(semantic).contiguous()
        torch.testing.assert_close(fusion()(spatial, semantic, visual, text), semantic, rtol=0, atol=0)

    def test_single_query_has_no_pair_correction(self):
        data = fusion_inputs(queries=1)
        result, trace = fusion()(*data, return_trace=True)
        torch.testing.assert_close(result, data[1], rtol=0, atol=0)
        self.assertEqual(trace["pairs"].shape, (2, 0))

    def test_cost_update_sums_to_zero(self):
        _, trace = fusion()(*fusion_inputs(), return_trace=True)
        torch.testing.assert_close(trace["delta"].sum(2), torch.zeros(2, 8, 3, 5), atol=1e-8, rtol=0)

    def test_chunking_and_activation_recomputation_preserve_gradients(self):
        first = fusion(pair_chunk=1, recompute_pairs=True).train()
        second = fusion(pair_chunk=32, recompute_pairs=False).train()
        second.load_state_dict(first.state_dict())
        data = fusion_inputs()
        a = [item.clone().requires_grad_() for item in data]
        b = [item.clone().requires_grad_() for item in data]
        output_a, output_b = first(*a), second(*b)
        torch.testing.assert_close(output_a, output_b, atol=1e-6, rtol=1e-5)
        objective = torch.randn_like(output_a)
        (output_a * objective).sum().backward()
        (output_b * objective).sum().backward()
        for left, right in zip(a, b):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)
        for (name, parameter), (_, other) in zip(first.named_parameters(), second.named_parameters()):
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)
        for prefix in ("spatial_", "semantic_", "text_", "visual_", "pair_condition", "message_projection"):
            self.assertGreater(sum(float(p.grad.abs().sum()) for name, p in first.named_parameters()
                                   if name.startswith(prefix)), 0, prefix)

    def test_bfloat16_autocast_is_finite(self):
        model = fusion().train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(*fusion_inputs())
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_shapes_are_rejected(self):
        spatial, semantic, visual, text = fusion_inputs()
        with self.assertRaises(ValueError):
            fusion()(spatial, semantic, visual, text[:2])
        with self.assertRaises(ValueError):
            fusion(kernel_size=2)

    def test_all_41_source_queries_remain_present(self):
        model = fusion(pair_chunk=16).eval()
        with torch.no_grad():
            output, trace = model(*fusion_inputs(queries=41, batch=1), return_trace=True)
        self.assertEqual(output.shape, (1, 8, 41, 3, 5))
        self.assertEqual(trace["pairs"].shape, (2, 820))
        self.assertTrue(torch.isfinite(output).all())


class ModelTests(DeterministicTest):
    def test_real_cafe_blocks_receive_gradients_and_checkpoint_roundtrips(self):
        base = official()
        restore_base = copy.deepcopy(base)
        config = CafePairConfig(stages=2, relation_dim=8, pair_chunk=2)
        model = CafePair(base, config).train()
        image, text = torch.randn(1, 3, 112, 224), torch.randn(3, 1024)
        output = model(image, text, return_aux=True)
        self.assertEqual(output["logits"].shape, (1, 3, 112, 224))
        self.assertEqual(output["fusion_update_rms"].shape, (2,))
        optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-4))
        F.cross_entropy(output["logits"], torch.randint(0, 3, (1, 112, 224))).backward()
        for prefix in ("spatial.0.", "channel.0.", "fusion.0.", "cafe.backbone.visual_model.backbone.blocks.3."):
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), prefix)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        before = model.fusion[0].message_projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fusion[0].message_projection.weight))
        payload = dict(format=model.checkpoint_format, architecture=model.architecture(), adapted_state=model.adapted_state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)
        restored = CafePair(restore_base, config_from_checkpoint(loaded)).eval()
        restored.load_adapted_state_dict(loaded["adapted_state"])
        model.eval()
        with torch.no_grad():
            expected = model(image, text)["logits"]
            torch.testing.assert_close(restored(image, text)["logits"], expected)
            torch.testing.assert_close(model(image, text, output_count=2)["logits"], expected[:, :2])
            order = torch.tensor([2, 0, 1])
            torch.testing.assert_close(model(image, text[order])["logits"], expected[:, order], atol=3e-6, rtol=2e-5)
        corrupt = dict(loaded["adapted_state"])
        corrupt.pop(next(iter(corrupt)))
        with self.assertRaises(ValueError):
            restored.load_adapted_state_dict(corrupt)

    def test_serial_attention_control_and_legacy_function_are_both_available(self):
        base = official()
        legacy = CafePCA(copy.deepcopy(base), PCADINOConfig(arm="serial", stages=2)).eval()
        corrected = CafePCA(copy.deepcopy(base), PCADINOConfig(arm="serial", stages=2, corrected_class_attention=True)).eval()
        self.assertIsInstance(corrected.cafe.aggregator[0].class_agg, QueryAxisChannelAggregator)
        image, text = torch.randn(1, 3, 112, 112), torch.randn(3, 1024)
        with torch.no_grad():
            torch.testing.assert_close(legacy(image, text)["logits"], base(image, text, pre_text_emb=True))
            result = corrected(image, text)["logits"]
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.shape, (1, 3, 112, 112))

    def test_config_rejects_wrong_checkpoint_family(self):
        with self.assertRaises(ValueError):
            config_from_checkpoint(dict(format="cafe_pca_v1", architecture={}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
