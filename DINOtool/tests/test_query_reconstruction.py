"""Contracts for query-conditioned evidence-preserving reconstruction."""
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
OFFICIAL = Path(os.getenv(
    "CAFE_OFFICIAL_ROOT",
    str(ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"),
))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(OFFICIAL / "CAFe_DINO")]
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_query_reconstruction import (
    CafeQueryReconstruction,
    EvidencePreservingQueryReconstructor,
    QueryReconstructionConfig,
    config_from_checkpoint,
    query_reconstruction_loss,
)


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


class ForbiddenUpsampler(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("AnyUp must not be used by query reconstruction")


def official():
    return CAFe_DINO(
        TinyVision(), None, ForbiddenUpsampler(), (7, 7), "cpu",
        aggregator_dim=16, aggregator_blocks=2,
    ).eval()


class QueryReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260926)
        torch.set_num_threads(2)
        stack = ExitStack()
        stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
        stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))
        self.addCleanup(stack.close)

    def test_reconstructor_preserves_cost_patch_means(self):
        module = EvidencePreservingQueryReconstructor(8, 32, 8, 0.1).eval()
        images = torch.randn(2, 3, 64, 96)
        cost = torch.randn(2, 8, 5, 4, 6)
        text = torch.randn(5, 32)
        with torch.no_grad():
            features, error = module(images, cost, text)
        pooled = F.adaptive_avg_pool2d(features, (4, 6))
        expected = cost.permute(0, 2, 1, 3, 4).reshape(10, 8, 4, 6)
        torch.testing.assert_close(pooled, expected, rtol=2e-5, atol=2e-6)
        self.assertLess(float(error), 1e-10)

    def test_model_is_query_permutation_equivariant_and_class_count_agnostic(self):
        model = CafeQueryReconstruction(
            official(), QueryReconstructionConfig(stages=2, relation_dim=8, guide_dim=8)
        ).eval()
        image, text = torch.randn(1, 3, 112, 224), torch.randn(5, 1024)
        order = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            native = model(image, text)["logits"]
            permuted = model(image, text[order])["logits"]
            smaller = model(image, text[:2])["logits"]
        self.assertEqual(native.shape, (1, 5, 112, 224))
        self.assertEqual(smaller.shape, (1, 2, 112, 224))
        torch.testing.assert_close(permuted, native[:, order], rtol=3e-5, atol=5e-6)

    def test_gradients_checkpoint_roundtrip_and_frozen_contract(self):
        base = official()
        restored_base = copy.deepcopy(base)
        config = QueryReconstructionConfig(stages=2, relation_dim=8, guide_dim=8)
        model = CafeQueryReconstruction(base, config).train()
        image, text = torch.randn(1, 3, 112, 112), torch.randn(3, 1024)
        target = torch.randint(0, 3, (1, 112, 112))
        target[:, :16, :16] = 255
        output = model(image, text, relation_target=target)
        loss, terms = query_reconstruction_loss(output, target, torch.randn_like(output["logits"]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms["coarse_consistency_loss"]))
        for prefix in (
            "fusion.0.spatial_relation.", "fusion.0.class_relation.",
            "reconstructor.image_encoder.", "reconstructor.cost_encoder.",
            "reconstructor.text_modulation.", "reconstructor.residual_projection.",
        ):
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(
                gradients and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                                  for g in gradients),
                prefix,
            )
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.cafe.upsampler.parameters()))

        payload = dict(
            format=model.checkpoint_format,
            architecture=model.architecture(),
            adapted_state=model.adapted_state_dict(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_reconstruction.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)
        restored = CafeQueryReconstruction(restored_base, config_from_checkpoint(loaded)).eval()
        restored.load_adapted_state_dict(loaded["adapted_state"])
        model.eval()
        with torch.no_grad():
            expected = model(image, text)["logits"]
            actual = restored(image, text)["logits"]
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
