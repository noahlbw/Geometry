"""Contracts for parallel support inference and fine-scale reobservation."""
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
from dinotool.cafe_support_reobservation import (
    CafeSupportReobservation,
    SupportReobservationConfig,
    config_from_checkpoint,
    ownership_supervision,
    soft_patch_labels,
    support_reobservation_loss,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList(
            nn.Linear(1024, 1024) for _ in range(4)
        )

    def encode_image_with_patch_tokens(self, images, normalize=False):
        del normalize
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            tokens = tokens + 0.02 * torch.tanh(block(tokens))
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        del q_chunk_size
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official():
    return CAFe_DINO(
        TinyVision(), None, TinyUpsampler(), (2, 2), "cpu",
        aggregator_dim=16, aggregator_blocks=2,
    ).eval()


class SupportReobservationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260927)
        torch.set_num_threads(2)
        stack = ExitStack()
        stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
        stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))
        self.addCleanup(stack.close)

    def test_partial_valid_patches_are_retained_by_soft_supervision(self):
        target = torch.full((1, 64, 64), 255, dtype=torch.long)
        target[:, :20, :32] = 0
        target[:, 20:24, :32] = 1
        proportions, coverage = soft_patch_labels(target, classes=3, height=2, width=2)
        self.assertGreater(float(coverage[0, 0, 0, 0]), 0.0)
        self.assertLess(float(coverage[0, 0, 0, 0]), 1.0)
        self.assertGreater(float(proportions[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(proportions[0, 1, 0, 0]), 0.0)
        loss = ownership_supervision(torch.randn(1, 9, 2, 2), target, classes=3)
        self.assertTrue(torch.isfinite(loss))

    def test_query_permutation_equivariance_and_class_count_agnostic(self):
        model = CafeSupportReobservation(
            official(), SupportReobservationConfig(hidden_dim=8)
        ).eval()
        image, text = torch.randn(1, 3, 64, 64), torch.randn(5, 1024)
        order = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            native = model(image, text)["logits"]
            permuted = model(image, text[order])["logits"]
            smaller = model(image, text[:2])["logits"]
        self.assertEqual(native.shape, (1, 5, 64, 64))
        self.assertEqual(smaller.shape, (1, 2, 64, 64))
        torch.testing.assert_close(permuted, native[:, order], rtol=5e-5, atol=1e-5)

    def test_gradients_frozen_contract_and_checkpoint_roundtrip(self):
        base = official()
        restored_base = copy.deepcopy(base)
        config = SupportReobservationConfig(hidden_dim=8)
        model = CafeSupportReobservation(base, config).train()
        image, text = torch.randn(1, 3, 64, 64), torch.randn(3, 1024)
        target = torch.randint(0, 3, (1, 64, 64))
        target[:, :8, :8] = 255
        output = model(image, text, ownership_target=target)
        loss, terms = support_reobservation_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms["fine_revision"]), 0.0)
        for prefix in ("semantic.", "ownership.", "reader."):
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(
                gradients and any(
                    gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                    for gradient in gradients
                ),
                prefix,
            )
        self.assertTrue(all(parameter.grad is None for parameter in model.cafe.parameters()))

        payload = {
            "format": model.checkpoint_format,
            "architecture": model.architecture(),
            "adapted_state": model.adapted_state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support_reobservation.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)
        restored = CafeSupportReobservation(
            restored_base, config_from_checkpoint(loaded)
        ).eval()
        restored.load_adapted_state_dict(loaded["adapted_state"])
        model.eval()
        with torch.no_grad():
            expected = model(image, text)["logits"]
            actual = restored(image, text)["logits"]
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
