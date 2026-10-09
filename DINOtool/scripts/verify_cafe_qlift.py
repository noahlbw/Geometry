#!/usr/bin/env python3
"""Dependency-free numerical and CAFe contracts for Q-Lift-DINO."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("CAFE_OFFICIAL_ROOT", ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"))
sys.path[:0] = [str(ROOT), str(OFFICIAL / "CAFe_DINO")]

from modeling.cafedino import CAFe_DINO
from dinotool.cafe_qlift import CafeQLift, config_from_checkpoint
from dinotool.qlift import QLiftConfig, QLiftDecoder


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList(nn.Linear(1024, 1024) for _ in range(4))

    def encode_image_with_patch_tokens(self, images, normalize=False):
        del normalize
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            tokens = tokens + 0.05 * torch.tanh(block(tokens))
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        del q_chunk_size
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official() -> nn.Module:
    return CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


def config(arm: str, **changes) -> QLiftConfig:
    return QLiftConfig(arm=arm, levels=2, guide_dim=32, hidden_dim=64, kernel=3, query_chunk=2, **changes)


class QLiftVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(419)
        torch.set_num_threads(2)

    def test_all_arms_reconstruct_odd_rectangular_cost(self) -> None:
        cost = torch.randn(1, 8, 3, 7, 11)
        x, y, text = torch.randn(1, 1024, 7, 11), torch.randn(1, 1024, 7, 11), torch.randn(3, 1024)
        for arm in ("skip", "haar", "image_lift", "query_lift"):
            with self.subTest(arm=arm), torch.no_grad():
                output, trace = QLiftDecoder(8, config(arm)).eval()(cost, x, y, text, return_trace=True)
                torch.testing.assert_close(output, cost, rtol=2e-5, atol=2e-6)
                self.assertLess(float(trace["max_cost_change"]), 2e-5)

    def test_conditioning_contract_and_cached_details(self) -> None:
        guide, text, other = torch.randn(2, 32, 5, 7), torch.randn(2, 1024), torch.randn(2, 1024)
        image, query = QLiftDecoder(8, config("image_lift")), QLiftDecoder(8, config("query_lift"))
        assert image.levels[0].lifting is not None and query.levels[0].lifting is not None
        with torch.no_grad():
            p0, u0 = image.levels[0].lifting._weights(guide, text, query_conditioned=False)
            p1, u1 = image.levels[0].lifting._weights(guide, other, query_conditioned=False)
            pq0, uq0 = query.levels[0].lifting._weights(guide, text, query_conditioned=True)
            pq1, uq1 = query.levels[0].lifting._weights(guide, other, query_conditioned=True)
        torch.testing.assert_close(p0, p1, rtol=0, atol=0)
        torch.testing.assert_close(u0, u1, rtol=0, atol=0)
        self.assertGreater(float((pq0 - pq1).abs().max()), 1e-6)
        self.assertGreater(float((uq0 - uq1).abs().max()), 1e-6)
        value = torch.randn(1, 8, 8, 10)
        approximation, details, cache = query.levels[0].lifting.analyze(value, guide[:1, :, :4, :5], text[:1], query_conditioned=True)
        base = query.levels[0].lifting.synthesize(approximation, details, cache)
        torch.testing.assert_close(base, value, rtol=2e-5, atol=2e-6)
        for stream in range(3):
            changed = details.clone()
            changed[:, stream, :, 1, 2] += 0.5
            self.assertFalse(torch.allclose(query.levels[0].lifting.synthesize(approximation, changed, cache), base))

    def test_pu_counterfactual_changes_complete_pu_pathway_with_native_semantic_inputs_fixed(self) -> None:
        cost = torch.randn(1, 8, 3, 7, 11)
        x, y, text = torch.randn(1, 1024, 7, 11), torch.randn(1, 1024, 7, 11), torch.randn(3, 1024)
        query = QLiftDecoder(8, config("query_lift")).eval()
        image = QLiftDecoder(8, config("image_lift")).eval()
        with torch.no_grad():
            for decoder in (query, image):
                for level in decoder.levels:
                    level.context.output.weight.normal_(std=0.02)
                    level.detail.output.weight.normal_(std=0.02)
            native, _ = query(cost, x, y, text)
            shuffled, _ = query(cost, x, y, text, lifting_text=text.roll(1, 0))
            image_only, _ = query(cost, x, y, text, lifting_query_conditioned=False)
            details_zeroed, _ = query(cost, x, y, text, detail_mode="zero")
            image_native, _ = image(cost, x, y, text)
            image_other, _ = image(cost, x, y, text, lifting_text=text.roll(1, 0))
        self.assertGreater(float((native - shuffled).abs().max()), 1e-6)
        self.assertGreater(float((native - image_only).abs().max()), 1e-6)
        self.assertGreater(float((native - details_zeroed).abs().max()), 1e-6)
        torch.testing.assert_close(image_native, image_other, rtol=0, atol=0)

    def test_same_seed_preserves_shared_control_initialization(self) -> None:
        torch.manual_seed(7741)
        skip = QLiftDecoder(8, config("skip"))
        torch.manual_seed(7741)
        query = QLiftDecoder(8, config("query_lift"))
        skip_state, query_state = skip.state_dict(), query.state_dict()
        self.assertEqual(set(skip_state), set(query_state))
        for name in skip_state:
            torch.testing.assert_close(skip_state[name], query_state[name], rtol=0, atol=0)
        counts = []
        for arm in ("skip", "haar", "image_lift", "query_lift"):
            counts.append(sum(parameter.numel() for parameter in QLiftDecoder(8, config(arm)).parameters() if parameter.requires_grad))
        self.assertEqual(len(set(counts)), 1)

    def test_cafer_adapter_warmup_gradients_and_checkpoint(self) -> None:
        base = official()
        restored_base = copy.deepcopy(base)
        model = CafeQLift(base, config("query_lift")).train()
        image, text = torch.randn(1, 3, 112, 224), F.normalize(torch.randn(3, 1024), dim=-1)
        target = torch.randint(0, 3, (1, 112, 224))
        optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-3))
        corr_before = model.cafe.corr_embed.weight.detach().clone()
        model.set_warmup(True)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image, text)["logits"], target).backward()
        optimizer.step()
        torch.testing.assert_close(model.cafe.corr_embed.weight, corr_before, rtol=0, atol=0)
        model.set_warmup(False)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(image, text)["logits"], target).backward()
            optimizer.step()
        self.assertFalse(torch.equal(model.cafe.corr_embed.weight, corr_before))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        payload = {"format": model.checkpoint_format, "architecture": model.architecture(), "adapted_state": model.adapted_state_dict()}
        restored = CafeQLift(restored_base, config_from_checkpoint(payload)).eval()
        restored.load_adapted_state_dict(payload["adapted_state"])
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"], rtol=2e-5, atol=2e-6)


def main() -> None:
    result = unittest.main(module=__name__, verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
