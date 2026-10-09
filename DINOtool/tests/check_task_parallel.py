"""Dependency-light preflight for the eight-GPU task-parallel training run."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ["CAFE_OFFICIAL_ROOT"])
sys.path[:0] = [str(ROOT), str(OFFICIAL / "CAFe_DINO")]

from modeling.cafedino import CAFe_DINO
from dinotool.cafe_pca import CafePCA, PCADINOConfig, TaskGatedParallelFusion


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(2)])

    def encode_image_with_patch_tokens(self, images, normalize=False):
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, images, features, q_chunk_size=None):
        return F.interpolate(features, size=images.shape[-2:], mode="bilinear", align_corners=False)


def main() -> None:
    torch.manual_seed(20260928)
    torch.set_num_threads(2)
    gate = TaskGatedParallelFusion(16)
    original = torch.randn(1, 16, 3, 2, 2)
    spatial = torch.randn_like(original)
    semantic = torch.randn_like(original)
    fused, weights = gate(original, spatial, semantic, return_weights=True)
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:, :, 0]))
    torch.testing.assert_close(fused, 0.5 * original + 0.25 * spatial + 0.25 * semantic)

    base = CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu",
                     aggregator_dim=16, aggregator_blocks=1).eval()
    pretrained = CafePCA(copy.deepcopy(base), PCADINOConfig(
        arm="task_parallel", stages=1, corrected_class_attention=True,
        visual_tune_blocks=0,
    ))
    fresh = CafePCA(copy.deepcopy(base), PCADINOConfig(
        arm="task_parallel", stages=1, corrected_class_attention=True,
        channel_init="fresh", visual_tune_blocks=0,
    ))
    for left, right in zip(pretrained.spatial.parameters(), fresh.spatial.parameters()):
        torch.testing.assert_close(left, right)
    assert any(not torch.equal(left, right) for left, right in
               zip(pretrained.channel.parameters(), fresh.channel.parameters()))
    assert all(not p.requires_grad for p in fresh.cafe.backbone.parameters())
    assert all(p.requires_grad for p in fresh.spatial.parameters())
    assert all(p.requires_grad for p in fresh.channel.parameters())
    assert fresh.checkpoint_format == "cafe_task_parallel_v1"
    assert fresh.architecture()["name"] == "DINO-TaskParallel-v1"
    assert fresh.architecture()["epl"] == "none"

    images = torch.randn(1, 3, 112, 112)
    text = F.normalize(torch.randn(4, 1024), dim=-1)
    target = torch.randint(0, 4, (1, 112, 112))
    result = fresh(images, text, return_aux=True)
    assert result["logits"].shape == (1, 4, 112, 112)
    loss = F.cross_entropy(result["logits"], target)
    loss.backward()
    for prefix in ("spatial.0.", "channel.0.", "fusion.0."):
        gradients = [parameter.grad for name, parameter in fresh.named_parameters()
                     if name.startswith(prefix)]
        assert gradients and any(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
                                 for value in gradients), prefix
    print("task_parallel_preflight_passed", flush=True)


if __name__ == "__main__":
    main()
