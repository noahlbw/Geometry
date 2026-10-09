"""PCA-DINO topology, FOD, attention-axis, and checkpoint contracts."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path[:0] = [str(ROOT), str(OFFICIAL / "CAFe_DINO")]
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_pca import (
    CafePCA,
    ExpertDrivenPerceptualLearning,
    PCADINOConfig,
    QueryAxisChannelAggregator,
    TaskGatedParallelFusion,
    feature_orthogonalization_loss,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(4)])

    def encode_image_with_patch_tokens(self, images, normalize=False):
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            tokens = tokens + 0.05 * torch.tanh(block(tokens))
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official(blocks=2):
    return CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu", aggregator_dim=16,
                     aggregator_blocks=blocks).eval()


def config(arm="pca_epl_fod", *, corrected=False, stages=2):
    return PCADINOConfig(arm=arm, experts=4, reduction_ratio=4, stages=stages,
                         corrected_class_attention=corrected)


def inputs(classes=4, height=112, width=112):
    return torch.randn(1, 3, height, width), F.normalize(torch.randn(classes, 1024), dim=-1)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(923)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


@pytest.mark.parametrize("arm", ["serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"])
def test_all_arms_preserve_multichannel_cafe_output_contract(arm):
    model = CafePCA(official(), config(arm)).eval()
    image, text = inputs(height=112, width=224)
    with torch.no_grad():
        result = model(image, text, return_aux=True)
    assert result["logits"].shape == (1, len(text), 112, 224)
    assert torch.isfinite(result["logits"]).all()
    assert torch.isfinite(result["fod_loss"])
    assert len(model.cafe.aggregator) == (2 if arm == "serial" else 0)
    assert len(model.fusion) == (2 if arm.startswith("pca_epl") or arm == "task_parallel" else 0)


def test_task_gate_starts_as_anchored_convex_fusion_and_trains_both_branches():
    gate = TaskGatedParallelFusion(16).eval()
    original = torch.randn(1, 16, 3, 2, 2)
    spatial = torch.randn_like(original)
    semantic = torch.randn_like(original)
    fused, weights = gate(original, spatial, semantic, return_weights=True)
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:, :, 0]))
    torch.testing.assert_close(fused, 0.5 * original + 0.25 * spatial + 0.25 * semantic)

    model = CafePCA(official(), PCADINOConfig(
        arm="task_parallel", stages=2, corrected_class_attention=True,
        channel_init="fresh", visual_tune_blocks=0,
    ))
    image, text = inputs()
    target = torch.randint(0, len(text), (1, 112, 112))
    result = model(image, text, return_aux=True)
    loss = F.cross_entropy(result["logits"], target)
    loss.backward()
    for prefix in ("spatial.0.", "channel.0.", "fusion.0."):
        gradients = [parameter.grad for name, parameter in model.named_parameters()
                     if name.startswith(prefix)]
        assert gradients and any(item is not None and torch.isfinite(item).all() and item.abs().sum() > 0
                                 for item in gradients), prefix
    assert all(not p.requires_grad for p in model.cafe.backbone.visual_model.backbone.blocks.parameters())


def test_fresh_class_initialization_keeps_spatial_pretraining_and_resets_class_weights():
    base = official()
    pretrained = CafePCA(copy.deepcopy(base), PCADINOConfig(arm="parallel", stages=2, visual_tune_blocks=0))
    fresh = CafePCA(copy.deepcopy(base), PCADINOConfig(
        arm="parallel", stages=2, channel_init="fresh", visual_tune_blocks=0,
    ))
    for left, right in zip(pretrained.spatial.parameters(), fresh.spatial.parameters()):
        torch.testing.assert_close(left, right)
    assert any(not torch.equal(left, right) for left, right in
               zip(pretrained.channel.parameters(), fresh.channel.parameters()))


def test_serial_control_is_exactly_the_loaded_published_cafe_function():
    base = official()
    wrapped = CafePCA(copy.deepcopy(base), config("serial")).eval()
    image, text = inputs()
    with torch.no_grad():
        expected = base(image, text, pre_text_emb=True)
        actual = wrapped(image, text)["logits"]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=4e-6)


def test_epl_coefficients_are_pixelwise_expert_probabilities():
    module = ExpertDrivenPerceptualLearning(16, experts=4).eval()
    spatial, semantic = torch.randn(2, 16, 3, 5, 7), torch.randn(2, 16, 3, 5, 7)
    fused, coefficients = module(spatial, semantic, return_coefficients=True)
    assert fused.shape == spatial.shape
    assert coefficients.shape == (2, 3, 4, 5, 7)
    torch.testing.assert_close(coefficients.sum(2), torch.ones(2, 3, 5, 7), rtol=1e-6, atol=1e-6)


def test_fod_is_zero_for_orthogonal_features_and_one_for_collinear_features():
    spatial = torch.zeros(1, 2, 1, 2, 2)
    semantic = torch.zeros_like(spatial)
    spatial[:, 0], semantic[:, 1] = 1, 1
    orthogonal, cosine = feature_orthogonalization_loss(spatial, semantic)
    torch.testing.assert_close(orthogonal, torch.tensor(0.0))
    torch.testing.assert_close(cosine, torch.tensor(0.0))
    collinear, cosine = feature_orthogonalization_loss(spatial, -spatial)
    torch.testing.assert_close(collinear, torch.tensor(1.0))
    torch.testing.assert_close(cosine, torch.tensor(-1.0))


def test_corrected_full_attention_couples_queries_while_published_path_does_not():
    published = official(1).aggregator[0].class_agg.eval()
    corrected = QueryAxisChannelAggregator(published).eval()
    cost = torch.randn(1, 16, 3, 2, 2)
    guidance = torch.randn(4, 3, 16)
    altered = cost.clone()
    altered[:, :, 2] += 50
    with torch.no_grad():
        old_a, old_b = published(cost, guidance), published(altered, guidance)
        new_a, new_b = corrected(cost, guidance), corrected(altered, guidance)
    torch.testing.assert_close(old_a[:, :, :2], old_b[:, :, :2], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(new_a[:, :, :2], new_b[:, :, :2])


def test_pca_is_query_permutation_equivariant():
    model = CafePCA(official(), config("pca_epl_fod")).eval()
    image, text = inputs()
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        native = model(image, text)["logits"]
        permuted = model(image, text[order])["logits"]
    torch.testing.assert_close(permuted, native[:, order], rtol=2e-5, atol=4e-6)


def test_pca_gradients_and_adapted_checkpoint_roundtrip():
    base = official()
    restore_base = copy.deepcopy(base)
    model = CafePCA(base, config("pca_epl_fod")).eval()
    image, text = inputs()
    target = torch.randint(0, len(text), (1, 112, 112))
    output = model(image, text, return_aux=True)
    loss = F.cross_entropy(output["logits"], target) + 0.001 * output["fod_loss"]
    loss.backward()
    for prefix in ("spatial.0.", "channel.0.", "fusion.0."):
        gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert gradients and any(item is not None and torch.isfinite(item).all() and item.abs().sum() > 0
                                 for item in gradients), prefix
    state = model.adapted_state_dict()
    restored = CafePCA(restore_base, config("pca_epl_fod")).eval()
    restored.load_adapted_state_dict(state)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"],
                                   rtol=2e-5, atol=4e-6)
