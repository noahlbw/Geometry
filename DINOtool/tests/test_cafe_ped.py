"""PED topology, gradients, query ordering, and checkpoint contracts."""
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
from dinotool.cafe_ped import CafePED
from dinotool.parallel_evidence import ParallelEvidenceConfig


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


def config(arm="ped"):
    return ParallelEvidenceConfig(arm=arm, evidence_dim=64, memory_tokens=8, memory_heads=8, read_heads=8,
                                  interaction_dim=16, message_hidden=64, visual_blocks=1, stages=2)


def official():
    return CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


def inputs(classes=4, height=112, width=112):
    return torch.randn(1, 3, height, width), F.normalize(torch.randn(classes, 1024), dim=-1)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(901)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


@pytest.mark.parametrize("arm", ["plain", "naive_parallel", "serial_content", "ped", "anchored_multiscale", "paired_support"])
def test_ped_variants_produce_multichannel_logits_for_rectangular_inputs(arm):
    model = CafePED(official(), config(arm)).eval()
    image, text = inputs(height=112, width=224)
    with torch.no_grad():
        output = model(image, text)["logits"]
    assert output.shape == (1, len(text), 112, 224)
    assert torch.isfinite(output).all()
    if arm == "plain":
        assert len(model.cafe.aggregator) == 2
    else:
        assert len(model.spatial) == len(model.channel) == 2
        assert len(model.cafe.aggregator) == 0


@pytest.mark.parametrize("arm", ["serial_content", "ped"])
def test_content_variants_are_query_permutation_equivariant(arm):
    model = CafePED(official(), config(arm)).eval()
    image, text = inputs()
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        scores = model(image, text)["logits"]
        reordered = model(image, text[order])["logits"]
    torch.testing.assert_close(reordered, scores[:, order], rtol=2e-5, atol=4e-6)


@pytest.mark.parametrize("arm", ["anchored_multiscale", "paired_support"])
def test_targeted_readout_starts_as_the_same_naive_parallel_cost_path(arm):
    base = official()
    naive = CafePED(copy.deepcopy(base), config("naive_parallel")).eval()
    targeted = CafePED(copy.deepcopy(base), config(arm)).eval()
    image, text = inputs()
    with torch.no_grad():
        naive_logits = naive(image, text)["logits"]
        targeted_logits = targeted(image, text)["logits"]
    torch.testing.assert_close(targeted_logits, naive_logits, rtol=2e-5, atol=4e-6)


def test_anchored_and_paired_controls_have_identical_trainable_capacity():
    anchored = CafePED(official(), config("anchored_multiscale"))
    paired = CafePED(official(), config("paired_support"))
    anchored_count = sum(parameter.numel() for parameter in anchored.parameters() if parameter.requires_grad)
    paired_count = sum(parameter.numel() for parameter in paired.parameters() if parameter.requires_grad)
    assert anchored_count == paired_count
    assert anchored.checkpoint_format == paired.checkpoint_format == "cafe_ped_v2"


def test_ped_new_path_and_migrated_paths_receive_gradients_then_checkpoint_roundtrips(tmp_path):
    base = official()
    restore_base = copy.deepcopy(base)
    model = CafePED(base, config("ped")).train()
    image, text = inputs()
    target = torch.randint(0, len(text), (1, 112, 112))
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-3))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image, text)["logits"], target).backward()
        optimizer.step()
    for prefix in ("visual_field.", "memory.", "reader.", "exchange.", "readout.", "spatial.0.", "channel.0."):
        active = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert active and any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in active), prefix
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert frozen and all(parameter.grad is None for parameter in frozen)
    state = model.adapted_state_dict()
    tuned_prefixes = tuple(f"cafe.backbone.visual_model.backbone.blocks.{index}." for index in model.tuned_indices)
    assert not any(key.startswith("cafe.upsampler.") for key in state)
    assert all(not key.startswith("cafe.backbone.") or key.startswith(tuned_prefixes) for key in state)
    # Deployment rebuilds the same official CAFe checkpoint before loading an
    # adapted PED state; model.cafe has intentionally removed its old decoder.
    restored = CafePED(restore_base, config("ped")).eval()
    restored.load_adapted_state_dict(state)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"], rtol=2e-5, atol=4e-6)
    path = tmp_path / "ped.pt"
    torch.save(state, path)
    assert path.is_file()


@pytest.mark.parametrize("arm", ["anchored_multiscale", "paired_support"])
def test_targeted_readout_receives_gradients_after_zero_feedback_has_opened(arm):
    model = CafePED(official(), config(arm)).train()
    image, text = inputs()
    target = torch.randint(0, len(text), (1, 112, 112))
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-3))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image, text)["logits"], target).backward()
        optimizer.step()
    for prefix in ("relation_reader.", "semantic_feedback.", "spatial_feedback.", "relation_feedback."):
        active = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert active and any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in active), prefix
