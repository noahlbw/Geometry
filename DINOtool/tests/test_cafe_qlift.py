"""CAFe adapter, trainability, and checkpoint contracts for Q-Lift."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]

from test_cafe_ped import inputs, official
from dinotool.cafe_qlift import CafeQLift, config_from_checkpoint
from dinotool.qlift import QLiftConfig
from eval_cafe_vc_protocols import checkpoint_config


def config(arm: str, *, tune_visual_blocks: int = 0) -> QLiftConfig:
    return QLiftConfig(arm=arm, levels=2, guide_dim=32, hidden_dim=64, kernel=3,
                       query_chunk=2, tune_visual_blocks=tune_visual_blocks)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(883)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False):
        yield


@pytest.mark.parametrize("arm", ("skip", "haar", "image_lift", "query_lift"))
def test_all_arms_cover_rectangular_images_and_keep_base_aggregator_unused(arm):
    model = CafeQLift(official(), config(arm)).eval()
    image, text = inputs(classes=3, height=112, width=224)
    with torch.no_grad():
        output = model(image, text, return_aux=True)
    assert output["logits"].shape == (1, 3, 112, 224)
    assert torch.isfinite(output["logits"]).all()
    assert len(model.cafe.aggregator) == 2
    assert not any(parameter.requires_grad for parameter in model.cafe.aggregator.parameters())
    assert output["qlift"]["max_cost_change"] < 2e-5


def test_warmup_freezes_cafe_readout_then_active_decoder_and_checkpoint_roundtrip(tmp_path):
    base = official()
    restored_base = copy.deepcopy(base)
    model = CafeQLift(base, config("query_lift")).train()
    image, text = inputs(classes=3)
    target = torch.randint(0, 3, (1, 112, 112))
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-3))
    corr_before = model.cafe.corr_embed.weight.detach().clone()
    model.set_warmup(True)
    assert not model.cafe.corr_embed.weight.requires_grad
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(image, text)["logits"], target).backward()
    optimizer.step()
    torch.testing.assert_close(model.cafe.corr_embed.weight, corr_before, rtol=0, atol=0)
    model.set_warmup(False)
    assert model.cafe.corr_embed.weight.requires_grad
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image, text)["logits"], target).backward()
        optimizer.step()
    assert not torch.equal(model.cafe.corr_embed.weight, corr_before)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    payload = {
        "format": model.checkpoint_format,
        "architecture": model.architecture(),
        "adapted_state": model.adapted_state_dict(),
    }
    path = tmp_path / "qlift.pt"
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=True)
    assert checkpoint_config(loaded) == config("query_lift")
    restored = CafeQLift(restored_base, config_from_checkpoint(loaded)).eval()
    restored.load_adapted_state_dict(loaded["adapted_state"])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"], rtol=2e-5, atol=2e-6)


def test_optional_visual_finetuning_is_explicit_and_warmup_closes_its_graph():
    model = CafeQLift(official(), config("query_lift", tune_visual_blocks=2)).train()
    assert model.tuned_indices == (2, 3)
    assert model.joint_finetuning
    model.set_warmup(True)
    assert not model.joint_finetuning
    assert all(not parameter.requires_grad for index in model.tuned_indices
               for parameter in model.cafe.backbone.visual_model.backbone.blocks[index].parameters())
    model.set_warmup(False)
    assert model.joint_finetuning
    assert all(parameter.requires_grad for index in model.tuned_indices
               for parameter in model.cafe.backbone.visual_model.backbone.blocks[index].parameters())
