"""CPU tests with real official CAFe aggregation and a small frozen encoder."""
import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party" / "DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(OFFICIAL / "CAFe_DINO"))
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_rc import CafeRC, CafeRCConfig, RegionalMemory, ContentReconstruction
from train_cafe_rc import paired_loss, sliding_logits, args_parser, validate_args


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(123)
    torch.set_num_threads(2)
    # Unit gradients use the portable attention reference; the separate real
    # eight-GPU smoke test exercises the optimized CUDA/checkpointed path.
    # PyTorch 2.11's CPU MKLDNN backward stalls on this frozen-head graph on
    # the experiment host. CUDA smoke tests retain the production backends.
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)

    def encode_image_with_patch_tokens(self, images, normalize=False):
        x = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        y = torch.tanh(x)
        return y.mean(1), y, x


class SmallUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official():
    return CAFe_DINO(SmallBackbone(), None, SmallUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


def inputs(classes=3, height=112, width=112):
    return torch.randn(1, 3, height, width), F.normalize(torch.randn(classes, 1024), dim=-1)


def test_baseline_matches_unmodified_official_forward():
    base = official()
    model = CafeRC(copy.deepcopy(base), CafeRCConfig(variant="baseline"), teacher=False).eval()
    image, text = inputs()
    with torch.no_grad():
        reference = base(image, text, pre_text_emb=True)
        actual = model(image, text)["logits"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("variant", ["regional", "content", "baseline"])
def test_query_order_equivariance_and_single_query(variant):
    model = CafeRC(official(), CafeRCConfig(content_dim=32, variant=variant), teacher=False).eval()
    image, text = inputs()
    with torch.no_grad():
        scores = model(image, text)["logits"]
        order = torch.tensor([2, 0, 1])
        permuted = model(image, text[order])["logits"]
        single = model(image, text[:1])["logits"]
    torch.testing.assert_close(permuted, scores[:, order], atol=3e-6, rtol=1e-5)
    assert single.shape == (1, 1, 112, 112)


def test_gradients_reach_new_decoder_but_not_frozen_components():
    model = CafeRC(official(), CafeRCConfig(content_dim=32)).train()
    image, text = inputs(5)
    output = model(image, text, output_count=3, preserve_from=3)
    loss = F.cross_entropy(output["logits"], torch.randint(0, 3, (1, 112, 112))) + output["preservation_loss"]
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert model.reconstruction[0].memory_v.weight.grad.abs().sum() > 0
    assert not model.cafe.backbone.training
    assert not model.cafe.upsampler.training
    assert not model.cafe.reduce_d.training
    assert not model.teacher.training


def test_visual_values_change_evidence_even_with_fixed_costs():
    block = ContentReconstruction(16, 1024, CafeRCConfig(content_dim=32)).eval()
    cost, text = torch.zeros(1, 16, 2, 7, 7), F.normalize(torch.randn(2, 1024), dim=-1)
    content = torch.randn(1, 32, 7, 7)
    with torch.no_grad():
        _, first = block(content, cost, text)
        _, second = block(content.flip(-1), cost, text)
    assert not torch.allclose(first, second, atol=1e-7)


def test_region_support_covers_borders_and_rectangular_grid():
    module = RegionalMemory(CafeRCConfig(content_dim=32))
    content = torch.randn(2, 32, 9, 17, requires_grad=True)
    updated, memory, support = module(content)
    assert support.any(0).all() and support.any(1).all()
    assert updated.shape == content.shape
    assert memory.shape == (2, 15, 32)
    (updated.square().mean() + memory.square().mean()).backward()
    assert content.grad is not None


def test_rectangular_forward_and_adapted_checkpoint_roundtrip():
    model = CafeRC(official(), CafeRCConfig(content_dim=32)).eval()
    image, text = inputs(height=224, width=112)
    with torch.no_grad():
        before = model(image, text)["logits"]
    state = model.adapted_state_dict()
    assert not any(key.startswith(("teacher.", "cafe.backbone.", "cafe.upsampler.")) for key in state)
    restored = CafeRC(copy.deepcopy(model.cafe), model.config, teacher=False).eval()
    restored.load_adapted_state_dict(state)
    with torch.no_grad():
        after = restored(image, text)["logits"]
    torch.testing.assert_close(after, before)
    with pytest.raises(ValueError):
        restored.load_adapted_state_dict({})


def test_all_ignore_loss_is_finite_differentiable_zero():
    scores = torch.randn(2, 8, 8, 8, requires_grad=True)
    loss = paired_loss(scores, torch.full((2, 8, 8), 255), torch.ones(8), 0.3)
    loss.backward()
    assert loss.item() == 0 and scores.grad is not None


def test_sliding_inference_covers_native_rectangle_without_resize():
    class ConstantModel:
        def __call__(self, image, text):
            return {"logits": torch.ones((1, len(text), *image.shape[-2:]))}
    image = torch.randn(1, 3, 143, 207)
    result = sliding_logits(ConstantModel(), image, torch.ones(2, 1024), 112, 80, "fp32")
    assert result.shape == (2, 143, 207)
    torch.testing.assert_close(result, torch.ones_like(result))


def test_target_training_path_rejected():
    args = args_parser().parse_args(["--official-root", ".", "--checkpoint", "base.pt", "--bpe-path", "bpe.gz", "--data-root", "/data/LoveDA/val", "--output-dir", "unused"])
    with pytest.raises(ValueError, match="LoveDA"):
        validate_args(args)


def test_cuda_shape_change_moves_official_attention_mask_to_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CafeRC(official(), CafeRCConfig(content_dim=32), teacher=False).cuda().eval()
    image, text = inputs(height=224, width=112)
    with torch.inference_mode():
        output = model(image.cuda(), text.cuda())["logits"]
    assert output.shape == (1, 3, 224, 112)
    assert torch.isfinite(output).all()
