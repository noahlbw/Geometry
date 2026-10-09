"""CAFe-VC contract tests with the real CAFe aggregation and tiny frozen encoder."""

import copy
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party" / "DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL / "CAFe_DINO"))
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_vc import CafeVC, CafeVCConfig, VisualCostReconstruction


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(321)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)

    def encode_image_with_patch_tokens(self, images, normalize=False):
        x = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        return x.mean(1), torch.tanh(x), x


class SmallUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official():
    return CAFe_DINO(SmallBackbone(), None, SmallUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


def inputs(classes=3, height=112, width=112):
    return torch.randn(1, 3, height, width), F.normalize(torch.randn(classes, 1024), dim=-1)


@pytest.mark.parametrize("arm", ["plain", "concat", "vc", "cost_only"])
def test_all_arms_begin_as_the_published_cafe_function(arm):
    base = official()
    blocks = 0 if arm == "plain" else 2
    model = CafeVC(copy.deepcopy(base), CafeVCConfig(arm=arm, content_dim=32, blocks=blocks)).eval()
    image, text = inputs()
    with torch.no_grad():
        reference = base(image, text, pre_text_emb=True)
        actual = model(image, text)["logits"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)


def test_vc_is_query_order_equivariant_and_keeps_original_output_path():
    model = CafeVC(official(), CafeVCConfig(arm="vc", content_dim=32, blocks=2)).eval()
    for block in model.reconstruction:
        nn.init.normal_(block.evidence[-1].weight, std=0.01)
    image, text = inputs()
    with torch.no_grad():
        scores = model(image, text)["logits"]
        order = torch.tensor([2, 0, 1])
        permuted = model(image, text[order])["logits"]
    torch.testing.assert_close(permuted, scores[:, order], rtol=1e-5, atol=3e-6)


def test_vc_learns_only_through_trainable_cost_decoder_after_zero_start():
    model = CafeVC(official(), CafeVCConfig(arm="vc", content_dim=32, blocks=2)).train()
    image, text = inputs(classes=4)
    target = torch.randint(0, 4, (1, 112, 112))
    output = model(image, text)["logits"]
    F.cross_entropy(output, target).backward()
    assert model.reconstruction[0].evidence[-1].weight.grad is not None
    assert model.reconstruction[0].evidence[-1].weight.grad.abs().sum() > 0
    assert model.cafe.backbone.conv.weight.grad is None
    assert model.cafe.upsampler.training is False
    assert model.cafe.reduce_d.training is False


def test_adapted_checkpoint_roundtrip_and_rectangular_input():
    config = CafeVCConfig(arm="vc", content_dim=32, blocks=2)
    model = CafeVC(official(), config).eval()
    for block in model.reconstruction:
        nn.init.normal_(block.evidence[-1].weight, std=0.01)
    image, text = inputs(height=224, width=112)
    with torch.no_grad():
        before = model(image, text)["logits"]
    state = model.adapted_state_dict()
    assert not any(name.startswith(("cafe.backbone.", "cafe.upsampler.", "cafe.reduce_d.")) for name in state)
    restored = CafeVC(copy.deepcopy(model.cafe), config).eval()
    restored.load_adapted_state_dict(state)
    with torch.no_grad():
        after = restored(image, text)["logits"]
    torch.testing.assert_close(after, before, rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize("kernel", [3, 7])
def test_neighbors_are_complete_pixel_vectors_including_padding(kernel):
    block = VisualCostReconstruction(16, 1024, CafeVCConfig(content_dim=8, kernel_size=kernel), spatial_read=True)
    content = torch.arange(2 * 8 * 5 * 6, dtype=torch.float32).reshape(2, 8, 5, 6)
    local, neighbors = block._local_values(content)
    padded = F.pad(content, (kernel // 2,) * 4)
    for row in range(5):
        for col in range(6):
            expected = padded[:, :, row:row + kernel, col:col + kernel].flatten(2).transpose(1, 2)
            torch.testing.assert_close(neighbors[:, row * 6 + col], expected, rtol=0, atol=0)
    torch.testing.assert_close(neighbors[:, :, kernel**2 // 2], local, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["concat", "vc", "cost_only"])
def test_every_new_parameter_gets_gradient_after_zero_start(arm):
    model = CafeVC(official(), CafeVCConfig(arm=arm, content_dim=32)).train()
    image, text = inputs(classes=4)
    target = torch.randint(0, 4, (1, 112, 112))
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image, text)["logits"], target).backward()
        optimizer.step()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not name.startswith("cafe."):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_active_capacity_of_default_reconstruction_controls_is_matched():
    counts = {}
    for arm in ("vc", "concat", "cost_only"):
        cafe = CAFe_DINO(SmallBackbone(), None, SmallUpsampler(), (7, 7), "cpu", aggregator_dim=64, aggregator_blocks=2)
        model = CafeVC(cafe, CafeVCConfig(arm=arm))
        counts[arm] = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cafe."))
    assert max(counts.values()) / min(counts.values()) < 1.05, counts


def test_resolution_change_preserves_zero_start_reference():
    base = official()
    model = CafeVC(copy.deepcopy(base), CafeVCConfig(arm="vc", content_dim=32)).eval()
    for height, width in ((112, 224), (224, 224), (112, 112)):
        image, text = inputs(height=height, width=width)
        for layer in base.aggregator:
            for block in (layer.spatial_agg.swin1, layer.spatial_agg.swin2):
                block.set_input_size((height // 16, width // 16), (7, 7))
        with torch.no_grad():
            actual = model(image, text)["logits"]
            reference = base(image, text, pre_text_emb=True)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)
