"""Matched-budget controls, explicit membership, gradients and audit accounting."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_cafe_rc import official, inputs, seed
from dinotool.cafe_rc import CafeRC, CafeRCConfig
from dinotool.cafe_membership import FlatMembershipDecoder
from dinotool.segmentation_errors import error_counts, summarize_errors
from eval_cafe_rc_loveda import model_label


def test_budget_is_identical_across_mask_variants():
    models = [FlatMembershipDecoder(dim=32, variant=name) for name in ("mask_plain", "mask_dual", "mask_fixed")]
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1
    assert list(models[0].state_dict()) == list(models[1].state_dict()) == list(models[2].state_dict())


@pytest.mark.parametrize("variant", ("mask_plain", "mask_dual", "mask_fixed"))
def test_membership_bounds_coverage_and_gradients(variant):
    model = FlatMembershipDecoder(dim=32, variant=variant)
    x, y = torch.randn(1, 1024, 7, 11), torch.randn(1, 1024, 7, 11)
    base = torch.randn(1, 3, 7, 11, requires_grad=True)
    text = F.normalize(torch.randn(3, 1024), dim=-1)
    result = model(x, y, base, text, return_states=True)
    torch.testing.assert_close(result["support"].sum(0), torch.ones(77))
    assert result["coarse_logits"].shape == base.shape
    assert ((result["region_probabilities"] >= 0) & (result["region_probabilities"] <= 1)).all()
    F.cross_entropy(result["coarse_logits"], torch.randint(0, 3, (1, 7, 11))).backward()
    assert base.grad is not None
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_plain_and_dual_only_differ_by_evidence_access():
    dual = FlatMembershipDecoder(dim=32, variant="mask_dual").eval()
    plain = copy.deepcopy(dual)
    plain.variant = "mask_plain"
    x, y = torch.randn(1, 1024, 7, 7), torch.randn(1, 1024, 7, 7)
    text = F.normalize(torch.randn(2, 1024), dim=-1)
    base = torch.randn(1, 2, 7, 7) * 4
    with torch.no_grad():
        actual = dual(x, y, base, text)["coarse_logits"]
        control = plain(x, y, base, text)["coarse_logits"]
        order = torch.tensor([1, 0])
        permuted = dual(x, y, base[:, order], text[order])["coarse_logits"]
    assert not torch.allclose(actual, control, atol=1e-7, rtol=0)
    torch.testing.assert_close(permuted, actual[:, order], atol=5e-5, rtol=5e-5)


def test_membership_ignores_common_pixelwise_class_logit_offset():
    model = FlatMembershipDecoder(dim=32).eval()
    x, y = torch.randn(1, 1024, 7, 7), torch.randn(1, 1024, 7, 7)
    text = F.normalize(torch.randn(3, 1024), dim=-1)
    base = torch.randn(1, 3, 7, 7)
    offset = torch.randn(1, 1, 7, 7) * 5
    with torch.no_grad():
        original = model(x, y, base, text)["coarse_logits"]
        shifted = model(x, y, base + offset, text)["coarse_logits"]
    torch.testing.assert_close(original, shifted, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("variant", ("mask_plain", "mask_dual", "mask_fixed"))
def test_full_wrapper_backward_and_reload(variant):
    model = CafeRC(official(), CafeRCConfig(content_dim=32, variant=variant)).train()
    image, text = inputs(classes=5)
    result = model(image, text, output_count=3, preserve_from=3)
    loss = F.cross_entropy(result["logits"], torch.randint(0, 3, (1, 112, 112))) + result["preservation_loss"]
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    model.eval()
    restored = CafeRC(copy.deepcopy(model.cafe), model.config, teacher=False).eval()
    restored.load_adapted_state_dict(model.adapted_state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"])


def test_error_distance_absent_class_and_ignore():
    target = np.zeros((9, 9), dtype=np.int64)
    target[4, 4] = 1
    target[0, 0] = 255
    prediction = target.copy()
    prediction[4, 5] = 1
    prediction[8, 8] = 1
    prediction[0, 1] = 2
    record = error_counts(prediction, target, 3, (1, 2, 4))
    assert record["valid_pixels"] == 80
    assert record["classes"][1]["fp"] == 2
    assert record["classes"][1]["fp_within_gt_distance"] == [1, 1, 1]
    assert record["classes"][2]["fp_without_class_in_image"] == 1
    summary = summarize_errors([record], ["a", "b", "c"], (1, 2, 4))
    assert summary["per_class"][1]["fp_beyond_gt_distance"] == [1, 1, 1]
    assert np.asarray(summary["confusion_matrix"]).sum() == 80


def test_trained_plain_is_not_labeled_as_rc():
    payload = {"rc_config": {"variant": "baseline"}}
    assert "source-trained plain" in model_label(payload, False)
    assert "frozen" in model_label(payload, True)
