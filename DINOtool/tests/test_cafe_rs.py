"""RS structure, training, and original-output-path contracts."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(OFFICIAL / "CAFe_DINO")]
from test_cafe_vc import official, inputs
from dinotool.cafe_rs import CafeRS, CafeRSConfig, rs_losses
from dinotool.cafe_rs_regions import make_layout, region_pool, region_write


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(71)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


def model(base=None):
    return CafeRS(base if base is not None else official(), CafeRSConfig(content_dim=32, stages=2, modes=4), official_root=str(OFFICIAL))


@pytest.mark.parametrize("shape", [(7, 7), (7, 14), (14, 7), (15, 9)])
def test_sparse_pool_write_matches_dense_and_preserves_constants(shape):
    layout = make_layout(*shape, torch.device("cpu"))
    weights = torch.rand(2, shape[0] * shape[1], 18).masked_fill(~layout.valid[None], 0)
    weights = weights / weights.sum(-1, keepdim=True)
    dense = weights.new_zeros(2, weights.shape[1], layout.count).scatter_add(2, layout.indices[None].expand(2, -1, -1), weights)
    values = torch.randn(2, weights.shape[1], 5)
    pooled, mass = region_pool(values, weights, layout)
    torch.testing.assert_close(pooled, dense.transpose(1, 2) @ values / dense.sum(1)[..., None].clamp_min(1e-6))
    torch.testing.assert_close(region_write(pooled, weights, layout), dense @ pooled)
    torch.testing.assert_close(region_write(torch.ones_like(pooled), weights, layout), torch.ones_like(values))


@pytest.mark.parametrize("queries", [1, 7, 41])
def test_zero_start_equivalence_with_published_path(queries):
    base = official()
    student = model(copy.deepcopy(base)).eval()
    image, text = inputs(classes=queries)
    with torch.no_grad():
        reference = base(image, text, pre_text_emb=True)
        output = student(image, text, return_aux=True, teacher=True)
    torch.testing.assert_close(output["logits"], reference, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(output["teacher_logits"], reference, rtol=1e-5, atol=3e-6)
    assert len(output["stages"]) == 2
    for stage in output["stages"]:
        torch.testing.assert_close(stage["weights"].sum(-1), torch.ones_like(stage["weights"][..., 0]))
        assert stage["weights"].masked_select(~output["layout"].valid[None]).abs().sum() == 0


def test_nonzero_query_equivariance_rectangles_and_reload():
    student = model().eval()
    for stage in student.reasoning:
        nn.init.normal_(stage.pixel_evidence[-1].weight, std=0.01)
    image, text = inputs(classes=3, height=112, width=224)
    with torch.no_grad():
        output = student(image, text, return_aux=True)
        order = torch.tensor([2, 0, 1])
        permuted = student(image, text[order])["logits"]
    torch.testing.assert_close(permuted, output["logits"][:, order], rtol=2e-5, atol=1e-5)
    assert not torch.allclose(output["stages"][0]["weights"], output["stages"][1]["weights"])
    restored = model(copy.deepcopy(student.cafe)).eval()
    restored.load_adapted_state_dict(student.adapted_state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], output["logits"], rtol=1e-5, atol=3e-6)


def test_every_new_parameter_learns_and_published_model_is_frozen():
    student = model().train()
    image, text = inputs(classes=4)
    target = torch.randint(0, 4, (1, 112, 112))
    # Include pure regions so both affinity directions have supervised examples.
    target[:, :64, :64] = 0
    target[:, 64:, :] = 1
    target[:, :16, :16] = 255
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = student(image, text, return_aux=True, teacher=True)
        loss, components = rs_losses(output, target, 0.1, 0.05, 0.02, 0.02)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    assert all(not m.training for m in student.cafe.modules())


def test_all_ignore_is_finite_differentiable_zero():
    student = model().train()
    image, text = inputs(classes=4)
    output = student(image, text, return_aux=True, teacher=True)
    loss, _ = rs_losses(output, torch.full((1, 112, 112), 255), 0.1, 0.05, 0.02, 0.02)
    assert torch.isfinite(loss) and loss == 0
    loss.backward()


def test_auxiliary_targets_cannot_move_supports():
    from dinotool.cafe_rs_regions import region_auxiliary_losses
    layout = make_layout(7, 7, torch.device("cpu"))
    weights = torch.randn(1, 49, 18, requires_grad=True)
    supported = weights.masked_fill(~layout.valid[None], -torch.inf).softmax(-1)
    logits = torch.randn(1, layout.count, 3, requires_grad=True)
    affinity = torch.randn(1, 49, 2, requires_grad=True)
    stage = dict(weights=supported, region_logits=logits, affinity_logits=affinity)
    loss = region_auxiliary_losses([stage], torch.zeros(1, 112, 112, dtype=torch.long), 3, (7, 7), layout)
    sum(loss.values()).backward()
    assert weights.grad is None
    assert logits.grad is not None and affinity.grad is not None
