"""Actual membership recurrence, source supervision, and CAFe compatibility."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]
from test_cafe_ped import official, inputs
from dinotool.cafe_ped import CafePED
from dinotool.parallel_evidence import ParallelEvidenceConfig
from dinotool.cafe_region_assembly import CafeRegionAssembly, assembly_loss, config_from_checkpoint
from dinotool.region_assembly import RegionAssemblyConfig, compose_membership, region_pool, membership_loss
from eval_cafe_vc_protocols import checkpoint_config


def config(arm="query_assembly"):
    return RegionAssemblyConfig(arm=arm, dim=32, heads=4, regions=6, parents=3,
                                proposal_layers=3, stages=2, query_chunk=2)


@pytest.fixture(autouse=True)
def cpu_reference():
    torch.manual_seed(319)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


@pytest.mark.parametrize("arm", ["flat_regions", "image_assembly", "query_assembly"])
def test_zero_start_preserves_serial_cafe_and_rectangular_output(arm):
    base = official()
    reference = CafePED(copy.deepcopy(base), ParallelEvidenceConfig(arm="plain", stages=2)).eval()
    model = CafeRegionAssembly(base, config(arm)).eval()
    image, text = inputs(classes=3, height=112, width=224)
    with torch.no_grad():
        expected = reference(image, text)["logits"]
        output = model(image, text, return_aux=True)
    torch.testing.assert_close(output["logits"], expected, rtol=0, atol=0)
    assert len(output["membership_logits"]) == 2
    assert output["membership_logits"][0].shape[-3:] == (3, 7, 14)
    assert torch.isfinite(output["membership_logits"][0]).all()


def test_parent_composition_conserves_mass_and_can_merge_children():
    children = torch.eye(3)[None]
    assignment = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    parent = compose_membership(children, assignment)
    torch.testing.assert_close(parent, assignment)
    torch.testing.assert_close(parent.sum(-1), torch.ones(1, 3))
    pooled = region_pool(torch.tensor([[[2.], [4.], [9.]]]), parent)
    torch.testing.assert_close(pooled, torch.tensor([[[3.], [9.]]]))
    assert torch.equal(region_pool(torch.randn(1, 3, 2), torch.zeros(1, 3, 2)), torch.zeros(1, 2, 2))


def test_text_changes_assembly_and_updates_are_read_on_the_next_round():
    model = CafeRegionAssembly(official(), config()).eval()
    image, text = inputs(classes=2)
    with torch.no_grad():
        output = model(image, text, return_regions=True)
    first, second = output["region_traces"][0]
    assert not torch.allclose(first["assignment"][0], first["assignment"][1], atol=1e-6)
    assert not torch.allclose(first["child_membership"], second["child_membership"], atol=1e-6)
    for record in (first, second):
        for key in ("child_membership", "parent_membership", "assignment"):
            values = record[key]
            torch.testing.assert_close(values.sum(-1), torch.ones_like(values[..., 0]))
    assert not torch.allclose(output["final_membership"][:, 0], output["final_membership"][:, 1], atol=1e-6)


def test_image_control_geometry_is_independent_of_text_at_every_stage():
    model = CafeRegionAssembly(official(), config("image_assembly")).eval()
    image, text = inputs(classes=2)
    with torch.no_grad():
        output = model(image, text, return_regions=True)
        other = model(image, torch.randn_like(text), return_regions=True)
    torch.testing.assert_close(output["final_membership"], other["final_membership"], rtol=0, atol=0)
    torch.testing.assert_close(output["final_membership"][:, 0], output["final_membership"][:, 1], rtol=1e-6, atol=1e-7)


def test_query_permutation_and_chunking_with_active_cost_writeback():
    model = CafeRegionAssembly(official(), config()).eval()
    for stage in model.assembly:
        torch.nn.init.normal_(stage.write[-1].weight, std=0.02)
    chunked = copy.deepcopy(model)
    for stage in chunked.assembly:
        stage.config = replace(stage.config, query_chunk=1)
    image, text = inputs(classes=3)
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        original = model(image, text)
        permuted = model(image, text[order])
        split = chunked(image, text)
        one = model(image, text[:1], return_aux=True)
        subset = model(image, text, output_count=2)
    torch.testing.assert_close(original["logits"][:, order], permuted["logits"], atol=5e-6, rtol=3e-5)
    torch.testing.assert_close(original["logits"], split["logits"], atol=5e-6, rtol=3e-5)
    torch.testing.assert_close(original["logits"][:, :2], subset["logits"], atol=5e-6, rtol=3e-5)
    assert one["logits"].shape == (1, 1, 112, 112)
    with pytest.raises(ValueError, match="entire query"):
        model(image, text, output_count=2, return_aux=True)


@pytest.mark.parametrize("arm", ["flat_regions", "image_assembly", "query_assembly"])
def test_training_gradients_checkpoint_and_freezing(arm, tmp_path):
    base = official()
    model = CafeRegionAssembly(copy.deepcopy(base), config(arm)).train()
    image, text = inputs(classes=3)
    target = torch.zeros(1, 112, 112, dtype=torch.long)
    target[:, 30:75] = 1
    target[:, 75:] = 2
    target[:, :8, :16] = 255
    teacher = torch.randn(1, 3, 112, 112, requires_grad=True)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-4, 1e-5))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(image, text, return_aux=True)
        loss, terms = assembly_loss(output, target, teacher)
        assert torch.isfinite(loss) and terms["region_loss"] > 0
        loss.backward()
        optimizer.step()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
    for prefix in ("proposals.", "assembly.0.read.", "assembly.0.write."):
        assert sum(float(p.grad.norm()) for n, p in model.named_parameters() if n.startswith(prefix)) > 0
    assert teacher.grad is None
    assert not model.cafe.upsampler.training
    assert not model.cafe.reduce_d.training
    payload = dict(format=model.checkpoint_format, architecture=model.architecture(), adapted_state=model.adapted_state_dict())
    path = tmp_path / "assembly.pt"
    torch.save(payload, path)
    payload = torch.load(path, weights_only=True)
    assert checkpoint_config(payload) == config(arm)
    restored = CafeRegionAssembly(base, config_from_checkpoint(payload)).eval()
    restored.load_adapted_state_dict(payload["adapted_state"])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"], rtol=0, atol=0)
    with pytest.raises(ValueError):
        config_from_checkpoint({**payload, "format": "cafe_ped_v2"})
    broken = payload["adapted_state"].copy()
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="keys differ"):
        restored.load_adapted_state_dict(broken)


def test_ignore_and_thin_structure_soft_occupancy():
    prediction = torch.tensor([[[[[0.]], [[2.]]]]], requires_grad=True)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[:, :, 0] = 1
    expected = F.binary_cross_entropy_with_logits(prediction.flatten(), torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(membership_loss([prediction], target, 2), expected)
    target.fill_(255)
    loss = membership_loss([prediction], target, 2)
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_batch_and_query_axes_remain_separate():
    model = CafeRegionAssembly(official(), config()).eval()
    for stage in model.assembly:
        torch.nn.init.normal_(stage.write[-1].weight, std=0.02)
    image, text = inputs(classes=3)
    images = torch.cat((image, image.flip(-1) * 0.7), 0)
    with torch.no_grad():
        batched = model(images, text, return_regions=True)
        separate = [model(item[None], text, return_regions=True) for item in images]
    torch.testing.assert_close(batched["logits"], torch.cat([item["logits"] for item in separate]), atol=5e-6, rtol=3e-5)
    torch.testing.assert_close(batched["final_membership"], torch.cat([item["final_membership"] for item in separate]),
                               atol=5e-6, rtol=3e-5)


def test_all_ignore_total_loss_is_zero_and_warmup_keeps_base_in_eval():
    model = CafeRegionAssembly(official(), config()).train()
    model.set_warmup(True)
    assert model.proposals.training and model.assembly.training
    assert not model.cafe.aggregator.training
    model.set_warmup(False)
    assert model.cafe.aggregator.training
    image, text = inputs(classes=2)
    output = model(image, text, return_aux=True)
    loss, _ = assembly_loss(output, torch.full((1, 112, 112), 255), torch.randn_like(output["logits"]))
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_entry_uses_same_source_parser_and_new_model_config(monkeypatch):
    from train_cafe_region_assembly import extend_parser, model_config
    from train_cafe_ped import parse_args
    argv = ["train"]
    for key in ("official-root", "checkpoint", "bpe-path", "coco-root", "oem-root", "output-dir"):
        argv.extend(["--" + key, "unused"])
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args(extend_parser=extend_parser)
    assert model_config(args) == RegionAssemblyConfig()
    assert (args.updates, args.validate_every, args.batch_size, args.accum_steps) == (20896, 2612, 2, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_forward_backward_is_finite():
    model = CafeRegionAssembly(official(), config()).cuda().train()
    image, text = (value.cuda() for value in inputs(classes=3))
    target = torch.randint(0, 3, (1, 112, 112), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image, text, return_aux=True)
        loss, _ = assembly_loss(output, target, torch.randn_like(output["logits"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
