from __future__ import annotations

import torch

from dinotool.config import CheckpointConfig
from dinotool.prompts import ClassSpec
from dinotool.uot import (
    UOTPrototypeConfig,
    UOTPrototypeDecoder,
    UOTOutput,
    load_uot_checkpoint,
    make_uot_checkpoint,
    uot_checkpoint_manifest,
    unbalanced_sinkhorn_log_plan,
)


def _decoder() -> UOTPrototypeDecoder:
    return UOTPrototypeDecoder(
        UOTPrototypeConfig(
            feature_dim=8,
            num_modes=3,
            epsilon=0.1,
            marginal_relaxation=0.5,
            sinkhorn_iterations=6,
        )
    )


def test_uot_decoder_accepts_variable_vocabularies_and_returns_unknown() -> None:
    decoder = _decoder().eval()
    patches = torch.randn(2, 8, 4, 5)
    for classes in (1, 3, 11):
        output = decoder(patches, torch.randn(classes, 8), return_diagnostics=True)
        assert isinstance(output, UOTOutput)
        assert output.logits.shape == (2, classes, 4, 5)
        assert output.unknown_probability.shape == (2, 4, 5)
        assert output.mode_mass.shape == (2, classes, 3, 4, 5)
        assert torch.isfinite(output.logits).all()
        assert torch.isfinite(output.unknown_probability).all()
        assert ((output.unknown_probability >= 0) & (output.unknown_probability <= 1)).all()


def test_unbalanced_sinkhorn_is_finite_and_differentiable() -> None:
    cost = torch.randn(2, 13, 7, requires_grad=True)
    plan = unbalanced_sinkhorn_log_plan(
        cost,
        epsilon=0.1,
        marginal_relaxation=0.5,
        unknown_prior=0.2,
        iterations=8,
    )
    assert plan.shape == cost.shape
    assert torch.isfinite(plan).all()
    plan.exp().sum().backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()


def test_uot_decoder_backward_reaches_mode_offsets() -> None:
    decoder = _decoder()
    patches = torch.randn(2, 8, 4, 5)
    text = torch.randn(4, 8)
    output = decoder(patches, text)
    output.square().mean().backward()
    assert decoder.mode_offsets.grad is not None
    assert torch.isfinite(decoder.mode_offsets.grad).all()


def test_uot_checkpoint_round_trip_includes_metadata(tmp_path) -> None:
    decoder = _decoder()
    checkpoints = CheckpointConfig.from_roots(tmp_path / "code", tmp_path / "weights")
    payload = make_uot_checkpoint(
        decoder=decoder,
        source_dataset={"name": "OpenEarthMap", "split": "train"},
        source_classes=(ClassSpec("road", ()),),
        training_config={"external_only": True},
        dino_checkpoints=checkpoints,
        epoch=2,
        validation={"mean_iou": 0.25},
    )
    checkpoint = tmp_path / "uot.pt"
    torch.save(payload, checkpoint)
    restored = load_uot_checkpoint(checkpoint)
    manifest = uot_checkpoint_manifest(checkpoint, restored)
    assert restored["decoder_config"] == payload["decoder_config"]
    assert manifest["training_epoch"] == 2
    assert manifest["source_dataset"]["name"] == "OpenEarthMap"
