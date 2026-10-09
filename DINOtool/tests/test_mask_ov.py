from pathlib import Path

import torch

from dinotool.config import CheckpointConfig
from dinotool.mask_ov import (
    MaskOVConfig,
    MultiScaleMaskQueryDecoder,
    load_mask_ov_checkpoint,
    make_mask_ov_checkpoint,
)
from dinotool.ov_adapter import DenseAdapterConfig, DenseTextAdapter
from dinotool.oem import OEM_CLASSES


def _decoder() -> MultiScaleMaskQueryDecoder:
    return MultiScaleMaskQueryDecoder(
        MaskOVConfig(
            feature_dim=8,
            adapter_hidden_dim=32,
            adapter_context_blocks=1,
            fusion_dim=32,
            num_queries=5,
            attention_heads=4,
            query_layers=2,
            feature_layers=(1, 2),
            dropout=0.0,
        )
    )


def _inputs(classes: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    features = tuple(torch.randn(2, 8, 4, 4, requires_grad=True) for _ in range(2))
    satellite = torch.randn(2, 8, 4, 4, requires_grad=True)
    rgb = torch.rand(2, 3, 4, 4)
    text = torch.randn(classes, 8)
    return features, satellite, rgb, text


def test_mask_decoder_accepts_variable_text_vocabularies() -> None:
    decoder = _decoder()
    for classes in (3, 7):
        features, satellite, rgb, text = _inputs(classes)
        output = decoder(features, satellite, rgb, text)
        assert output.residual_logits.shape == (2, classes, 4, 4)
        assert output.mask_logits.shape == (2, 5, 4, 4)
        assert output.class_scores.shape == (2, 5, classes)
        assert output.semantic_mask.shape == (2, classes, 4, 4)


def test_mask_decoder_backpropagates_through_masks_and_queries() -> None:
    decoder = _decoder()
    features, satellite, rgb, text = _inputs(4)
    output = decoder(features, satellite, rgb, text)
    loss = output.residual_logits.square().mean() + output.mask_logits.square().mean() + output.class_scores.square().mean()
    loss.backward()

    assert decoder.query_tokens.grad is not None
    assert decoder.feature_projections[0].weight.grad is not None
    assert features[0].grad is not None
    assert satellite.grad is not None


def test_mask_checkpoint_round_trip(tmp_path: Path) -> None:
    adapter_config = DenseAdapterConfig(feature_dim=8, hidden_dim=32, context_blocks=1, use_satellite_features=True)
    adapter = DenseTextAdapter(adapter_config)
    decoder = _decoder()
    checkpoint = make_mask_ov_checkpoint(
        adapter=adapter,
        decoder=decoder,
        source_dataset={"name": "external-test"},
        source_classes=OEM_CLASSES,
        training_config={"unit_test": True},
        dino_checkpoints=CheckpointConfig.from_roots(tmp_path),
        epoch=3,
        validation={"mean_iou": 0.12},
    )
    path = tmp_path / "mask.pt"
    torch.save(checkpoint, path)
    loaded = load_mask_ov_checkpoint(path)

    assert loaded["format_version"] == 1
    assert loaded["mask_config"]["num_queries"] == 5
    assert loaded["adapter_config"] == adapter_config.__dict__
    assert loaded["decoder_state_dict"]["query_tokens"].shape == (5, 32)
