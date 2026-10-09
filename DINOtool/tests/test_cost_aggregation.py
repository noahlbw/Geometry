from __future__ import annotations

import torch

from dinotool.config import CheckpointConfig
from dinotool.cost_aggregation import (
    CostAggregationConfig,
    CostAggregationDecoder,
    cost_aggregation_checkpoint_manifest,
    load_cost_aggregation_checkpoint,
    make_cost_aggregation_checkpoint,
)
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import ClassSpec


def _decoder() -> CostAggregationDecoder:
    return CostAggregationDecoder(
        CostAggregationConfig(
            feature_dim=8,
            hidden_dim=16,
            context_blocks=1,
            attention_heads=4,
            window_size=3,
        )
    )


def test_cost_decoder_starts_as_dino_cosine_cost() -> None:
    decoder = _decoder().eval()
    patches = torch.nn.functional.normalize(torch.randn(2, 8, 5, 7), dim=1)
    text = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)

    expected = DINOTextSegmenter.similarity_logits(patches, text)
    actual = decoder(patches, text)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_cost_decoder_accepts_variable_text_vocabularies() -> None:
    decoder = _decoder().eval()
    patches = torch.randn(2, 8, 5, 7)

    assert decoder(patches, torch.randn(3, 8)).shape == (2, 3, 5, 7)
    assert decoder(patches, torch.randn(11, 8)).shape == (2, 11, 5, 7)


def test_cost_decoder_backward_reaches_learned_readout() -> None:
    decoder = _decoder()
    patches = torch.randn(2, 8, 5, 7)
    text = torch.randn(4, 8)
    target = torch.randn(2, 4, 5, 7)

    torch.nn.functional.mse_loss(decoder(patches, text), target).backward()

    assert decoder.reducer[-1].weight.grad is not None
    assert torch.count_nonzero(decoder.reducer[-1].weight.grad) > 0


def test_cost_checkpoint_round_trip_includes_source_metadata(tmp_path) -> None:
    decoder = _decoder()
    checkpoints = CheckpointConfig.from_roots(tmp_path / "code", tmp_path / "weights")
    payload = make_cost_aggregation_checkpoint(
        decoder=decoder,
        source_dataset={"name": "OpenEarthMap", "split": "train"},
        source_classes=(ClassSpec("road", ()),),
        training_config={"external_only": True},
        dino_checkpoints=checkpoints,
        epoch=3,
        validation={"mean_iou": 0.5},
    )
    checkpoint = tmp_path / "cost.pt"
    torch.save(payload, checkpoint)

    restored = load_cost_aggregation_checkpoint(checkpoint)
    manifest = cost_aggregation_checkpoint_manifest(checkpoint, restored)

    assert restored["decoder_config"] == payload["decoder_config"]
    assert manifest["training_epoch"] == 3
    assert manifest["source_classes"] == [{"name": "road", "synonyms": []}]
