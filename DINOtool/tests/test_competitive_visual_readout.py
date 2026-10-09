from __future__ import annotations

import torch

from dinotool.competitive_visual_readout import (
    CVERConfig,
    CVERSegmenter,
    all_class_hypothesis_scores,
    class_contrast,
    competitive_pair_correction,
    self_attended,
    visual_constrained_relations,
)
from dinotool.tcpr import TCPRTextBank

from test_tcpr import _TinyBackbone


def _bank() -> TCPRTextBank:
    return TCPRTextBank(
        features=torch.nn.functional.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
        ]), dim=-1),
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("first", "second"),
        alias_names=("first", "first-like", "second", "second-like"),
    )


def test_class_contrast_uses_strongest_other_class() -> None:
    scores = torch.tensor([[[0.8, 0.5, 0.1]]])
    output = class_contrast(scores)
    torch.testing.assert_close(output, torch.tensor([[[0.3, -0.3, -0.7]]]))


def test_all_class_response_keeps_off_diagonal_evidence() -> None:
    aliases = torch.tensor([[[[0.8, 0.7, 0.2, 0.1]], [[0.6, 0.5, 0.9, 0.8]]]])
    output = all_class_hypothesis_scores(
        aliases,
        torch.tensor([0, 0, 1, 1]),
        2,
        torch.ones(4, dtype=torch.bool),
        0.07,
    )
    assert output.shape == (1, 2, 1, 2)
    assert output[0, 0, 0, 0] > output[0, 0, 0, 1]
    assert output[0, 1, 0, 1] > output[0, 1, 0, 0]


def test_pair_correction_is_zero_sum_and_rewards_exclusive_response() -> None:
    base = torch.tensor([[[0.6, 0.5]]])
    responses = torch.tensor([[[[0.9, 0.4]], [[0.55, 0.60]]]])
    quality = torch.ones((1, 2, 1))
    corrected, pairs, evidence, _ = competitive_pair_correction(
        base, responses, quality, torch.ones((1, 1), dtype=torch.bool), CVERConfig()
    )
    assert pairs.tolist() == [[[0, 1]]]
    assert evidence.item() > 0
    torch.testing.assert_close(corrected.sum(-1), base.sum(-1))
    assert corrected[0, 0, 0] > base[0, 0, 0]
    assert corrected[0, 0, 1] < base[0, 0, 1]


def test_visual_relation_uniform_evidence_stays_finite() -> None:
    geometry = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]])
    native = geometry.clone()
    raw = torch.eye(2)[None]
    contrast = torch.zeros((1, 2, 2))
    valid = torch.ones((1, 2), dtype=torch.bool)
    routes, quality, kl, raw_fit, semantic_fit = visual_constrained_relations(
        geometry, native, raw, contrast, valid, CVERConfig()
    )
    assert routes.shape == (1, 2, 2, 2)
    torch.testing.assert_close(routes.sum(-1), torch.ones((1, 2, 2)))
    assert torch.isfinite(routes).all()
    assert torch.isfinite(quality).all()
    assert torch.isfinite(kl).all()
    assert torch.isfinite(raw_fit).all()
    assert torch.isfinite(semantic_fit).all()


def test_self_attention_uses_matching_patch_values() -> None:
    attention = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.25, 0.75], [0.0, 0.60, 0.40]]]])
    values = torch.tensor([[[[10.0], [2.0], [5.0]]]])
    output = self_attended(attention, values, prefix=1)
    torch.testing.assert_close(output[0, 0, 1:, 0], torch.tensor([2.0, 5.0]))


def test_complete_cver_tensor_path_returns_competitive_readout() -> None:
    torch.manual_seed(11)
    model = CVERSegmenter(_TinyBackbone(), CVERConfig())
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    result = model.read_prepared(prepared, _bank())
    assert result.geometry_logits.shape == result.logits.shape == (1, 2, 2, 2)
    assert result.response_matrix.shape == (1, 2, 4, 2)
    assert result.route_quality.shape == (1, 2, 2, 2)
    assert result.pair_ids.shape == (1, 2, 2, 2)
    assert result.pair_evidence.shape == (1, 2, 2)
    assert torch.isfinite(result.logits).all()
