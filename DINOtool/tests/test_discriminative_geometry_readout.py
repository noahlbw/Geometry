from __future__ import annotations

import torch

from dinotool.discriminative_geometry_readout import (
    DGERConfig,
    DGERSegmenter,
    consensus_geometry_correction,
    discriminative_geometry_correction,
    enforce_residual_partition,
    pairwise_alias_scores,
    pairwise_alias_weights,
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
        alias_names=("first", "first-specific", "second", "second-specific"),
    )


def test_pairwise_alias_weights_are_normalized_within_positive_class() -> None:
    bank = _bank()
    weights, separation = pairwise_alias_weights(bank.features, bank, DGERConfig())
    assert weights.shape == (2, 2, 4)
    torch.testing.assert_close(weights[0, 1].sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[1, 0].sum(), torch.tensor(1.0))
    assert torch.equal(weights[0, 1, 2:], torch.zeros(2))
    assert torch.equal(weights[1, 0, :2], torch.zeros(2))
    assert separation[0, 1] > 0
    torch.testing.assert_close(separation, separation.T)


def test_pairwise_alias_scores_keep_competitor_conditioning() -> None:
    bank = _bank()
    weights, _ = pairwise_alias_weights(bank.features, bank, DGERConfig())
    pixels = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    scores = pairwise_alias_scores(pixels @ bank.features.T, weights)
    assert scores.shape == (1, 2, 2, 2)
    assert scores[0, 0, 0, 1] > scores[0, 0, 1, 0]
    assert scores[0, 1, 1, 0] > scores[0, 1, 0, 1]


def test_discriminative_correction_is_zero_sum_and_can_reverse_a_pair() -> None:
    base = torch.tensor([[[0.55, 0.50], [0.60, 0.40]]])
    pair = torch.zeros((1, 2, 2, 2))
    pair[0, :, 0, 1] = torch.tensor([-0.20, -0.15])
    pair[0, :, 1, 0] = torch.tensor([0.20, 0.15])
    geometry = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    separation = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    config = DGERConfig(evidence_topk=1, laplacian_strength=2.0, correction_strength=4.0,
                        correction_clip=0.5)
    corrected, _, propagated, gate, _ = discriminative_geometry_correction(
        base, pair, pair, geometry, separation, torch.ones((1, 2), dtype=torch.bool), config
    )
    torch.testing.assert_close(corrected.sum(-1), base.sum(-1))
    assert torch.all(propagated < 0)
    assert torch.all(gate > 0)
    assert corrected[0, 0, 1] > corrected[0, 0, 0]


def test_agreeing_geometry_prediction_is_not_reinforced() -> None:
    base = torch.tensor([[[0.55, 0.50], [0.60, 0.40]]])
    pair = torch.zeros((1, 2, 2, 2))
    pair[0, :, 0, 1] = 0.2
    pair[0, :, 1, 0] = -0.2
    geometry = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    separation = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    corrected, _, propagated, gate, _ = discriminative_geometry_correction(
        base, pair, pair, geometry, separation, torch.ones((1, 2), dtype=torch.bool),
        DGERConfig(evidence_topk=1),
    )
    assert torch.all(propagated > 0)
    assert torch.equal(gate, torch.zeros_like(gate))
    torch.testing.assert_close(corrected, base)


def test_complete_dger_tensor_path() -> None:
    torch.manual_seed(17)
    model = DGERSegmenter(_TinyBackbone(), DGERConfig(evidence_topk=2))
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    result = model.read_prepared(prepared, _bank())
    assert result.geometry_logits.shape == result.logits.shape == (1, 2, 2, 2)
    assert result.pair_ids.shape == (1, 2, 2, 2)
    assert result.propagated_evidence.shape == (1, 2, 2)
    assert result.correction_gate.shape == (1, 2, 2)
    assert torch.isfinite(result.logits).all()


def test_residual_partition_crossing_is_reverted() -> None:
    base = torch.tensor([[[0.4, 0.6], [0.7, 0.3]]])
    corrected = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]])
    output, reverted = enforce_residual_partition(
        corrected, base, ("background", "target"), ("background",)
    )
    torch.testing.assert_close(output, base)
    assert reverted.tolist() == [[True, True]]


def _consensus_case():
    base = torch.tensor([[[0.50, 0.49, 0.20], [0.50, 0.49, 0.20]]])
    local = torch.tensor([[[0.10, 0.60, 0.00], [0.10, 0.60, 0.00]]])
    pairs = local[..., :, None].expand(1, 2, 3, 3).clone()
    graph = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    return base, local, pairs, graph, valid


def _correct(base, local, pairs, graph, valid, *, native=None, native_pairs=None, names=None):
    return consensus_geometry_correction(
        base, local, local if native is None else native,
        pairs, pairs if native_pairs is None else native_pairs,
        graph, valid, ("a", "b", "c") if names is None else names,
        DGERConfig(consensus_correction=True),
    )


def test_consensus_flip_preserves_other_class_probabilities():
    base, local, pairs, graph, valid = _consensus_case()
    actual, _, _, _, diag = _correct(base, local, pairs, graph, valid)
    assert diag["eligible"].all()
    assert (actual.argmax(-1) == 1).all()
    torch.testing.assert_close(actual[..., 2], base[..., 2], rtol=0, atol=0)
    before, after = (torch.softmax(item / 0.07, dim=-1) for item in (base, actual))
    torch.testing.assert_close(before[..., 2], after[..., 2])
    torch.testing.assert_close(before[..., :2].sum(-1), after[..., :2].sum(-1))


def test_consensus_requires_both_local_views_and_all_class_winner():
    base, local, pairs, graph, valid = _consensus_case()
    native = local.clone()
    native[..., 2] = 0.8  # B beats A, but both views do not name B as the winner.
    actual, *_ = _correct(base, local, pairs, graph, valid, native=native)
    torch.testing.assert_close(actual, base, rtol=0, atol=0)
    native_pairs = -pairs
    actual, *_ = _correct(base, local, pairs, graph, valid, native_pairs=native_pairs)
    torch.testing.assert_close(actual, base, rtol=0, atol=0)


def test_consensus_neighborhood_cannot_overrule_disagreeing_pixel():
    base, local, pairs, graph, valid = _consensus_case()
    pairs[:, 0, 0, 1] = 0.9
    pairs[:, 0, 1, 0] = 0.1
    actual, *_ = _correct(base, local, pairs, graph, valid)
    torch.testing.assert_close(actual[:, 0], base[:, 0], rtol=0, atol=0)
    # The same contrary token also vetoes its neighbor's correction.
    torch.testing.assert_close(actual[:, 1], base[:, 1], rtol=0, atol=0)


def test_consensus_requires_evidence_stronger_than_geometry():
    base, local, pairs, graph, valid = _consensus_case()
    base[..., 0] = 1.2
    actual, *_ = _correct(base, local, pairs, graph, valid)
    torch.testing.assert_close(actual, base, rtol=0, atol=0)


def test_consensus_ignores_invalid_neighbors_and_empty_support():
    base, local, pairs, graph, valid = _consensus_case()
    for mask in (torch.zeros_like(valid), torch.tensor([[True, False]])):
        actual, *_ = _correct(base, local, pairs, graph, mask)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
    actual, *_ = _correct(base[:, :1], local[:, :1], pairs[:, :1], graph[:, :1, :1], valid[:, :1])
    torch.testing.assert_close(actual, base[:, :1], rtol=0, atol=0)


def test_consensus_does_not_modify_residual_pairs():
    base, local, pairs, graph, valid = _consensus_case()
    for names in (("background", "b", "c"), ("a", "background", "c")):
        actual, *_ = _correct(base, local, pairs, graph, valid, names=names)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)


def test_consensus_is_equivariant_to_class_order():
    base, local, pairs, graph, valid = _consensus_case()
    actual, *_ = _correct(base, local, pairs, graph, valid)
    order = torch.tensor([2, 0, 1])
    permuted, *_ = _correct(
        base[..., order], local[..., order], pairs[:, :, order][:, :, :, order], graph, valid,
        names=("c", "a", "b"),
    )
    torch.testing.assert_close(permuted, actual[..., order])


def test_consensus_readout_keeps_legacy_result_and_reuses_text_cache():
    torch.manual_seed(17)
    backbone = _TinyBackbone()
    model = DGERSegmenter(backbone, DGERConfig(consensus_correction=True, evidence_topk=2))
    legacy = DGERSegmenter(backbone, DGERConfig(evidence_topk=2))
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    bank = _bank()
    first = model.read_prepared(prepared, bank)
    second = model.read_prepared(prepared, bank)
    original = legacy.read_prepared(prepared, bank)
    assert len(model._pair_text_cache) == 1
    torch.testing.assert_close(first.legacy_logits, original.logits)
    torch.testing.assert_close(first.logits, second.logits)
    assert torch.isfinite(first.logits).all()
