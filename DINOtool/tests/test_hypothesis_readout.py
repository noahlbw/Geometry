from __future__ import annotations

import torch

from dinotool.hypothesis_readout import (
    HEROConfig,
    HEROSegmenter,
    consensus_evidence,
    deterministic_alias_split,
    hypothesis_conditioned_relations,
    preserve_residual_partition,
    uniform_subset_scores,
)
from dinotool.tcpr import TCPRTextBank

from test_tcpr import _TinyBackbone


def _bank() -> TCPRTextBank:
    return TCPRTextBank(
        features=torch.nn.functional.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0],
        ]), dim=-1),
        parent_indices=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        canonical_mask=torch.tensor([True, False, False, False, True, False, False, False]),
        class_names=("first", "second"),
        alias_names=("a", "aa", "aaa", "aaaa", "b", "bb", "bbb", "bbbb"),
    )


def test_alias_split_is_balanced_disjoint_and_complete() -> None:
    bank = _bank()
    first = deterministic_alias_split(bank, "fixed")
    second = deterministic_alias_split(bank, "fixed")
    assert torch.equal(first, second)
    assert not bool((first[0] & first[1]).any())
    assert bool((first[0] | first[1]).all())
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        assert int((first[0] & members).sum()) == 2
        assert int((first[1] & members).sum()) == 2


def test_uniform_subset_score_removes_alias_count_bias() -> None:
    scores = torch.tensor([[[0.4, 0.4, 0.1, 0.1]]])
    parents = torch.tensor([0, 0, 1, 1])
    output = uniform_subset_scores(
        scores, parents, 2, torch.ones(4, dtype=torch.bool), temperature=0.07
    )
    torch.testing.assert_close(output, torch.tensor([[[0.4, 0.1]]]), atol=1e-6, rtol=0)


def test_uniform_text_support_exactly_returns_geometry() -> None:
    geometry = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
    alias = torch.zeros((1, 2, 4))
    parents = torch.tensor([0, 0, 1, 1])
    valid = torch.ones((1, 2), dtype=torch.bool)
    output = hypothesis_conditioned_relations(
        geometry,
        alias,
        parents,
        2,
        torch.ones(4, dtype=torch.bool),
        valid,
        HEROConfig(query_chunk=1),
    )
    torch.testing.assert_close(output[:, 0], geometry)
    torch.testing.assert_close(output[:, 1], geometry)


def test_consensus_requires_cross_fold_direction_agreement() -> None:
    first = torch.tensor([1.0, -2.0, 3.0, -4.0])
    second = torch.tensor([0.5, -3.0, -1.0, 2.0])
    output = consensus_evidence(first, second)
    torch.testing.assert_close(output, torch.tensor([0.5, -2.0, 0.0, 0.0]))


def test_residual_partition_preserves_background_boundary() -> None:
    base = torch.tensor([[[0.8, 0.5, 0.4], [0.2, 0.9, 0.7]]])
    delta = torch.tensor([[[0.1, 0.6, -0.2], [0.8, -0.3, 0.5]]])
    output = preserve_residual_partition(
        delta, base, ("background", "first", "second"), ("background",)
    )
    torch.testing.assert_close(output[0, 0], torch.zeros(3))
    assert float(output[0, 1, 0]) == 0.0
    assert float(output[0, 1, 1]) == 0.0
    assert float(output[0, 1, 2]) > 0.0
    before_background = base.argmax(-1) == 0
    after_background = (base + output).argmax(-1) == 0
    assert torch.equal(before_background, after_background)


def test_complete_hero_tensor_path_returns_all_readouts() -> None:
    torch.manual_seed(7)
    model = HEROSegmenter(_TinyBackbone(), HEROConfig(query_chunk=2))
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    result = model.read_prepared(prepared, _bank())
    expected = (1, 2, 2, 2)
    assert result.geometry_logits.shape == expected
    assert result.cross_logits.shape == expected
    assert result.consensus_logits.shape == expected
    assert result.same_group_logits.shape == expected
    assert result.fold_deltas.shape == (2, 1, 4, 2)
    assert torch.isfinite(result.cross_logits).all()
    assert torch.isfinite(result.consensus_logits).all()

