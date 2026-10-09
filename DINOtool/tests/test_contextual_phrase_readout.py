from __future__ import annotations

import torch

from dinotool.contextual_phrase_readout import (
    conditioned_alias_scores,
    masked_geometry,
    normalized_logmeanexp,
    sample_overview_aliases,
)
from dinotool.hypothesis_readout import uniform_subset_scores
from dinotool.tcpr import TCPRTextBank


def _bank() -> TCPRTextBank:
    return TCPRTextBank(
        features=torch.eye(4),
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("a", "b"),
        alias_names=("a0", "a1", "b0", "b1"),
    )


def test_uniform_local_aliases_reduce_to_unconditioned_context() -> None:
    bank = _bank()
    local = torch.zeros((1, 3, 4))
    context = torch.tensor([[[0.1, 0.5, 0.4, 0.2]]]).expand(1, 3, 4)
    actual = conditioned_alias_scores(local, context, bank, 0.07)
    expected = uniform_subset_scores(
        context, bank.parent_indices, bank.class_count,
        torch.ones(4, dtype=torch.bool), 0.07,
    )
    torch.testing.assert_close(actual, expected)


def test_local_phrase_posterior_selects_matching_context_alias() -> None:
    bank = _bank()
    local = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    context = torch.tensor([[[0.8, 0.1, 0.7, 0.2]]])
    scores = conditioned_alias_scores(local, context, bank, 0.07)
    assert scores[0, 0, 0] > 0.75
    assert scores[0, 0, 1] < 0.30


def test_overview_sampling_respects_original_image_coordinates() -> None:
    overview = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    sampled = sample_overview_aliases(
        overview,
        image_height=32,
        image_width=32,
        tile_top=0,
        tile_left=0,
        grid_height=2,
        grid_width=2,
        patch_size=16,
    )
    torch.testing.assert_close(sampled[0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_masked_geometry_removes_invalid_keys_and_uses_identity_for_invalid_queries() -> None:
    geometry = torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]])
    valid = torch.tensor([[True, True, False]])
    actual = masked_geometry(geometry, valid, 1e-6)
    torch.testing.assert_close(actual[0, :2, 2], torch.zeros(2))
    torch.testing.assert_close(actual[0, :2].sum(-1), torch.ones(2))
    torch.testing.assert_close(actual[0, 2], torch.tensor([0.0, 0.0, 1.0]))


def test_normalized_scale_union_is_identity_for_equal_evidence() -> None:
    values = torch.tensor([[[0.2, 0.2], [0.7, 0.7]]])
    torch.testing.assert_close(normalized_logmeanexp(values, 0.07), values[..., 0])


def test_normalized_scale_union_retains_strong_single_scale_evidence() -> None:
    values = torch.tensor([[[0.8, 0.0]]])
    union = normalized_logmeanexp(values, 0.07)
    assert 0.7 < union.item() < 0.8
