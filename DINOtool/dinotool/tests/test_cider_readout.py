from __future__ import annotations

import torch

from dinotool.cider_readout import (
    decision_consistent_edges,
    gather_pair_directions,
    pairwise_text_directions,
)


def test_pairwise_text_basis_is_antisymmetric() -> None:
    text = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0], [0.1, 0.9, 0.0],
    ]), dim=-1)
    parents = torch.tensor([0, 0, 1, 1])
    canonical = torch.tensor([True, False, True, False])
    weights = torch.tensor([[0.6, 0.4, 0.7, 0.3]])
    directions = pairwise_text_directions(
        text, weights, parents, canonical, 2, 0.07, 1e-6
    )
    torch.testing.assert_close(directions[:, 0, 1], -directions[:, 1, 0])
    torch.testing.assert_close(
        directions[:, 0, 1].norm(dim=-1), torch.ones(1), atol=1e-6, rtol=0
    )


def test_gather_pair_directions_uses_query_specific_pairs() -> None:
    directions = torch.zeros(1, 3, 3, 2)
    directions[0, 0, 1] = torch.tensor([1.0, 0.0])
    directions[0, 2, 1] = torch.tensor([0.0, 1.0])
    pairs = torch.tensor([[[0, 1], [2, 1]]])
    gathered = gather_pair_directions(directions, pairs)
    torch.testing.assert_close(gathered, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))


def test_decision_consistency_rewards_matching_signs_and_rejects_invalid() -> None:
    semantic = torch.tensor([[[2.0, -2.0, 2.0, -2.0]]])
    influence = torch.tensor([[[3.0, -3.0, -3.0, 3.0]]])
    edge, magnitude = decision_consistent_edges(
        semantic,
        influence,
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        1e-6,
    )
    assert float(edge[0, 0, 0]) > 0
    assert float(edge[0, 0, 1]) > 0
    assert float(edge[0, 0, 2]) == 0
    assert float(edge[0, 0, 3]) == 0
    assert float(magnitude.item()) > 0

    blocked, blocked_magnitude = decision_consistent_edges(
        semantic,
        influence,
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        1e-6,
    )
    assert torch.count_nonzero(blocked) == 0
    assert torch.count_nonzero(blocked_magnitude) == 0


def test_decision_demand_scales_evidence_without_changing_edge_order() -> None:
    semantic = torch.tensor([[[2.0, -2.0]]])
    influence = torch.tensor([[[3.0, -3.0]]])
    high_edge, high = decision_consistent_edges(
        semantic, influence, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), 1e-6
    )
    low_edge, low = decision_consistent_edges(
        semantic, influence, torch.full((1, 1), 0.1),
        torch.ones(1, 1, dtype=torch.bool), 1e-6
    )
    torch.testing.assert_close(high_edge, low_edge)
    torch.testing.assert_close(low, high * 0.1)
