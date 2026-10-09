from __future__ import annotations

import torch

from dinotool.grounded_context import (
    alias_text_margin, center_innovation, context_box, context_crop, evidence_graph,
    geometry_alias_reliability, geometry_alias_selection, geometry_alias_weights, solve_innovation,
    weighted_alias_scores,
)
from dinotool.tcpr import TCPRTextBank


def _bank() -> TCPRTextBank:
    features = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], [0.98, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], [0.2, 0.98, 0.0, 0.0],
    ])
    return TCPRTextBank(
        features=torch.nn.functional.normalize(features, dim=-1),
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("a", "b"),
        alias_names=("a", "a descriptor", "b", "b descriptor"),
    )


def test_scene_offset_does_not_change_centered_innovation():
    torch.manual_seed(12)
    residual = torch.randn(1, 7, 4)
    valid = torch.tensor([[True] * 6 + [False]])
    offset = torch.tensor([[[1.0, -3.0, 2.0, 0.5]]])
    torch.testing.assert_close(center_innovation(residual, valid),
                               center_innovation(residual + offset, valid))
    assert torch.equal(center_innovation(residual, valid)[:, -1], torch.zeros(1, 4))


def test_solver_matches_dense_linear_system_and_conserves_centered_mass():
    torch.manual_seed(5)
    valid = torch.tensor([[True] * 7 + [False]])
    geometry = torch.rand(1, 8, 8).softmax(-1)
    scores = torch.randn(1, 8, 3)
    graph = evidence_graph(geometry, scores, valid, 0.07)
    torch.testing.assert_close(graph, graph.transpose(-1, -2))
    assert graph.sum(-1).max() <= 1.000001
    assert graph[:, -1].sum() == 0 and graph[:, :, -1].sum() == 0
    residual = center_innovation(torch.randn(1, 8, 5), valid)
    operator = 2 * torch.eye(8)[None] + torch.diag_embed(graph.sum(-1)) - graph
    expected = torch.linalg.solve(operator, residual)
    actual = solve_innovation(residual, graph, valid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.sum(1), torch.zeros(1, 5), atol=1e-6, rtol=0)


def test_solver_does_not_cross_disconnected_visual_components():
    graph = torch.tensor([[[0., 1., 0., 0.], [1., 0., 0., 0.],
                           [0., 0., 0., 1.], [0., 0., 1., 0.]]])
    residual = torch.tensor([[[1.], [-1.], [0.], [0.]]])
    actual = solve_innovation(residual, graph, torch.ones(1, 4, dtype=torch.bool))
    assert actual[:, 2:].abs().max() == 0


def test_identical_observations_require_no_correction():
    graph = torch.ones(1, 4, 4) / 4
    valid = torch.ones(1, 4, dtype=torch.bool)
    residual = torch.zeros(1, 4, 3)
    assert solve_innovation(residual, graph, valid).abs().max() == 0


def test_context_field_is_bounded_and_square_without_image_stretching():
    assert context_box(1024, 1024, 384, 384, 512, 1024) == (0, 0, 1024)
    assert context_box(3000, 4000, 2488, 3488, 512, 1024) == (1976, 2976, 1024)
    image = torch.arange(3 * 32 * 48).reshape(3, 32, 48).float()
    box = context_box(32, 48, 0, 0, 512, 1024)
    crop = context_crop(image, box)
    assert crop.shape == (1, 3, 48, 48)
    torch.testing.assert_close(crop[0, :, :32], image)


def test_semantic_graph_and_solver_are_equivariant_to_class_permutation():
    torch.manual_seed(9)
    valid = torch.ones(1, 6, dtype=torch.bool)
    geometry = torch.rand(1, 6, 6).softmax(-1)
    scores = torch.randn(1, 6, 3)
    permutation = torch.tensor([2, 0, 1])
    first = evidence_graph(geometry, scores, valid, 0.07)
    second = evidence_graph(geometry, scores[..., permutation], valid, 0.07)
    torch.testing.assert_close(first, second)


def test_geometry_alias_selection_keeps_canonical_alias_and_normalizes_weights():
    bank = _bank()
    valid = torch.tensor([[True, True, True, False]])
    local = torch.tensor([[[0.9, 0.4, 0.1, 0.0],
                           [0.8, 0.7, 0.1, 0.1],
                           [0.0, 0.1, 0.8, 0.7],
                           [0.0, 0.0, 0.0, 0.0]]])
    geometry = torch.eye(4)[None]
    reliability, _ = geometry_alias_reliability(local, geometry, bank, valid, 0.07)
    admitted, weights = geometry_alias_selection(
        reliability, bank, valid, topk=1, temperature=0.07, canonical_prior=0.2
    )
    assert bool(admitted[0, 0]) and bool(admitted[0, 2])
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        torch.testing.assert_close(weights[0, :3, members].sum(-1), torch.ones(3))
    assert weights[0, 3].abs().max() == 0


def test_local_alias_weights_are_finite_on_padding_and_normalized_on_valid_tokens():
    bank = _bank()
    valid = torch.tensor([[True, True, True, False]])
    reliability = torch.tensor([[[0.9, 0.1, 0.1, 0.2],
                                 [0.8, 0.7, 0.2, 0.1],
                                 [0.1, 0.3, 0.8, 0.6],
                                 [0.0, 0.0, 0.0, 0.0]]])
    weights = geometry_alias_weights(
        reliability, bank, valid, temperature=0.2, uniform_prior=0.35, canonical_prior=0.2
    )
    assert torch.isfinite(weights).all()
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        torch.testing.assert_close(weights[0, :3, members].sum(-1), torch.ones(3))
    assert weights[0, 3].abs().max() == 0


def test_padding_scores_and_graph_stay_finite():
    bank = _bank()
    valid = torch.tensor([[True, True, True, False]])
    aliases = torch.randn(1, 4, 4)
    reliability = torch.randn_like(aliases)
    weights = geometry_alias_weights(
        reliability, bank, valid, temperature=0.2, uniform_prior=0.35, canonical_prior=0.2
    )
    scores = weighted_alias_scores(aliases, bank, weights, 0.07)
    graph = evidence_graph(torch.eye(4)[None], scores, valid, 0.07)
    correction = solve_innovation(torch.randn(1, 4, 3), graph, valid)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(graph).all()
    assert torch.isfinite(correction).all()


def test_rejected_aliases_do_not_change_weighted_class_score():
    bank = _bank()
    aliases = torch.tensor([[[0.2, 0.6, 0.1, 0.7]]])
    weights = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    first = weighted_alias_scores(aliases, bank, weights, 0.07)
    aliases[..., 1] = 100.0
    aliases[..., 3] = 100.0
    second = weighted_alias_scores(aliases, bank, weights, 0.07)
    torch.testing.assert_close(first, second)


def test_text_margin_penalizes_alias_nearer_to_the_competing_canonical_class():
    bank = _bank()
    margin = alias_text_margin(bank, 0.07)
    assert margin[0] > margin[1]
    assert margin[2] > margin[3]
