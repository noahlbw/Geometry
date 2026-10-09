from __future__ import annotations

import torch
from types import SimpleNamespace

from dinotool.contextual_phrase_readout import ContextualPhraseConfig
from dinotool.contrastive_context import (
    ContrastiveContextConfig, alias_matches, local_pair_graph,
    pairwise_evidence, reconstruct_pairs,
)
from dinotool.contrastive_context_v2 import (
    METHODS as V2_METHODS, PairwiseReconciliationSegmenter,
)
from dinotool.tcpr import TCPRConfig, TCPRTextBank


def _bank(features: torch.Tensor) -> TCPRTextBank:
    return TCPRTextBank(
        features=features,
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("building", "roof"),
        alias_names=("building", "roof", "rooftop", "roof plane"),
    )


def test_identical_cross_class_alias_has_zero_discriminative_evidence():
    bank = _bank(torch.tensor([
        [0., 1., 0.], [1., 0., 0.],
        [1., 0., 0.], [0., 0., 1.],
    ]))
    matching = alias_matches(bank)
    assert matching[2][0][1].tolist() == [1, 2]
    assert matching[3][0][0].tolist() == [2, 1]
    scores = torch.tensor([[[0.2, 0.9, 0.9, 0.1]]])
    forward = pairwise_evidence(scores, matching, 0.07)
    reverse_bank = _bank(bank.features[[2, 3, 0, 1]])
    reverse = pairwise_evidence(scores[..., [2, 3, 0, 1]], alias_matches(reverse_bank), 0.07)
    torch.testing.assert_close(forward, -reverse)
    assert forward.item() < 0
    shared_only = torch.tensor([[[0.9, 0.9, 0.9, 0.9]]])
    torch.testing.assert_close(pairwise_evidence(shared_only, matching, 0.07),
                               torch.zeros(1, 1, 1))


def test_signed_pair_reconstruction_and_zero_revision():
    bank = _bank(torch.tensor([
        [1., 0.], [0., 1.], [0., 1.], [1., 0.],
    ]))
    matching = alias_matches(bank)
    base = torch.tensor([[[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    original_pair = base[..., 0:1] - base[..., 1:2]
    config = ContrastiveContextConfig(solver_steps=64)
    torch.testing.assert_close(reconstruct_pairs(
        base, (original_pair, original_pair), matching, None, valid, config,
    ), base)
    target = torch.tensor([[[-0.4], [-0.4], [1.0]]])
    updated = reconstruct_pairs(base, (target, target), matching, None, valid, config)
    assert updated[0, 0, 0] < base[0, 0, 0]
    assert updated[0, 0, 1] > base[0, 0, 1]
    torch.testing.assert_close(updated[0, 2], base[0, 2])
    assert torch.isfinite(updated).all()


def test_graph_edges_are_local_symmetric_and_do_not_cross_mask():
    geometry = torch.ones(1, 4, 4) / 4
    evidence = torch.tensor([[[1.], [1.], [-1.], [-1.]]])
    valid = torch.tensor([[True, True, False, True]])
    ids, weights = local_pair_graph(geometry, evidence, valid, 1, 4, 1, 0.07)
    graph = torch.zeros(1, 4, 4, 1)
    graph[:, torch.arange(4)[:, None], ids] = weights
    torch.testing.assert_close(graph, graph.transpose(1, 2))
    assert graph[0, 1, 2, 0] == 0
    assert graph[0, 0, 2, 0] == 0
    assert graph[0, 2].abs().sum() == 0
    assert graph[0, :, 2].abs().sum() == 0
    assert weights.sum(2).max() <= 1.000001


def test_v2_readout_keeps_matched_baselines_and_returns_finite_fields():
    torch.manual_seed(9)
    model = PairwiseReconciliationSegmenter(
        SimpleNamespace(device=torch.device("cpu"), patch_size=16),
        ContextualPhraseConfig(), TCPRConfig(maximum_aliases_per_class=2),
    )
    bank = _bank(torch.tensor([
        [1., 0., 0.], [0., 1., 0.],
        [0., 1., 0.], [0., 0., 1.],
    ]))
    features = torch.randn(1, 4, 3)
    prepared = SimpleNamespace(
        geometry_projected=features,
        native_projected=features + 0.1,
        geometry_patch_conditional=torch.eye(4)[None],
        grid_height=2, grid_width=2,
    )
    context = torch.randn(1, 4, 2, 2)
    outputs, diagnostics = model.read_grounded(
        prepared, None, context, bank, height=32, width=32,
        top=0, left=0, box=(0, 0, 32),
        valid_mask=torch.tensor([[True, True], [True, False]]),
    )
    assert set(outputs) == set(V2_METHODS)
    for result in outputs.values():
        assert result.shape == (1, 2, 2, 2)
        assert torch.isfinite(result).all()
    assert diagnostics["mean_final_correction"] >= 0
