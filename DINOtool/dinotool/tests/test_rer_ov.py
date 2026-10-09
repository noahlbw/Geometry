from __future__ import annotations

import torch

from dinotool.contrastive_support import (
    ContrastiveSupportConfig,
    ContrastiveSupportResult,
    build_contrastive_support,
    sparse_geometry_candidates,
)
from dinotool.evidence_consensus import EvidenceConsensusConfig, solve_evidence_consensus
from dinotool.evidence_vocabulary import EncodedEvidenceBank


def _support(availability: torch.Tensor, evidence: torch.Tensor) -> ContrastiveSupportResult:
    pair = torch.tensor([[[[0, 1]], [[0, 1]], [[0, 1]]]], dtype=torch.long)
    indices = torch.tensor([[[1, 2], [0, 2], [1, 0]]], dtype=torch.long)
    weights = torch.full((1, 3, 2), 0.5)
    return ContrastiveSupportResult(
        pair_ids=pair,
        evidence=evidence,
        availability=availability,
        phrase_support=torch.ones(1, 3, 2),
        phrase_directions=torch.ones(1, 3, 1, 2),
        candidate_indices=indices,
        candidate_weights=weights,
    )


def test_sparse_geometry_candidates_excludes_self_and_normalizes() -> None:
    geometry = torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    indices, weights = sparse_geometry_candidates(geometry, valid, 2, 1e-6)
    query = torch.arange(3)[None, :, None]
    assert not bool((indices == query).any())
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3))


def test_consensus_zero_evidence_exactly_returns_anchor() -> None:
    scores = torch.tensor([[[0.2, 0.1], [0.1, 0.3], [0.4, 0.2]]])
    availability = torch.zeros(1, 3, 1, 1)
    evidence = torch.zeros_like(availability)
    result = solve_evidence_consensus(
        scores, _support(availability, evidence), torch.ones(1, 3, dtype=torch.bool)
    )
    torch.testing.assert_close(result.scores, scores)
    torch.testing.assert_close(result.delta, torch.zeros_like(scores))


def test_consensus_positive_pair_evidence_increases_first_margin() -> None:
    scores = torch.zeros(1, 3, 2)
    availability = torch.ones(1, 3, 1, 1)
    evidence = torch.ones_like(availability)
    result = solve_evidence_consensus(
        scores,
        _support(availability, evidence),
        torch.ones(1, 3, dtype=torch.bool),
        EvidenceConsensusConfig(spatial_weight=0.0, evidence_weight=1.0),
    )
    assert bool(((result.scores[..., 0] - result.scores[..., 1]) > 0).all())


def test_consensus_preserves_common_class_offset() -> None:
    scores = torch.zeros(1, 3, 2)
    availability = torch.ones(1, 3, 1, 1)
    evidence = torch.ones_like(availability)
    result = solve_evidence_consensus(
        scores, _support(availability, evidence), torch.ones(1, 3, dtype=torch.bool)
    )
    torch.testing.assert_close(result.delta.sum(-1), torch.zeros(1, 3), atol=1e-6, rtol=0)


def test_contrastive_support_full_tensor_path_is_finite() -> None:
    torch.manual_seed(7)
    batch, patches, channels, classes, phrases = 1, 9, 8, 3, 6
    raw = torch.nn.functional.normalize(torch.randn(batch, patches, channels), dim=-1)
    aligned = torch.nn.functional.normalize(torch.randn(batch, patches, channels), dim=-1)
    class_text = torch.nn.functional.normalize(torch.randn(classes, channels), dim=-1)
    evidence_text = torch.nn.functional.normalize(torch.randn(phrases, channels), dim=-1)
    geometry = torch.softmax(torch.randn(batch, patches, patches), dim=-1)
    scores = torch.randn(batch, patches, classes)
    valid = torch.ones(batch, patches, dtype=torch.bool)
    bank = EncodedEvidenceBank(
        features=evidence_text,
        family_indices=torch.tensor([0, 0, 0, 1, 1, 1]),
        anchor_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
        priors=torch.ones(phrases),
        phrases=tuple(f"p{index}" for index in range(phrases)),
        families=("appearance", "structure"),
        class_names=("a", "b", "c"),
    )
    result = build_contrastive_support(
        raw,
        aligned,
        geometry,
        class_text,
        scores,
        bank,
        valid,
        ContrastiveSupportConfig(
            candidate_members=4,
            reconstruction_steps=2,
            query_chunk=4,
        ),
    )
    assert result.evidence.shape == (batch, patches, 3, 2)
    assert result.availability.shape == result.evidence.shape
    assert bool(torch.isfinite(result.evidence).all())
    assert bool(torch.isfinite(result.availability).all())
