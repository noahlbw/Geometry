"""Contracts for fixed-slot alias evidence and conservative ownership."""
from __future__ import annotations

import torch

from dinotool.competitive_ownership import (
    CompetitiveOwnershipReadout, OwnershipConfig, alias_families,
    positive_excess_readout,
)
from dinotool.gear_ov import GearObservations, GearText, class_scores
from dinotool.tcpr import TCPRTextBank


def _bank():
    return TCPRTextBank(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([True, False, True, False]),
        ("road", "other"), ("road", "street", "other", "ground"),
    )


def test_full_retention_is_exactly_original_aggregation():
    scores = torch.tensor([[4.0, 1.0, 2.0, -1.0]])
    reference = torch.zeros_like(scores)
    expected = class_scores(scores, _bank().parent_indices, 2)
    actual = positive_excess_readout(scores, reference, torch.ones_like(scores),
                                     _bank().parent_indices, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_rejection_only_suppresses_positive_excess_and_keeps_denominator():
    scores = torch.tensor([[4.0, -2.0, 3.0, -3.0]])
    baseline = torch.zeros_like(scores)
    retention = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    result = positive_excess_readout(scores, baseline, retention,
                                     _bank().parent_indices, 2)
    expected = class_scores(torch.tensor([[0.0, -2.0, 3.0, -3.0]]),
                            _bank().parent_indices, 2)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_family_exclusion_and_unidentifiable_fallback():
    bank = _bank()
    families = alias_families(bank, 0.999)
    assert sorted(len(group) for group in families) == [1, 1, 1, 1]
    solver = CompetitiveOwnershipReadout(bank, OwnershipConfig(family_cosine=0.999))
    scores = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
    teacher, reference, identifiable = solver._family_reference(scores)
    assert bool(identifiable.all())
    assert teacher[0, 0, 0] == scores[0, 1]
    assert reference[0, 0] == scores[0, 1]


def test_synthetic_tile_is_finite_without_any_supervision():
    bank = _bank()
    solver = CompetitiveOwnershipReadout(bank)
    text = GearText(bank.features, bank.parent_indices, torch.empty(4, 0),
                    torch.eye(4), 2)
    views = GearObservations(
        local_aligned=torch.randn(32, 32, 2),
        detail_aligned=torch.randn(64, 64, 2),
        context_aligned=torch.randn(16, 16, 2),
        detail_raw=torch.randn(64, 64, 3),
        context_raw=torch.randn(16, 16, 3),
        actual_height=377, actual_width=489,
    )
    logits, diagnostics = solver.solve(views, text, torch.zeros(1, 2, 512, 512))
    assert logits.shape == (1, 2, 512, 512)
    assert bool(torch.isfinite(logits).all())
    assert 0 <= diagnostics["mean_rejection"] <= 1


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_") and callable(check):
            check()
    print("4 competitive ownership checks passed")
