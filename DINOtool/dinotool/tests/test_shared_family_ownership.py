"""Behavioral checks for shared family ownership and the matched readout."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearObservations, GearText, class_scores
from dinotool.shared_family_ownership import SharedFamilyOwnershipReadout
from dinotool.tcpr import TCPRTextBank


def _bank() -> TCPRTextBank:
    return TCPRTextBank(
        F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                  [0., 1.], [0.6, 0.8]]), dim=-1),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([True, False, True, False]),
        ("wall", "roof"), ("wall", "roof", "roof", "building"),
    )


def _views() -> GearObservations:
    return GearObservations(
        torch.randn(32, 32, 2), torch.randn(64, 64, 2),
        torch.randn(16, 16, 2), torch.randn(64, 64, 3),
        torch.randn(16, 16, 3), 377, 489,
    )


def _baseline(views: GearObservations, bank: TCPRTextBank) -> torch.Tensor:
    logits = []
    for features in (views.local_aligned, views.detail_aligned, views.context_aligned):
        scores = features @ bank.features.T / 0.07
        scores = class_scores(scores, bank.parent_indices, bank.class_count) * 0.07
        logits.append(F.interpolate(scores.permute(2, 0, 1)[None], size=(512, 512),
                                    mode="bilinear", align_corners=False))
    return torch.stack(logits).mean(0)


def test_one_semantic_family_has_one_reference_across_parent_classes():
    bank = _bank()
    solver = SharedFamilyOwnershipReadout(bank)
    scores = torch.tensor([[0., 8., 8., 5.]])
    teacher, reference, identifiable = solver._reference(scores)
    roof_family = int(solver.alias_to_family[1])
    assert roof_family == int(solver.alias_to_family[2])
    assert bool(identifiable[roof_family])
    torch.testing.assert_close(teacher[0, roof_family], torch.tensor([0., 5.]))
    torch.testing.assert_close(reference[0, 1], torch.tensor(0.))
    torch.testing.assert_close(reference[0, 2], torch.tensor(5.))


def test_single_family_per_class_recovers_multiscale():
    bank = TCPRTextBank(torch.eye(2), torch.tensor([0, 1]),
                        torch.tensor([True, True]), ("wall", "roof"),
                        ("wall", "roof"))
    views = _views()
    multiscale = _baseline(views, bank)
    text = GearText(bank.features, bank.parent_indices, torch.empty(2, 0),
                    torch.eye(2), 2)
    predicted, diagnostics = SharedFamilyOwnershipReadout(bank).solve(
        views, text, multiscale)
    torch.testing.assert_close(predicted, multiscale, atol=1e-6, rtol=1e-6)
    assert diagnostics["identifiable_family_fraction"] == 0


def test_shared_ownership_is_independent_of_duplicate_source_label():
    bank = _bank()
    solver = SharedFamilyOwnershipReadout(bank)
    swapped = TCPRTextBank(bank.features, torch.tensor([0, 1, 0, 1]),
                           torch.tensor([True, True, False, False]),
                           bank.class_names, bank.alias_names)
    other = SharedFamilyOwnershipReadout(swapped)
    scores = torch.tensor([[0., 8., 8., 5.]])
    teacher, _, _ = solver._reference(scores)
    swapped_teacher, _, _ = other._reference(scores)
    roof_family = int(solver.alias_to_family[1])
    posterior = teacher.softmax(-1)[0, roof_family]
    assert posterior[1] > 0.99
    assert int(solver.parents[1]) != int(solver.parents[2])
    torch.testing.assert_close(posterior, swapped_teacher.softmax(-1)[
        0, other.alias_to_family[1]])


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_") and callable(check):
            check()
    print("3 shared-family ownership checks passed")
