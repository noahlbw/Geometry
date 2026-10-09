"""Semantic contracts for the frozen v3 cross-dataset readout."""
from __future__ import annotations

import math
import torch

from dinotool.discriminative_ownership import (
    DiscriminativeOwnershipReadout, negative_class_contrast, ownership_seed,
)
from dinotool.gear_ov import GearText
from dinotool.tcpr import TCPRTextBank
from dinotool.tests.test_shared_family_ownership import _bank, _views, _baseline


def _disjoint_bank():
    return TCPRTextBank(torch.eye(4), torch.tensor([0, 0, 1, 1]),
                        torch.tensor([True, False, True, False]),
                        ("road", "building"), ("road", "street", "building", "house"))


def test_absent_family_retains_all_rival_evidence_and_shift_invariance():
    solver = DiscriminativeOwnershipReadout(_disjoint_bank())
    scores = torch.tensor([[0., 0., 4., 4.], [-2., 1., 3., 5.]])
    teacher, _, identifiable = solver._reference(scores)
    assert identifiable.all()
    torch.testing.assert_close(teacher[0, 0], torch.tensor([0., 4.]))
    shifted, _, _ = solver._reference(scores + torch.tensor([[5.], [-7.]]))
    torch.testing.assert_close(shifted.softmax(-1), teacher.softmax(-1))


def test_reference_matches_independent_leave_family_enumeration():
    bank = _bank()
    solver = DiscriminativeOwnershipReadout(bank)
    scores = torch.tensor([[0., 8., 8., 5.], [3., -2., -2., 1.]])
    actual, _, identifiable = solver._reference(scores)
    for g, excluded in enumerate(solver.families):
        if not identifiable[g]:
            continue
        for c in range(bank.class_count):
            values = []
            for members in solver.families:
                if members == excluded:
                    continue
                indices = [i for i in members if int(bank.parent_indices[i]) == c]
                if indices:
                    values.append(torch.logsumexp(scores[:, indices], -1) - math.log(len(indices)))
            expected = torch.logsumexp(torch.stack(values, -1), -1) - math.log(len(values))
            torch.testing.assert_close(actual[:, g, c], expected)


def test_agreeing_uncertainty_and_conflicting_views_are_unknown():
    for classes in (2, 5, 8, 12):
        seed = ownership_seed(torch.full((3, 2, classes), 1 / classes))
        torch.testing.assert_close(seed[..., -1], torch.ones(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(seed[..., :-1], torch.zeros(2, classes), atol=1e-6, rtol=0)
    opposed = torch.tensor([[[1., 0.]], [[0., 1.]]])
    torch.testing.assert_close(ownership_seed(opposed), torch.tensor([[0., 0., 1.]]))
    certain = torch.tensor([[[0., 1.]], [[0., 1.]], [[0., 1.]]])
    torch.testing.assert_close(ownership_seed(certain), torch.tensor([[0., 1., 0.]]))


def test_contrast_retains_discriminative_words_and_rejects_wrong_parent():
    # Two spatial blocks per class. Word 0 supports c0; word 1 supports c1.
    values = torch.tensor([[4., 0.], [4., 0.], [0., 4.], [0., 4.]])
    response = values[None].expand(3, -1, -1)
    anchors = torch.zeros(4, 2, 2, dtype=torch.bool)
    anchors[:2, :, 0] = True
    anchors[2:, :, 1] = True
    harm, supported = negative_class_contrast(response, anchors, 2, 1.96)
    assert harm[0, 0, 1] == 0
    assert harm[1, 0, 1] == 1
    assert harm[0, 1, 0] == 1
    assert harm[1, 1, 0] == 0
    assert supported[:, 0, 1].all()
    # A contradictory view removes evidence for that same suppression.
    disagreement = response.clone()
    disagreement[2] = response[2].flip(0)
    assert negative_class_contrast(disagreement, anchors, 2, 1.96)[0].sum() == 0


def test_insufficient_or_ambiguous_reference_never_forces_deletion():
    response = torch.tensor([[[0.], [5.]]]).expand(3, -1, -1)
    anchors = torch.eye(2, dtype=torch.bool)[:, None, :]
    harm, supported = negative_class_contrast(response, anchors, 2, 1.96)
    assert not bool(supported.any())
    assert harm.sum() == 0
    response = torch.ones(3, 4, 1)
    anchors = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)[:, None]
    assert negative_class_contrast(response, anchors, 2, 1.96)[0].sum() == 0


def test_full_readout_fallback_matches_all_native_views_on_partial_tile():
    bank = TCPRTextBank(torch.eye(2), torch.tensor([0, 1]),
                        torch.tensor([True, True]), ("wall", "roof"), ("wall", "roof"))
    torch.manual_seed(19)
    views = _views()
    baseline = _baseline(views, bank)
    text = GearText(bank.features, bank.parent_indices, torch.empty(2, 0), torch.eye(2), 2)
    outputs, diagnostics = DiscriminativeOwnershipReadout(bank).solve_all(views, text, baseline)
    for output in outputs.values():
        torch.testing.assert_close(output, baseline, atol=1e-6, rtol=1e-6)
    assert diagnostics["discriminative_mean_rejection"] == 0
    assert diagnostics["ownership_mean_unknown"] == 1


if __name__ == "__main__":
    torch.set_num_threads(2)
    checks = [(name, check) for name, check in list(globals().items())
              if name.startswith("test_") and callable(check)]
    for name, check in checks:
        check()
        print("PASS", name)
    print(f"{len(checks)} discriminative ownership checks passed")
