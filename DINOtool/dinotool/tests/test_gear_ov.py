"""Checks the alias-space reconstruction and spatial attribution contracts."""
from __future__ import annotations

import torch

from dinotool.gear_ov import (
    GearConfig, GearOVSegmenter, GearText, build_support, build_text_basis, center,
    observe_context, solve_evidence,
)


def _toy_text() -> GearText:
    return GearText(
        features=torch.eye(2), parents=torch.arange(2),
        basis=torch.tensor([[1.0], [-1.0]]) / 2**0.5,
        whitening=torch.eye(2), class_count=2,
    )


def test_text_basis_removes_shared_alias_mode() -> None:
    from types import SimpleNamespace

    features = torch.tensor([[1.0, 0.2, -0.3], [0.1, 0.9, 0.4], [-0.3, 0.2, 1.0]])
    bank = SimpleNamespace(features=features, parent_indices=torch.arange(3),
                           class_count=3, validate=lambda: None)
    text = build_text_basis(bank, 0.1)
    torch.testing.assert_close(text.basis.mean(0), torch.zeros(text.basis.shape[1]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(text.basis @ text.basis.T,
                               (text.features - text.features.mean(0))
                               @ (text.features - text.features.mean(0)).T,
                               atol=1e-6, rtol=1e-5)


def test_exact_query_groups_do_not_add_class_names() -> None:
    class Backbone:
        patch_size = 16

        def encode_text_aliases(self, groups, *, templates):
            assert groups == (("surface,other",), ("facade", "building wall"))
            assert templates
            return (torch.eye(3), torch.tensor([0, 1, 1]),
                    torch.tensor([True, True, False]))

    model = GearOVSegmenter(Backbone())
    bank = model.encode_alias_groups(("other", "wall"),
                                     (("surface,other",), ("facade", "building wall")))
    assert bank.alias_names == ("surface,other", "facade", "building wall")
    assert bank.class_names == ("other", "wall")


def test_context_responsibility_is_alias_specific() -> None:
    scores = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.5, 0.5]])
    variable = scores.clone().requires_grad_(True)
    response = observe_context(variable, indices, weights, 1.0)
    first = torch.autograd.grad(response[0, 0], variable, retain_graph=True)[0]
    second = torch.autograd.grad(response[0, 1], variable)[0]
    assert first[0, 0] > 0.99 and first[1, 0] < 0.01
    assert second[1, 1] > 0.99 and second[0, 1] < 0.01


def test_reconstruction_is_identity_when_observations_agree() -> None:
    local = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.5, 0.5]])
    context = center(observe_context(local, indices, weights, 1.0))
    reconstructed, diagnostics = solve_evidence(
        local, local, context, _toy_text(), indices, weights,
        torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool), GearConfig(),
    )
    torch.testing.assert_close(reconstructed, local)
    assert diagnostics["iterations"] == 0
    assert diagnostics["objective_initial"] < 1e-10


def test_detail_and_context_can_drive_signed_correction() -> None:
    local = torch.zeros(2, 2)
    target = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.5, 0.5]])
    context = center(observe_context(target, indices, weights, 1.0))
    reconstructed, diagnostics = solve_evidence(
        local, target, context, _toy_text(), indices, weights,
        torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool), GearConfig(),
    )
    assert diagnostics["objective_final"] < diagnostics["objective_initial"]
    assert reconstructed[0, 0] > 0 and reconstructed[0, 1] < 0
    assert reconstructed[1, 0] < 0 and reconstructed[1, 1] > 0


def test_support_retains_center_children_and_valid_mass() -> None:
    raw_detail = torch.randn(64, 64, 4)
    raw_context = torch.randn(16, 16, 4)
    fine_valid = torch.ones(64, 64, dtype=torch.bool)
    fine_valid[50:] = False
    context_valid = torch.ones(16, 16, dtype=torch.bool)
    context_valid[13:] = False
    indices, weights = build_support(raw_detail, raw_context, fine_valid,
                                     context_valid, GearConfig())
    assert indices.shape == (256, 64)
    assert weights.shape == (256, 64)
    assert bool((weights[context_valid.flatten()].sum(1) - 1).abs().max() < 1e-5)
    assert not bool(weights[~context_valid.flatten()].any())
    assert bool(fine_valid.flatten()[indices[weights > 0]].all())
    center_indices = indices[0, :16]
    assert set(center_indices.tolist()) == {
        row * 64 + col for row in range(4) for col in range(4)
    }


if __name__ == "__main__":
    checks = [value for name, value in list(globals().items())
              if name.startswith("test_") and callable(value)]
    for check in checks:
        check()
    print(f"{len(checks)} GEAR-OV checks passed.")
