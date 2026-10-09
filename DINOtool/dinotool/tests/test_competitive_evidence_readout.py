from __future__ import annotations

import torch

from dinotool.competitive_evidence_readout import (
    calibrate_edge_bias,
    redistribute_candidate_attention,
    validated_prior_difference,
)
from dinotool.contrastive_support import (
    ContrastiveSupportConfig,
    _reconstruction_cost,
    reconstruction_solution,
)


def test_reconstruction_solution_preserves_old_cost_api() -> None:
    torch.manual_seed(13)
    query = torch.nn.functional.normalize(torch.randn(1, 5, 7), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(1, 5, 4, 7), dim=-1)
    prior = torch.softmax(torch.randn(1, 5, 4), dim=-1)
    config = ContrastiveSupportConfig(reconstruction_steps=3)
    coefficients, cost = reconstruction_solution(query, candidates, prior, config)
    torch.testing.assert_close(cost, _reconstruction_cost(query, candidates, prior, config))
    torch.testing.assert_close(coefficients.sum(-1), torch.ones(1, 5))


def test_zero_edge_bias_returns_geometry_bit_exact() -> None:
    geometry = torch.tensor([[
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.5, 0.3, 0.1],
        [0.2, 0.2, 0.4, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ]])
    indices = torch.tensor([[[1, 2], [0, 2], [0, 3], [1, 2]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    output, row_kl = redistribute_candidate_attention(
        geometry,
        indices,
        torch.zeros_like(indices, dtype=torch.float32),
        valid,
        maximum_kl=0.03,
        bisection_steps=10,
        epsilon=1e-6,
    )
    assert torch.equal(output, geometry)
    assert torch.count_nonzero(row_kl) == 0


def test_attention_redistribution_preserves_mass_and_untouched_entries() -> None:
    geometry = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    indices = torch.tensor([[[1, 2]]])
    bias = torch.tensor([[[5.0, -5.0]]])
    output, row_kl = redistribute_candidate_attention(
        geometry,
        indices,
        bias,
        torch.ones(1, 1, dtype=torch.bool),
        maximum_kl=0.03,
        bisection_steps=16,
        epsilon=1e-6,
    )
    torch.testing.assert_close(output[..., 0], geometry[..., 0])
    torch.testing.assert_close(output[..., 3], geometry[..., 3])
    torch.testing.assert_close(output.gather(-1, indices).sum(-1), torch.tensor([[0.5]]))
    torch.testing.assert_close(output.sum(-1), geometry.sum(-1))
    assert float(row_kl.max()) <= 0.0301


def test_correct_prior_beats_matched_null_in_synthetic_case() -> None:
    query = torch.tensor([[[1.0, 0.0]]])
    candidates = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
    real = torch.tensor([[[0.90, 0.05, 0.05]]])
    null_a = torch.tensor([[[0.05, 0.90, 0.05]]])
    null_b = torch.tensor([[[0.05, 0.05, 0.90]]])
    edge, gain, raw_gain = validated_prior_difference(
        query,
        candidates,
        real,
        (null_a, null_b),
        torch.ones(1, 1, dtype=torch.bool),
        ContrastiveSupportConfig(
            reconstruction_kl=0.25,
            reconstruction_steps=3,
            reconstruction_step_size=0.1,
        ),
        gain_temperature=0.02,
    )
    assert bool(torch.isfinite(edge).all())
    assert float(raw_gain.item()) > 0
    assert float(gain.item()) > 0
    assert float(edge[..., 0].item()) > 0


def test_invalid_evidence_cannot_change_edges() -> None:
    query = torch.tensor([[[1.0, 0.0]]])
    candidates = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    real = torch.tensor([[[0.9, 0.1]]])
    null = torch.tensor([[[0.1, 0.9]]])
    edge, gain, raw_gain = validated_prior_difference(
        query,
        candidates,
        real,
        (null,),
        torch.zeros(1, 1, dtype=torch.bool),
        ContrastiveSupportConfig(reconstruction_steps=2),
        gain_temperature=0.02,
    )
    assert bool(torch.isfinite(raw_gain).all())
    assert torch.count_nonzero(edge) == 0
    assert torch.count_nonzero(gain) == 0


def test_edge_calibration_preserves_gain_and_removes_common_offset() -> None:
    edge = torch.tensor([[[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]]])
    gain = torch.tensor([[0.2, 0.7]])
    calibrated = calibrate_edge_bias(edge, gain, 1e-6)
    torch.testing.assert_close(calibrated[..., 0, :].mean(-1), torch.zeros(1))
    torch.testing.assert_close(
        calibrated[..., 0, :].square().mean(-1).sqrt(), torch.tensor([0.2])
    )
    assert torch.count_nonzero(calibrated[..., 1, :]) == 0
