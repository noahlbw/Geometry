import torch

from dinotool.config import TLPConfig
from dinotool.tlp import (
    apply_tlp_state,
    build_tlp_state,
    semantic_correlation,
    text_aware_laplacian_propagation,
)


def test_semantic_correlation_is_symmetric() -> None:
    features = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    matrix = semantic_correlation(features, TLPConfig())
    assert torch.allclose(matrix, matrix.T, atol=1e-6)
    assert torch.isfinite(matrix).all()


def test_tlp_preserves_constant_logits() -> None:
    logits = torch.tensor([[[[2.0] * 6] * 5, [[-1.0] * 6] * 5]])
    rgb = torch.full((1, 3, 5, 6), 0.4)
    text = torch.eye(2)
    result, diagnostics = text_aware_laplacian_propagation(
        logits,
        rgb,
        text,
        TLPConfig(smoothing_strength=0.8, cg_tolerance=1e-6),
    )
    assert torch.allclose(result, logits, atol=1e-5)
    assert diagnostics.relative_residual <= 1e-5


def test_tlp_smooths_an_impulse_and_converges() -> None:
    logits = torch.zeros((1, 2, 9, 9))
    logits[:, 0, 4, 4] = 4.0
    logits[:, 1] = -logits[:, 0]
    rgb = torch.full((1, 3, 9, 9), 0.5)
    text = torch.eye(2)
    result, diagnostics = text_aware_laplacian_propagation(
        logits,
        rgb,
        text,
        TLPConfig(smoothing_strength=1.0, cg_max_iterations=100, cg_tolerance=1e-5),
    )
    assert result[0, 0, 4, 4] < logits[0, 0, 4, 4]
    assert result[0, 0, 4, 3] > 0
    assert diagnostics.relative_residual <= 2e-5
    assert torch.isfinite(result).all()


def test_tlp_handles_a_single_patch() -> None:
    logits = torch.tensor([[[[0.3]], [[-0.2]]]])
    result, _ = text_aware_laplacian_propagation(
        logits,
        torch.zeros((1, 3, 1, 1)),
        torch.eye(2),
        TLPConfig(),
    )
    assert torch.allclose(result, logits, atol=1e-6)


def test_frozen_state_reproduces_legacy_tlp() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((1, 3, 8, 9), generator=generator)
    rgb = torch.rand((1, 3, 8, 9), generator=generator)
    text = torch.randn((3, 5), generator=generator)
    config = TLPConfig(cg_max_iterations=80, cg_tolerance=1e-6)
    legacy, legacy_diagnostics = text_aware_laplacian_propagation(logits, rgb, text, config)
    state = build_tlp_state(logits, rgb, text, config)
    replay, replay_diagnostics = apply_tlp_state(logits, state)
    assert torch.allclose(replay, legacy, atol=1e-6, rtol=1e-6)
    assert replay_diagnostics.iterations == legacy_diagnostics.iterations
    assert abs(replay_diagnostics.relative_residual - legacy_diagnostics.relative_residual) < 1e-12


def test_frozen_state_replays_other_logits_without_changing_coefficients() -> None:
    anchor = torch.zeros((1, 2, 6, 7))
    anchor[:, 0, 2:4, 2:5] = 2.0
    rgb = torch.full((1, 3, 6, 7), 0.5)
    text = torch.eye(2)
    state = build_tlp_state(anchor, rgb, text, TLPConfig(cg_tolerance=1e-6))
    held_out = -anchor
    replay, diagnostics = apply_tlp_state(held_out, state)
    assert replay.shape == held_out.shape
    assert torch.isfinite(replay).all()
    assert diagnostics.relative_residual <= 2e-5


def test_frozen_state_rejects_mismatched_grid() -> None:
    logits = torch.zeros((1, 2, 4, 4))
    state = build_tlp_state(logits, torch.zeros((1, 3, 4, 4)), torch.eye(2), TLPConfig())
    try:
        apply_tlp_state(torch.zeros((1, 2, 4, 5)), state)
    except ValueError as error:
        assert "grid dimensions" in str(error)
    else:
        raise AssertionError("Expected mismatched replay grid to fail.")
