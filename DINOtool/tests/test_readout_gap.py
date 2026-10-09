from __future__ import annotations

import numpy as np

from dinotool.readout_gap import (
    affine_probe_parameters,
    apply_dual_ridge_map,
    centered_classifier_directions,
    error_decomposition,
    fit_dual_ridge_map,
    fit_logistic_probe,
    native_text_scores,
    paired_image_bootstrap_delta,
    prediction_from_scores,
    token_text_scores,
)


def test_native_and_token_pooling_are_distinct_and_well_formed() -> None:
    tokens = np.asarray([[[3.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    native = native_text_scores(tokens.mean(axis=1), text)
    token = token_text_scores(tokens, text)
    assert native.shape == token.shape == (1, 2)
    assert not np.allclose(native, token)
    assert prediction_from_scores(native, [10, 11]).tolist() == [10]


def test_error_decomposition_includes_reverse_loss_and_balanced_net_gap() -> None:
    target = np.asarray([0, 0, 1, 1])
    open_prediction = np.asarray([0, 1, 1, 0])
    visual_prediction = np.asarray([0, 0, 0, 1])
    summary, per_class, kinds = error_decomposition(target, open_prediction, visual_prediction, [0, 1])
    assert kinds.tolist() == ["A", "B", "C", "B"]
    assert summary["A"] == 1
    assert summary["B"] == 2
    assert summary["C"] == 1
    assert summary["D"] == 0
    assert summary["balanced_net_accuracy_gap_percent"] == 25.0
    assert per_class[0]["rer"] == 1.0


def test_paired_bootstrap_assigns_one_multiplicity_per_image() -> None:
    target = np.asarray([0, 1, 0, 1])
    reference = np.asarray([0, 1, 1, 0])
    candidate = np.asarray([0, 1, 0, 1])
    result = paired_image_bootstrap_delta(
        target, reference, candidate, ["image-a", "image-a", "image-b", "image-b"], [0, 1], samples=200, seed=7
    )
    assert result["invalid_draws"] == 0
    assert result["ci95_low_points"] == 0.0
    assert result["ci95_high_points"] == 100.0


def test_affine_probe_inversion_preserves_multiclass_logits() -> None:
    values = np.asarray([[-2.0, 0.0], [-1.0, 0.2], [1.0, 0.1], [2.0, -0.1], [0.0, 2.0], [0.1, 3.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    fit = fit_logistic_probe(values, labels, c_value=1.0, seed=1)
    weights, intercept = affine_probe_parameters(fit)
    raw = values / np.linalg.norm(values, axis=1, keepdims=True)
    standardized_logits = ((raw - fit.mean) / fit.scale) @ fit.coefficients.T + fit.intercept
    inverted_logits = raw @ weights.T + intercept
    assert np.allclose(standardized_logits, inverted_logits, atol=1e-5)
    directions, centered, _ = centered_classifier_directions(fit)
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-6)
    assert directions.shape == weights.shape


def test_dual_ridge_map_recovers_seen_pairs_without_a_dense_map() -> None:
    text = np.eye(3, dtype=np.float32)
    visual = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mapping = fit_dual_ridge_map(text, visual, lambda_value=1e-6)
    assert np.allclose(apply_dual_ridge_map(mapping, text), visual, atol=1e-4)
