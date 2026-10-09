"""Small, source-only utilities for the frozen DINO readout-gap audit.

This module intentionally contains no segmentation architecture. Its inputs
are extracted region features, and fitting only receives source cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


EPSILON = 1e-12


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return float32 L2 normalization without accepting zero vectors."""
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if bool(np.any(norms <= EPSILON)):
        raise ValueError("Cannot normalize a zero-norm feature vector.")
    return (array / norms).astype(np.float32, copy=False)


def region_means(tokens: np.ndarray) -> np.ndarray:
    """Mean pool a [regions, tokens, channels] tensor in its native space."""
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("tokens must be a non-empty [regions, tokens, channels] tensor.")
    return values.mean(axis=1, dtype=np.float64).astype(np.float32)


def native_text_scores(region_vectors: np.ndarray, text_vectors: np.ndarray) -> np.ndarray:
    """Cosine(mean(region tokens), text), retaining the given feature space."""
    region = l2_normalize(np.asarray(region_vectors, dtype=np.float32))
    text = l2_normalize(np.asarray(text_vectors, dtype=np.float32))
    if region.ndim != 2 or text.ndim != 2 or region.shape[1] != text.shape[1]:
        raise ValueError("region/text vectors must have matching channel dimensions.")
    return (region @ text.T).astype(np.float32)


def token_text_scores(tokens: np.ndarray, text_vectors: np.ndarray) -> np.ndarray:
    """Mean of per-token cosines: CAFe's input-cost pooling rule."""
    values = np.asarray(tokens, dtype=np.float32)
    text = l2_normalize(np.asarray(text_vectors, dtype=np.float32))
    if values.ndim != 3 or values.shape[-1] != text.shape[-1]:
        raise ValueError("tokens/text dimensionality does not match.")
    normalized_tokens = l2_normalize(values)
    return np.einsum("ntd,cd->ntc", normalized_tokens, text, optimize=True).mean(axis=1).astype(np.float32)


def top2_margin(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have at least two candidates.")
    partitioned = np.partition(values, kth=values.shape[1] - 2, axis=1)
    return (partitioned[:, -1] - partitioned[:, -2]).astype(np.float32)


def prediction_from_scores(scores: np.ndarray, class_indices: Sequence[int]) -> np.ndarray:
    candidates = np.asarray(class_indices, dtype=np.int64)
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(candidates):
        raise ValueError("scores columns must equal the candidate vocabulary length.")
    return candidates[np.argmax(values, axis=1)]


def nearest_centroid_scores(
    source_region_vectors: np.ndarray,
    source_labels: np.ndarray,
    evaluation_region_vectors: np.ndarray,
    class_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Source-centroid visual readout using the locked normalization sequence."""
    source = l2_normalize(np.asarray(source_region_vectors, dtype=np.float32))
    target = l2_normalize(np.asarray(evaluation_region_vectors, dtype=np.float32))
    labels = np.asarray(source_labels, dtype=np.int64)
    candidates = np.asarray(class_indices, dtype=np.int64)
    centroids: list[np.ndarray] = []
    for label in candidates:
        rows = source[labels == label]
        if not len(rows):
            raise ValueError(f"No source regions for class {int(label)}.")
        centroids.append(l2_normalize(rows.mean(axis=0, dtype=np.float64, keepdims=True))[0])
    centroid_array = np.stack(centroids).astype(np.float32)
    return (target @ centroid_array.T).astype(np.float32), centroid_array


@dataclass(frozen=True)
class ProbeFit:
    """Source-only standardized L2-logistic fit and provenance."""

    classes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    effective_rank: int
    c_value: float
    seed: int

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size + self.intercept.size)


def affine_probe_parameters(fit: ProbeFit) -> tuple[np.ndarray, np.ndarray]:
    """Invert the scaler so probe logits are expressed in native feature space.

    For a binary sklearn fit, ``coef_`` represents one signed decision normal.
    Expanding it into ±half normals gives an explicit, centered two-class gauge
    with exactly the same pairwise decision function.
    """
    if np.any(fit.scale <= EPSILON):
        raise ValueError("Probe scaler contains a zero scale.")
    coefficients = np.asarray(fit.coefficients, dtype=np.float64) / fit.scale[None, :]
    intercept = np.asarray(fit.intercept, dtype=np.float64) - (np.asarray(fit.coefficients, dtype=np.float64) * (fit.mean / fit.scale)[None, :]).sum(axis=1)
    if len(fit.classes) == 2 and coefficients.shape[0] == 1:
        coefficients = np.concatenate((-0.5 * coefficients, 0.5 * coefficients), axis=0)
        intercept = np.concatenate((-0.5 * intercept, 0.5 * intercept), axis=0)
    if coefficients.shape[0] != len(fit.classes):
        raise ValueError("Probe coefficient rows do not align with its class order.")
    return coefficients.astype(np.float32), intercept.astype(np.float32)


def centered_classifier_directions(fit: ProbeFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a common-gauge centered visual direction per fitted class."""
    weights, intercept = affine_probe_parameters(fit)
    centered = weights - weights.mean(axis=0, keepdims=True)
    return l2_normalize(centered), centered.astype(np.float32), intercept


def pairwise_direction_rows(
    class_indices: Sequence[int], text_directions: np.ndarray, visual_directions: np.ndarray
) -> list[dict[str, float | int]]:
    """Pairwise angular agreement, immune to a common vector shift in weights."""
    classes = np.asarray(class_indices, dtype=np.int64)
    text = np.asarray(text_directions, dtype=np.float32)
    visual = np.asarray(visual_directions, dtype=np.float32)
    if text.shape != visual.shape or text.shape[0] != len(classes):
        raise ValueError("Text/visual direction arrays must align with classes.")
    rows: list[dict[str, float | int]] = []
    for first in range(len(classes)):
        for second in range(first + 1, len(classes)):
            text_normal = l2_normalize((text[first] - text[second])[None, :])[0]
            visual_normal = l2_normalize((visual[first] - visual[second])[None, :])[0]
            cosine = float(np.clip(np.dot(text_normal, visual_normal), -1.0, 1.0))
            rows.append({
                "class_a": int(classes[first]), "class_b": int(classes[second]), "cosine": cosine,
                "angle_degrees": float(np.degrees(np.arccos(cosine))),
            })
    return rows


@dataclass(frozen=True)
class DualRidgeMap:
    """Compact dual-form ridge map from text directions to visual directions."""

    anchors: np.ndarray
    solve: np.ndarray
    targets: np.ndarray
    lambda_value: float

    @property
    def effective_parameter_count(self) -> int:
        return int(self.anchors.size + self.solve.size + self.targets.size)


def fit_dual_ridge_map(text_directions: np.ndarray, visual_targets: np.ndarray, *, lambda_value: float) -> DualRidgeMap:
    """Fit average-squared-loss ridge in dual form without a D×D parameter matrix."""
    text = np.asarray(text_directions, dtype=np.float64)
    target = np.asarray(visual_targets, dtype=np.float64)
    if text.ndim != 2 or target.shape != text.shape or len(text) < 1 or lambda_value <= 0:
        raise ValueError("Ridge map needs matched non-empty [classes, channels] rows and positive lambda.")
    gram = text @ text.T
    solve = np.linalg.inv(gram + len(text) * float(lambda_value) * np.eye(len(text)))
    return DualRidgeMap(text.astype(np.float32), solve.astype(np.float32), target.astype(np.float32), float(lambda_value))


def apply_dual_ridge_map(mapping: DualRidgeMap, text_directions: np.ndarray) -> np.ndarray:
    values = np.asarray(text_directions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != mapping.anchors.shape[1]:
        raise ValueError("Text directions do not match the ridge-map space.")
    coefficients = values @ mapping.anchors.T @ mapping.solve
    return (coefficients @ mapping.targets).astype(np.float32)


@dataclass(frozen=True)
class ResidualRidgeMap:
    """Identity/global base plus a source-fitted residual map in a fixed basis."""

    base: DualRidgeMap | None
    residual: DualRidgeMap
    basis: np.ndarray | None
    kind: str


def _residual_basis(residuals: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or not 1 <= rank < min(values.shape):
        raise ValueError("Residual rank must be a non-trivial reduction for the observed seen-class residual matrix.")
    _, _, right = np.linalg.svd(values, full_matrices=False)
    return right[:rank].T.astype(np.float32)


def fit_identity_residual_map(
    text_directions: np.ndarray, visual_directions: np.ndarray, *, lambda_value: float, rank: int | None = None
) -> ResidualRidgeMap:
    """Fit T2 (rank constrained) or T4 (full) around the identity text prior."""
    text = np.asarray(text_directions, dtype=np.float32)
    visual = np.asarray(visual_directions, dtype=np.float32)
    residuals = visual - text
    if rank is None:
        return ResidualRidgeMap(None, fit_dual_ridge_map(text, residuals, lambda_value=lambda_value), None, "identity_full")
    basis = _residual_basis(residuals, rank)
    coefficients = residuals @ basis
    return ResidualRidgeMap(None, fit_dual_ridge_map(text, coefficients, lambda_value=lambda_value), basis, "identity_lowrank")


def fit_global_residual_map(
    text_directions: np.ndarray, visual_directions: np.ndarray, *, lambda_value: float, rank: int
) -> ResidualRidgeMap:
    """Fit T3: freeze a global ridge map, then fit a low-rank residual map."""
    text = np.asarray(text_directions, dtype=np.float32)
    visual = np.asarray(visual_directions, dtype=np.float32)
    base = fit_dual_ridge_map(text, visual, lambda_value=lambda_value)
    residuals = visual - apply_dual_ridge_map(base, text)
    basis = _residual_basis(residuals, rank)
    return ResidualRidgeMap(base, fit_dual_ridge_map(text, residuals @ basis, lambda_value=lambda_value), basis, "global_lowrank")


def apply_residual_ridge_map(mapping: ResidualRidgeMap, text_directions: np.ndarray) -> np.ndarray:
    values = np.asarray(text_directions, dtype=np.float32)
    base = values if mapping.base is None else apply_dual_ridge_map(mapping.base, values)
    residual = apply_dual_ridge_map(mapping.residual, values)
    if mapping.basis is not None:
        residual = residual @ mapping.basis.T
    return (base + residual).astype(np.float32)


def fit_logistic_probe(
    source_region_vectors: np.ndarray,
    source_labels: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> ProbeFit:
    """Fit the pre-registered L2 multiclass logistic comparator source-only."""
    if c_value <= 0:
        raise ValueError("c_value must be positive.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = l2_normalize(np.asarray(source_region_vectors, dtype=np.float32))
    labels = np.asarray(source_labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("source feature/label rows do not match.")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=float(c_value), class_weight="balanced", max_iter=5000,
        random_state=int(seed), solver="lbfgs",
    )
    classifier.fit(standardized, labels)
    return ProbeFit(
        classes=np.asarray(classifier.classes_, dtype=np.int64),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        effective_rank=int(np.linalg.matrix_rank(standardized)),
        c_value=float(c_value), seed=int(seed),
    )


def probe_scores(fit: ProbeFit, region_vectors: np.ndarray) -> np.ndarray:
    features = l2_normalize(np.asarray(region_vectors, dtype=np.float32)).astype(np.float64)
    if features.ndim != 2 or features.shape[1] != fit.mean.shape[0]:
        raise ValueError("evaluation feature dimensionality does not match probe fit.")
    return (((features - fit.mean) / fit.scale) @ fit.coefficients.T + fit.intercept).astype(np.float32)


def classification_summary(target: np.ndarray, prediction: np.ndarray, class_indices: Sequence[int]) -> dict[str, Any]:
    """Region-level classification values; deliberately not segmentation mIoU."""
    from sklearn.metrics import confusion_matrix, f1_score

    target_array = np.asarray(target, dtype=np.int64)
    prediction_array = np.asarray(prediction, dtype=np.int64)
    classes = np.asarray(class_indices, dtype=np.int64)
    if target_array.shape != prediction_array.shape:
        raise ValueError("target/prediction shapes do not match.")
    matrix = confusion_matrix(target_array, prediction_array, labels=classes)
    support = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), support, out=np.zeros(len(classes), dtype=np.float64), where=support > 0)
    return {
        "balanced_accuracy_percent": float(recalls.mean() * 100.0),
        "macro_f1_percent": float(f1_score(target_array, prediction_array, labels=classes, average="macro", zero_division=0) * 100.0),
        "accuracy_percent": float(np.mean(target_array == prediction_array) * 100.0),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class_recall_percent": {str(label): float(recall * 100.0) for label, recall in zip(classes, recalls)},
    }


def error_decomposition(
    target: np.ndarray,
    open_prediction: np.ndarray,
    visual_prediction: np.ndarray,
    class_indices: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Compute paired A/B/C/D errors and class-balanced aggregates."""
    y = np.asarray(target, dtype=np.int64)
    open_pred = np.asarray(open_prediction, dtype=np.int64)
    visual_pred = np.asarray(visual_prediction, dtype=np.int64)
    if y.shape != open_pred.shape or y.shape != visual_pred.shape:
        raise ValueError("target and predictions must have matching shapes.")
    open_correct = open_pred == y
    visual_correct = visual_pred == y
    error_type = np.full(len(y), "D", dtype="U1")
    error_type[open_correct & visual_correct] = "A"
    error_type[(~open_correct) & visual_correct] = "B"
    error_type[open_correct & (~visual_correct)] = "C"
    rows: list[dict[str, Any]] = []
    balanced_rer: list[float] = []
    balanced_reverse: list[float] = []
    balanced_net: list[float] = []
    for label in np.asarray(class_indices, dtype=np.int64):
        mask = y == label
        a = int(np.sum(mask & (error_type == "A")))
        b = int(np.sum(mask & (error_type == "B")))
        c = int(np.sum(mask & (error_type == "C")))
        d = int(np.sum(mask & (error_type == "D")))
        n = a + b + c + d
        rer = None if b + d == 0 else b / (b + d)
        reverse = None if a + c == 0 else c / (a + c)
        net = np.nan if n == 0 else (b - c) / n
        if rer is not None:
            balanced_rer.append(rer)
        if reverse is not None:
            balanced_reverse.append(reverse)
        if not np.isnan(net):
            balanced_net.append(float(net))
        rows.append({
            "class_index": int(label), "A": a, "B": b, "C": c, "D": d, "support": n,
            "rer": rer, "reverse_loss": reverse,
            "net_accuracy_gap_percent": None if np.isnan(net) else float(net * 100.0),
            "visual_recall_independence_reference": None if n == 0 else float((a + b) / n),
            "low_error_support": bool((b + d) < 20),
        })
    total_a, total_b, total_c, total_d = (int(np.sum(error_type == item)) for item in "ABCD")
    return ({
        "A": total_a, "B": total_b, "C": total_c, "D": total_d,
        "pooled_rer": None if total_b + total_d == 0 else float(total_b / (total_b + total_d)),
        "pooled_reverse_loss": None if total_a + total_c == 0 else float(total_c / (total_a + total_c)),
        "pooled_net_accuracy_gap_percent": float((total_b - total_c) / len(y) * 100.0),
        "balanced_rer": None if not balanced_rer else float(np.mean(balanced_rer)),
        "balanced_reverse_loss": None if not balanced_reverse else float(np.mean(balanced_reverse)),
        "balanced_net_accuracy_gap_percent": None if not balanced_net else float(np.mean(balanced_net) * 100.0),
    }, rows, error_type)


def paired_image_bootstrap_delta(
    target: np.ndarray,
    reference_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    groups: Sequence[str],
    class_indices: Sequence[int],
    *, samples: int, seed: int, alpha: float = 0.05,
) -> dict[str, Any]:
    """Whole-image paired bootstrap for BA(candidate)-BA(reference)."""
    if samples < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("samples must be positive and alpha must be in (0, 1).")
    y = np.asarray(target, dtype=np.int64)
    reference = np.asarray(reference_prediction, dtype=np.int64)
    candidate = np.asarray(candidate_prediction, dtype=np.int64)
    group_array = np.asarray(groups)
    if not (y.shape == reference.shape == candidate.shape == group_array.shape):
        raise ValueError("bootstrap inputs must share their row shape.")
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    classes = np.asarray(class_indices, dtype=np.int64)
    reference_correct, candidate_correct = reference == y, candidate == y
    generator = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    invalid = 0
    for draw_index in range(samples):
        multiplicity = np.bincount(generator.integers(0, len(unique_groups), size=len(unique_groups)), minlength=len(unique_groups))
        row_weights = multiplicity[inverse]
        class_deltas: list[float] = []
        for label in classes:
            mask = y == label
            denominator = int(row_weights[mask].sum())
            if denominator == 0:
                class_deltas = []
                break
            correctness_delta = candidate_correct[mask].astype(np.int8) - reference_correct[mask].astype(np.int8)
            class_deltas.append(float((row_weights[mask] * correctness_delta).sum() / denominator))
        if not class_deltas:
            deltas[draw_index] = np.nan
            invalid += 1
        else:
            deltas[draw_index] = np.mean(class_deltas) * 100.0
    valid_deltas = deltas[~np.isnan(deltas)]
    if not len(valid_deltas):
        raise RuntimeError("Every paired bootstrap draw lacked an evaluation class.")
    return {
        "samples": int(samples), "alpha": float(alpha), "invalid_draws": int(invalid),
        "mean_delta_points": float(valid_deltas.mean()),
        "ci95_low_points": float(np.quantile(valid_deltas, alpha / 2.0)),
        "ci95_high_points": float(np.quantile(valid_deltas, 1.0 - alpha / 2.0)),
    }


def paired_confusions(
    target: np.ndarray, open_prediction: np.ndarray, visual_prediction: np.ndarray, class_indices: Iterable[int]
) -> list[dict[str, int]]:
    """All observed ordered open-readout confusions with paired recoveries."""
    y, open_pred, visual_pred = (np.asarray(value, dtype=np.int64) for value in (target, open_prediction, visual_prediction))
    rows: list[dict[str, int]] = []
    for truth in class_indices:
        for predicted in class_indices:
            if truth == predicted:
                continue
            mask = (y == truth) & (open_pred == predicted)
            count = int(mask.sum())
            if count:
                rows.append({
                    "gt_class_index": int(truth), "open_predicted_class_index": int(predicted), "count": count,
                    "recoverable_count": int(np.sum(mask & (visual_pred == y))),
                    "reverse_count": int(np.sum((y == truth) & (visual_pred == predicted) & (open_pred == y))),
                })
    return rows
