from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal CPU environments
    cv2 = None


@dataclass(frozen=True)
class AlignmentReport:
    method: str
    feature_dim: int
    stored_parameters: int
    effective_degrees_of_freedom: int
    fitted_patch_pairs: int
    regularization: float | None = None


@dataclass(frozen=True)
class SpatialMetrics:
    region_iou: float
    boundary_f1: float
    connectivity_recall: float
    leakage_ratio: float


class AlignmentStatistics:
    """Streaming sufficient statistics for layer-to-final linear alignment."""

    def __init__(self, feature_dim: int, *, dtype: torch.dtype = torch.float64) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.cross = torch.zeros((feature_dim, feature_dim), dtype=dtype)
        self.gram = torch.zeros((feature_dim, feature_dim), dtype=dtype)
        self.patch_pairs = 0

    def update(self, source: Tensor, target: Tensor) -> None:
        source = _as_rows(source, self.feature_dim).detach().to(device="cpu", dtype=self.dtype)
        target = _as_rows(target, self.feature_dim).detach().to(device="cpu", dtype=self.dtype)
        if source.shape != target.shape:
            raise ValueError("source and target alignment features must have identical shape.")
        self.cross.add_(source.transpose(0, 1) @ target)
        self.gram.add_(source.transpose(0, 1) @ source)
        self.patch_pairs += int(source.shape[0])

    def orthogonal_procrustes(
        self,
        *,
        solve_device: torch.device | str | None = None,
    ) -> tuple[Tensor, AlignmentReport]:
        if self.patch_pairs < 1:
            raise RuntimeError("No patch pairs were accumulated.")
        cross = self.cross if solve_device is None else self.cross.to(solve_device)
        left, _, right_t = torch.linalg.svd(cross, full_matrices=False)
        matrix = (left @ right_t).to(device="cpu", dtype=torch.float32)
        dim = self.feature_dim
        return matrix, AlignmentReport(
            method="orthogonal_procrustes",
            feature_dim=dim,
            stored_parameters=dim * dim,
            effective_degrees_of_freedom=dim * (dim - 1) // 2,
            fitted_patch_pairs=self.patch_pairs,
        )

    def ridge(
        self,
        regularization: float,
        *,
        solve_device: torch.device | str | None = None,
    ) -> tuple[Tensor, AlignmentReport]:
        if self.patch_pairs < 1:
            raise RuntimeError("No patch pairs were accumulated.")
        if regularization <= 0:
            raise ValueError("regularization must be positive.")
        gram = self.gram if solve_device is None else self.gram.to(solve_device)
        cross = self.cross if solve_device is None else self.cross.to(solve_device)
        identity = torch.eye(self.feature_dim, dtype=self.dtype, device=gram.device)
        matrix = torch.linalg.solve(gram + regularization * identity, cross).to(device="cpu", dtype=torch.float32)
        dim = self.feature_dim
        return matrix, AlignmentReport(
            method="ridge_to_final",
            feature_dim=dim,
            stored_parameters=dim * dim,
            effective_degrees_of_freedom=dim * dim,
            fitted_patch_pairs=self.patch_pairs,
            regularization=regularization,
        )

    def ridge_many(
        self,
        regularizations: Sequence[float],
        *,
        solve_device: torch.device | str | None = None,
    ) -> dict[float, tuple[Tensor, AlignmentReport]]:
        """Solve several Ridge candidates from one eigendecomposition."""
        values = tuple(float(value) for value in regularizations)
        if not values or any(value <= 0 for value in values):
            raise ValueError("regularizations must contain positive values.")
        if self.patch_pairs < 1:
            raise RuntimeError("No patch pairs were accumulated.")
        gram = self.gram if solve_device is None else self.gram.to(solve_device)
        cross = self.cross if solve_device is None else self.cross.to(solve_device)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0.0)
        projected_cross = eigenvectors.transpose(0, 1) @ cross
        dim = self.feature_dim
        results = {}
        for value in values:
            scaled = projected_cross / (eigenvalues + value).unsqueeze(1)
            matrix = (eigenvectors @ scaled).to(device="cpu", dtype=torch.float32)
            results[value] = (
                matrix,
                AlignmentReport(
                    method="ridge_to_final",
                    feature_dim=dim,
                    stored_parameters=dim * dim,
                    effective_degrees_of_freedom=dim * dim,
                    fitted_patch_pairs=self.patch_pairs,
                    regularization=value,
                ),
            )
        return results


def apply_alignment(features: Tensor, matrix: Tensor | None) -> Tensor:
    if matrix is None:
        return F.normalize(features.float(), dim=-1)
    if features.shape[-1] != matrix.shape[0] or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Alignment matrix must be square and match the feature dimension.")
    return F.normalize(features.float() @ matrix.to(features.device, dtype=torch.float32), dim=-1)


def cosine_logits(features: Tensor, text_features: Tensor, matrix: Tensor | None = None) -> Tensor:
    aligned = apply_alignment(features, matrix)
    text = F.normalize(text_features.float().to(aligned.device), dim=-1)
    if aligned.shape[-1] != text.shape[-1]:
        raise ValueError("Image and text feature dimensions do not match.")
    return aligned @ text.transpose(0, 1)


def grid_edge_affinity(
    features: Tensor,
    height: int,
    width: int,
    *,
    temperature: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Return fixed four-neighbour edge weights from feature or head tensors.

    ``features`` is either ``[P, D]`` or ``[P, heads, head_dim]``. Head-level
    similarities are averaged before applying the common temperature.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if features.shape[0] != height * width:
        raise ValueError("feature count does not match the requested patch grid.")
    if features.ndim == 2:
        normalized = F.normalize(features.float(), dim=-1).reshape(height, width, -1)
        horizontal_similarity = (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1)
        vertical_similarity = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    elif features.ndim == 3:
        normalized = F.normalize(features.float(), dim=-1).reshape(height, width, features.shape[1], -1)
        horizontal_similarity = (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1).mean(dim=-1)
        vertical_similarity = (normalized[:-1] * normalized[1:]).sum(dim=-1).mean(dim=-1)
    else:
        raise ValueError("features must have shape [P,D] or [P,heads,head_dim].")
    horizontal = torch.exp((horizontal_similarity.clamp(-1.0, 1.0) - 1.0) / temperature)
    vertical = torch.exp((vertical_similarity.clamp(-1.0, 1.0) - 1.0) / temperature)
    return horizontal, vertical


def combine_edge_affinities(
    affinities: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    if not affinities:
        raise ValueError("At least one affinity is required.")
    horizontal_shapes = {tuple(item[0].shape) for item in affinities}
    vertical_shapes = {tuple(item[1].shape) for item in affinities}
    if len(horizontal_shapes) != 1 or len(vertical_shapes) != 1:
        raise ValueError("All affinities must share the same grid shape.")
    horizontal = torch.stack([item[0] for item in affinities]).mean(dim=0)
    vertical = torch.stack([item[1] for item in affinities]).mean(dim=0)
    return horizontal, vertical


def seeded_grid_diffusion(
    initial_scores: Tensor,
    horizontal: Tensor,
    vertical: Tensor,
    *,
    seed_labels: Tensor | None = None,
    iterations: int = 20,
    restart: float = 0.15,
) -> Tensor:
    """Diffuse class scores on a fixed four-neighbour grid with clamped seeds."""

    if initial_scores.ndim != 3:
        raise ValueError("initial_scores must have shape [classes,height,width].")
    batched_seeds = None if seed_labels is None else seed_labels.unsqueeze(0)
    return seeded_grid_diffusion_batched(
        initial_scores.unsqueeze(0),
        horizontal.unsqueeze(0),
        vertical.unsqueeze(0),
        seed_labels=batched_seeds,
        iterations=iterations,
        restart=restart,
    )[0]


def seeded_grid_diffusion_batched(
    initial_scores: Tensor,
    horizontal: Tensor,
    vertical: Tensor,
    *,
    seed_labels: Tensor | None = None,
    iterations: int = 20,
    restart: float = 0.15,
) -> Tensor:
    """Batched equivalent of :func:`seeded_grid_diffusion`.

    Each batch item may use a different affinity graph and seed map.  This is
    important for the semantic/spatial pair matrix: evaluating all layer pairs
    in one tensor avoids thousands of tiny GPU launches without changing the
    fixed propagation rule.
    """

    if initial_scores.ndim != 4:
        raise ValueError("initial_scores must have shape [batch,classes,height,width].")
    batch, classes, height, width = initial_scores.shape
    if horizontal.shape != (batch, height, max(width - 1, 0)) or vertical.shape != (
        batch,
        max(height - 1, 0),
        width,
    ):
        raise ValueError("Affinity edge shapes do not match the score grid.")
    if iterations < 1 or not 0.0 <= restart <= 1.0:
        raise ValueError("iterations and restart are invalid.")
    initial = torch.softmax(initial_scores.float(), dim=1)
    scores = initial.clone()
    horizontal = horizontal.to(device=scores.device, dtype=scores.dtype)
    vertical = vertical.to(device=scores.device, dtype=scores.dtype)
    degree = torch.ones((batch, height, width), device=scores.device, dtype=scores.dtype)
    if horizontal.numel():
        degree[:, :, :-1] += horizontal
        degree[:, :, 1:] += horizontal
    if vertical.numel():
        degree[:, :-1] += vertical
        degree[:, 1:] += vertical
    seed_mask = None
    seed_one_hot = None
    if seed_labels is not None:
        if seed_labels.shape != (batch, height, width):
            raise ValueError("seed_labels must match the score grid.")
        seed_labels = seed_labels.to(scores.device)
        seed_mask = seed_labels >= 0
        invalid = seed_mask & (seed_labels >= classes)
        if bool(invalid.any()):
            raise ValueError("seed_labels contains a class outside the score tensor.")
        seed_one_hot = F.one_hot(seed_labels.clamp_min(0).long(), classes).permute(0, 3, 1, 2).to(scores.dtype)
    for _ in range(iterations):
        propagated = scores.clone()
        if horizontal.numel():
            propagated[:, :, :, :-1] += scores[:, :, :, 1:] * horizontal.unsqueeze(1)
            propagated[:, :, :, 1:] += scores[:, :, :, :-1] * horizontal.unsqueeze(1)
        if vertical.numel():
            propagated[:, :, :-1] += scores[:, :, 1:] * vertical.unsqueeze(1)
            propagated[:, :, 1:] += scores[:, :, :-1] * vertical.unsqueeze(1)
        propagated = propagated / degree.unsqueeze(1)
        scores = restart * initial + (1.0 - restart) * propagated
        if seed_mask is not None and seed_one_hot is not None:
            scores = torch.where(seed_mask.unsqueeze(1), seed_one_hot, scores)
        scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return scores


def semantic_seed_labels(
    logits: Tensor,
    *,
    confidence_threshold: float = 0.65,
    logit_scale: float = 1.0,
    minimum_seeds_per_class: int = 1,
) -> Tensor:
    if logits.ndim != 3 or minimum_seeds_per_class < 0 or logit_scale <= 0:
        raise ValueError("logits must be CHW and minimum_seeds_per_class non-negative.")
    probabilities = torch.softmax(logits.float() * float(logit_scale), dim=0)
    confidence, labels = probabilities.max(dim=0)
    seeds = torch.full(labels.shape, -1, device=labels.device, dtype=torch.long)
    accepted = confidence >= confidence_threshold
    seeds[accepted] = labels[accepted]
    flat_probabilities = probabilities.flatten(1)
    for class_index in range(logits.shape[0]):
        existing = int((seeds == class_index).sum().item())
        needed = minimum_seeds_per_class - existing
        if needed <= 0:
            continue
        candidates = torch.topk(flat_probabilities[class_index], k=min(needed, labels.numel())).indices
        rows = torch.div(candidates, labels.shape[1], rounding_mode="floor")
        columns = candidates % labels.shape[1]
        seeds[rows, columns] = class_index
    return seeds


def gt_seed_labels(target: Tensor, *, fraction: float = 0.10, minimum_per_class: int = 1) -> Tensor:
    """Choose deterministic interior GT seeds for structure-only diagnostics."""

    if target.ndim != 2 or not 0 < fraction <= 1 or minimum_per_class < 1:
        raise ValueError("Invalid target or GT seed settings.")
    seeds = torch.full(target.shape, -1, device=target.device, dtype=torch.long)
    valid_classes = torch.unique(target[target >= 0]).tolist()
    for class_index in valid_classes:
        mask = target == int(class_index)
        count = int(mask.sum().item())
        if count == 0:
            continue
        neighbour_count = F.conv2d(
            mask.float()[None, None],
            torch.ones((1, 1, 3, 3), device=target.device),
            padding=1,
        )[0, 0]
        ranking = neighbour_count.masked_fill(~mask, -1).flatten()
        requested = max(minimum_per_class, int(round(count * fraction)))
        indices = torch.topk(ranking, k=min(requested, count)).indices
        rows = torch.div(indices, target.shape[1], rounding_mode="floor")
        columns = indices % target.shape[1]
        seeds[rows, columns] = int(class_index)
    return seeds


def binary_spatial_metrics(prediction: np.ndarray, target: np.ndarray, *, boundary_tolerance: int = 1) -> SpatialMetrics:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be same-shaped 2D masks.")
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    region_iou = intersection / union if union else 1.0
    predicted_pixels = int(prediction.sum())
    leakage = int(np.logical_and(prediction, ~target).sum()) / max(predicted_pixels, 1)
    return SpatialMetrics(
        region_iou=float(region_iou),
        boundary_f1=_boundary_f1(prediction, target, boundary_tolerance),
        connectivity_recall=_connectivity_recall(prediction, target),
        leakage_ratio=float(leakage),
    )


def oracle_pair_matrix(
    pair_scores: Mapping[tuple[int, int], float],
    layers: Sequence[int],
) -> dict[str, object]:
    ordered = tuple(int(layer) for layer in layers)
    expected = {(semantic, spatial) for semantic in ordered for spatial in ordered}
    missing = sorted(expected - set(pair_scores))
    if missing:
        raise ValueError(f"Missing semantic/spatial layer pairs: {missing}")
    matrix = [[float(pair_scores[(semantic, spatial)]) for spatial in ordered] for semantic in ordered]
    best_semantic, best_spatial = max(expected, key=lambda pair: pair_scores[pair])
    best_same = max(ordered, key=lambda layer: pair_scores[(layer, layer)])
    off_diagonal = [pair for pair in expected if pair[0] != pair[1]]
    best_off_diagonal = max(off_diagonal, key=lambda pair: pair_scores[pair]) if off_diagonal else None
    result = {
        "layers": list(ordered),
        "matrix": matrix,
        "best_pair": {
            "semantic_layer": best_semantic,
            "spatial_layer": best_spatial,
            "score": float(pair_scores[(best_semantic, best_spatial)]),
        },
        "best_same_depth": {
            "layer": best_same,
            "score": float(pair_scores[(best_same, best_same)]),
        },
        "dual_depth_oracle_gap": float(
            pair_scores[(best_semantic, best_spatial)] - pair_scores[(best_same, best_same)]
        ),
    }
    if best_off_diagonal is not None:
        result["best_off_diagonal_pair"] = {
            "semantic_layer": best_off_diagonal[0],
            "spatial_layer": best_off_diagonal[1],
            "score": float(pair_scores[best_off_diagonal]),
            "gap_over_best_same_depth": float(
                pair_scores[best_off_diagonal] - pair_scores[(best_same, best_same)]
            ),
        }
    return result


def _as_rows(features: Tensor, feature_dim: int) -> Tensor:
    if features.shape[-1] != feature_dim:
        raise ValueError("Feature dimension does not match the alignment statistics.")
    return features.reshape(-1, feature_dim)


def _boundary(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
        return np.logical_and(mask, eroded[0, 0].numpy() < 0.5)
    source = mask.astype(np.uint8)
    eroded = cv2.erode(source, np.ones((3, 3), dtype=np.uint8))
    return np.logical_and(mask, eroded == 0)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    if cv2 is None:
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        return F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0].numpy() > 0
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    pred_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0
    precision = np.logical_and(pred_boundary, _dilate(target_boundary, tolerance)).sum() / pred_count
    recall = np.logical_and(target_boundary, _dilate(pred_boundary, tolerance)).sum() / target_count
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def _connectivity_recall(prediction: np.ndarray, target: np.ndarray) -> float:
    if cv2 is None:
        return _connectivity_recall_python(prediction, target)
    target_count, target_ids = cv2.connectedComponents(target.astype(np.uint8), connectivity=4)
    if target_count <= 1:
        return 1.0 if not prediction.any() else 0.0
    _, predicted_ids = cv2.connectedComponents(prediction.astype(np.uint8), connectivity=4)
    weighted_score = 0.0
    for component_id in range(1, target_count):
        labels = predicted_ids[target_ids == component_id]
        labels = labels[labels > 0]
        if labels.size:
            counts = np.bincount(labels.astype(np.int64))
            weighted_score += int(counts.max())
    return float(weighted_score / max(int(target.sum()), 1))


def _connectivity_recall_python(prediction: np.ndarray, target: np.ndarray) -> float:
    target_components = _components_python(target)
    if not target_components:
        return 1.0 if not prediction.any() else 0.0
    predicted_components = _components_python(prediction)
    predicted_ids = np.full(prediction.shape, -1, dtype=np.int32)
    for component_id, component in enumerate(predicted_components):
        for row, column in component:
            predicted_ids[row, column] = component_id
    weighted_score = 0.0
    total = 0
    for component in target_components:
        labels = [predicted_ids[row, column] for row, column in component if predicted_ids[row, column] >= 0]
        if labels:
            counts = np.bincount(np.asarray(labels, dtype=np.int64))
            weighted_score += int(counts.max())
        total += len(component)
    return float(weighted_score / max(total, 1))


def _components_python(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                component.append((current_row, current_column))
                for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(component)
    return components
