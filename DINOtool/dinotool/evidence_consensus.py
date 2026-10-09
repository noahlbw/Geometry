"""Joint class-margin and spatial consensus solver for RER-OV."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contrastive_support import ContrastiveSupportResult


@dataclass(frozen=True)
class EvidenceConsensusConfig:
    anchor_weight: float = 1.0
    spatial_weight: float = 0.12
    evidence_weight: float = 0.35
    cg_iterations: int = 18
    cg_tolerance: float = 1e-5
    epsilon: float = 1e-7

    def validate(self) -> None:
        if self.anchor_weight <= 0 or self.spatial_weight < 0 or self.evidence_weight < 0:
            raise ValueError("Consensus weights must be non-negative and anchor_weight positive.")
        if self.cg_iterations < 1 or min(self.cg_tolerance, self.epsilon) <= 0:
            raise ValueError("CG iterations and tolerances must be positive.")


@dataclass(frozen=True)
class EvidenceConsensusDiagnostics:
    iterations: int
    relative_residual: float
    mean_update_norm: float
    changed_fraction: float
    evidence_coverage: float


@dataclass(frozen=True)
class EvidenceConsensusResult:
    scores: Tensor
    delta: Tensor
    diagnostics: EvidenceConsensusDiagnostics


@torch.inference_mode()
def solve_evidence_consensus(
    initial_scores: Tensor,
    support: ContrastiveSupportResult,
    valid: Tensor,
    config: EvidenceConsensusConfig = EvidenceConsensusConfig(),
    *,
    family_weights: Tensor | None = None,
    excluded_family: int | None = None,
) -> EvidenceConsensusResult:
    """Solve the fixed-coefficient SPD RER objective with matrix-free CG."""
    config.validate()
    scores = initial_scores.float()
    batch, patches, classes = scores.shape
    pairs = support.pair_ids
    if pairs.shape[:2] != (batch, patches) or pairs.shape[-1] != 2:
        raise ValueError("support pair_ids do not match initial_scores.")
    if valid.shape != (batch, patches):
        raise ValueError("valid must have shape [B,N].")
    family_count = support.family_count
    if family_weights is None:
        family_weights = torch.ones_like(support.availability)
    if family_weights.shape != support.availability.shape:
        raise ValueError("family_weights must match support availability [B,N,M,F].")
    weighted_availability = support.availability * family_weights.clamp_min(0.0)
    if excluded_family is not None:
        if excluded_family < 0 or excluded_family >= family_count:
            raise ValueError("excluded_family is out of range.")
        weighted_availability = weighted_availability.clone()
        weighted_availability[..., excluded_family] = 0.0
    pair_weight = weighted_availability.sum(-1)
    target = (
        weighted_availability * support.evidence
    ).sum(-1) / pair_weight.clamp_min(config.epsilon)
    target = torch.where(pair_weight > config.epsilon, target, torch.zeros_like(target))
    initial_margin = _pair_margin(scores, pairs)
    requested_delta = target - initial_margin

    adjacency = _symmetric_sparse_adjacency(
        support.candidate_indices, support.candidate_weights, valid, patches
    )
    degree = adjacency.sum(-1)
    rhs = config.evidence_weight * _pair_scatter(
        pair_weight * requested_delta, pairs, classes
    )
    rhs = rhs * valid[..., None]

    pair_diagonal = torch.zeros_like(scores)
    first, second = pairs[..., 0], pairs[..., 1]
    pair_diagonal.scatter_add_(-1, first, pair_weight)
    pair_diagonal.scatter_add_(-1, second, pair_weight)
    preconditioner = (
        config.anchor_weight
        + config.spatial_weight * degree[..., None]
        + config.evidence_weight * pair_diagonal
    ).clamp_min(config.epsilon)

    def matrix_vector(value: Tensor) -> Tensor:
        laplacian = degree[..., None] * value - adjacency @ value
        margins = _pair_margin(value, pairs)
        pair_term = _pair_scatter(pair_weight * margins, pairs, classes)
        output = (
            config.anchor_weight * value
            + config.spatial_weight * laplacian
            + config.evidence_weight * pair_term
        )
        return output * valid[..., None]

    delta = torch.zeros_like(scores)
    residual = rhs.clone()
    z = residual / preconditioner
    direction = z.clone()
    rz = _batch_dot(residual, z)
    rhs_norm = _batch_dot(rhs, rhs).sqrt().clamp_min(config.epsilon)
    iterations = 0
    relative = torch.ones(batch, device=scores.device)
    for iteration in range(config.cg_iterations):
        product = matrix_vector(direction)
        denominator = _batch_dot(direction, product).clamp_min(config.epsilon)
        alpha = rz / denominator
        delta = delta + alpha[:, None, None] * direction
        residual = residual - alpha[:, None, None] * product
        relative = _batch_dot(residual, residual).sqrt() / rhs_norm
        iterations = iteration + 1
        if bool((relative <= config.cg_tolerance).all()):
            break
        z = residual / preconditioner
        new_rz = _batch_dot(residual, z)
        beta = new_rz / rz.clamp_min(config.epsilon)
        direction = z + beta[:, None, None] * direction
        rz = new_rz

    delta = delta * valid[..., None]
    output = scores + delta
    old_label = scores.argmax(-1)
    new_label = output.argmax(-1)
    valid_count = max(int(valid.sum()), 1)
    evidence_patch = pair_weight.sum(-1) > config.epsilon
    diagnostics = EvidenceConsensusDiagnostics(
        iterations=iterations,
        relative_residual=float(relative.max().item()),
        mean_update_norm=float(delta[valid].norm(dim=-1).mean().item()) if bool(valid.any()) else 0.0,
        changed_fraction=float(((old_label != new_label) & valid).sum().item() / valid_count),
        evidence_coverage=float((evidence_patch & valid).sum().item() / valid_count),
    )
    return EvidenceConsensusResult(scores=output, delta=delta, diagnostics=diagnostics)


def _symmetric_sparse_adjacency(
    indices: Tensor,
    weights: Tensor,
    valid: Tensor,
    patches: int,
) -> Tensor:
    batch = indices.shape[0]
    directed = torch.zeros((batch, patches, patches), device=weights.device, dtype=torch.float32)
    directed.scatter_add_(-1, indices, weights.float())
    adjacency = 0.5 * (directed + directed.transpose(1, 2))
    mask = valid[:, :, None] & valid[:, None, :]
    adjacency = adjacency * mask
    diagonal = torch.eye(patches, device=weights.device, dtype=torch.bool)[None]
    return adjacency.masked_fill(diagonal, 0.0)


def _pair_margin(values: Tensor, pairs: Tensor) -> Tensor:
    first = values.gather(-1, pairs[..., 0])
    second = values.gather(-1, pairs[..., 1])
    return first - second


def _pair_scatter(margins: Tensor, pairs: Tensor, classes: int) -> Tensor:
    output = torch.zeros((*margins.shape[:2], classes), device=margins.device, dtype=margins.dtype)
    output.scatter_add_(-1, pairs[..., 0], margins)
    output.scatter_add_(-1, pairs[..., 1], -margins)
    return output


def _batch_dot(first: Tensor, second: Tensor) -> Tensor:
    return (first * second).flatten(1).sum(-1)
