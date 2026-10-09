"""Text-proposed, visually verified support reconstruction for RER-OV."""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch
import torch.nn.functional as F
from torch import Tensor

from .evidence_vocabulary import EncodedEvidenceBank


@dataclass(frozen=True)
class ContrastiveSupportConfig:
    candidate_members: int = 24
    candidate_classes: int = 3
    direction_temperature: float = 0.08
    anchor_prior: float = 0.04
    response_temperature: float = 0.07
    reconstruction_kl: float = 0.05
    reconstruction_steps: int = 5
    reconstruction_step_size: float = 0.20
    evidence_temperature: float = 0.08
    separation_scale: float = 0.30
    query_chunk: int = 64
    epsilon: float = 1e-6

    def validate(self) -> None:
        positive = (
            self.candidate_members, self.candidate_classes, self.direction_temperature,
            self.response_temperature, self.reconstruction_kl, self.reconstruction_steps,
            self.reconstruction_step_size, self.evidence_temperature, self.separation_scale,
            self.query_chunk, self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("Contrastive support counts and scales must be positive.")
        if self.anchor_prior < 0:
            raise ValueError("anchor_prior must be non-negative.")


@dataclass(frozen=True)
class ContrastiveSupportResult:
    pair_ids: Tensor
    evidence: Tensor
    availability: Tensor
    phrase_support: Tensor
    phrase_directions: Tensor
    candidate_indices: Tensor
    candidate_weights: Tensor

    @property
    def family_count(self) -> int:
        return self.evidence.shape[-1]


def select_candidate_pairs(scores: Tensor, maximum_classes: int) -> Tensor:
    """Return globally oriented class pairs from each patch's leading classes."""
    if scores.ndim != 3 or scores.shape[-1] < 2:
        raise ValueError("scores must have shape [B,N,C] with at least two classes.")
    count = min(maximum_classes, scores.shape[-1])
    classes = torch.topk(scores, count, dim=-1).indices
    combinations = list(itertools.combinations(range(count), 2))
    pairs = []
    for first, second in combinations:
        a = torch.minimum(classes[..., first], classes[..., second])
        b = torch.maximum(classes[..., first], classes[..., second])
        pairs.append(torch.stack((a, b), dim=-1))
    return torch.stack(pairs, dim=-2)


def sparse_geometry_candidates(
    geometry: Tensor,
    valid: Tensor,
    maximum: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    if geometry.ndim != 3 or geometry.shape[1] != geometry.shape[2]:
        raise ValueError("geometry must have shape [B,N,N].")
    batch, patches, _ = geometry.shape
    if valid.shape != (batch, patches):
        raise ValueError("valid must have shape [B,N].")
    logits = geometry.float().clamp_min(epsilon).log()
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    diagonal = torch.eye(patches, device=geometry.device, dtype=torch.bool)[None]
    logits = logits.masked_fill(diagonal, -torch.inf)
    count = min(maximum, max(patches - 1, 1))
    values, indices = torch.topk(logits, count, dim=-1)
    weights = torch.softmax(values, dim=-1)
    weights = torch.where(torch.isfinite(values), weights, torch.zeros_like(weights))
    weights = weights * valid[..., None]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(epsilon)
    return indices, weights


@torch.inference_mode()
def build_contrastive_support(
    raw_features: Tensor,
    aligned_features: Tensor,
    geometry: Tensor,
    class_text: Tensor,
    class_scores: Tensor,
    evidence_bank: EncodedEvidenceBank,
    valid: Tensor,
    config: ContrastiveSupportConfig = ContrastiveSupportConfig(),
    *,
    phrase_weights: Tensor | None = None,
    pair_ids: Tensor | None = None,
) -> ContrastiveSupportResult:
    """Build per-family A/B evidence without multiplying raw DINO features by text."""
    config.validate()
    evidence_bank.validate()
    raw = F.normalize(raw_features.float(), dim=-1)
    aligned = F.normalize(aligned_features.float(), dim=-1)
    text = F.normalize(evidence_bank.features.float(), dim=-1)
    classes = F.normalize(class_text.float(), dim=-1)
    batch, patches, _ = raw.shape
    if aligned.shape[:2] != (batch, patches) or class_scores.shape[:2] != (batch, patches):
        raise ValueError("Image and score tensors must share [B,N].")
    if valid.shape != (batch, patches):
        raise ValueError("valid must have shape [B,N].")
    if class_scores.shape[-1] != classes.shape[0]:
        raise ValueError("class_text does not match class_scores.")

    pairs = select_candidate_pairs(class_scores, config.candidate_classes) if pair_ids is None else pair_ids
    if pairs.ndim != 4 or pairs.shape[:2] != (batch, patches) or pairs.shape[-1] != 2:
        raise ValueError("pair_ids must have shape [B,N,M,2].")
    pair_count = pairs.shape[-2]
    phrase_count = text.shape[0]
    family_count = evidence_bank.family_count
    indices, geometry_weights = sparse_geometry_candidates(
        geometry, valid, config.candidate_members, config.epsilon
    )

    image_text = aligned @ text.T
    centered = image_text.clone()
    for image_index in range(batch):
        if bool(valid[image_index].any()):
            median = image_text[image_index, valid[image_index]].median(dim=0).values
            centered[image_index] = image_text[image_index] - median
    responses = (centered / config.response_temperature).clamp(-12.0, 12.0)

    class_description = classes @ text.T
    first, second = pairs[..., 0], pairs[..., 1]
    first_relation = class_description[first]
    second_relation = class_description[second]
    anchors = evidence_bank.anchor_indices
    anchor_direction = (
        (anchors[None, None, None, :] == first[..., None]).float()
        - (anchors[None, None, None, :] == second[..., None]).float()
    )
    direction = torch.tanh(
        (first_relation - second_relation + config.anchor_prior * anchor_direction)
        / config.direction_temperature
    )
    base_phrase_weight = direction.abs() * evidence_bank.priors[None, None, None]
    if phrase_weights is not None:
        if phrase_weights.shape != (batch, patches, pair_count, phrase_count):
            raise ValueError("phrase_weights must have shape [B,N,M,R].")
        base_phrase_weight = base_phrase_weight * phrase_weights.clamp_min(0.0)

    evidence = torch.zeros(
        (batch, patches, pair_count, family_count), device=raw.device, dtype=torch.float32
    )
    availability = torch.zeros_like(evidence)
    phrase_support = torch.zeros((batch, patches, phrase_count), device=raw.device)
    b_index = torch.arange(batch, device=raw.device)[:, None, None]

    for start in range(0, patches, config.query_chunk):
        stop = min(start + config.query_chunk, patches)
        selected = indices[:, start:stop]
        candidate_raw = raw[b_index, selected]
        candidate_response = responses[b_index, selected]
        query_raw = raw[:, start:stop]
        local_geometry = geometry_weights[:, start:stop]
        phrase_support[:, start:stop] = (
            local_geometry[..., None] * F.softplus(candidate_response)
        ).sum(dim=-2)

        for family in range(family_count):
            members = evidence_bank.family_indices == family
            if not bool(members.any()):
                continue
            local_response = candidate_response[..., members]
            local_direction = direction[:, start:stop, :, members]
            local_weight = base_phrase_weight[:, start:stop, :, members]
            plus_prior, plus_valid = _side_prior(
                local_response, local_direction, local_weight, local_geometry,
                positive=True, epsilon=config.epsilon,
            )
            minus_prior, minus_valid = _side_prior(
                local_response, local_direction, local_weight, local_geometry,
                positive=False, epsilon=config.epsilon,
            )
            plus_cost = torch.zeros_like(plus_valid, dtype=torch.float32)
            minus_cost = torch.zeros_like(minus_valid, dtype=torch.float32)
            for pair_index in range(pair_count):
                plus_cost[..., pair_index] = _reconstruction_cost(
                    query_raw, candidate_raw, plus_prior[..., pair_index, :], config
                )
                minus_cost[..., pair_index] = _reconstruction_cost(
                    query_raw, candidate_raw, minus_prior[..., pair_index, :], config
                )
            difference = (minus_cost - plus_cost) / config.evidence_temperature
            difference = difference.clamp(-6.0, 6.0)
            separation = 0.5 * (plus_prior - minus_prior).abs().sum(-1)
            confidence = torch.tanh(difference.abs())
            available = (
                plus_valid & minus_valid & valid[:, start:stop, None]
                & (first[:, start:stop] != second[:, start:stop])
            )
            reliability = (separation / config.separation_scale).clamp(0.0, 1.0) * confidence
            evidence[:, start:stop, :, family] = torch.where(
                available, difference, torch.zeros_like(difference)
            )
            availability[:, start:stop, :, family] = torch.where(
                available, reliability, torch.zeros_like(reliability)
            )

    return ContrastiveSupportResult(
        pair_ids=pairs,
        evidence=evidence,
        availability=availability,
        phrase_support=phrase_support,
        phrase_directions=direction,
        candidate_indices=indices,
        candidate_weights=geometry_weights,
    )


def _side_prior(
    response: Tensor,
    direction: Tensor,
    weights: Tensor,
    geometry: Tensor,
    *,
    positive: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    oriented = direction if positive else -direction
    phrase_weight = weights * F.relu(oriented)
    mass = phrase_weight.sum(-1)
    logits = response[:, :, None] + phrase_weight.clamp_min(epsilon).log()[:, :, :, None]
    logits = logits.masked_fill(phrase_weight[:, :, :, None] <= 0, -torch.inf)
    description_score = torch.logsumexp(logits, dim=-1)
    prior_logits = geometry.clamp_min(epsilon).log()[:, :, None] + description_score
    prior = torch.softmax(prior_logits, dim=-1)
    valid = torch.isfinite(prior_logits).any(-1) & (mass > epsilon)
    fallback = geometry[:, :, None].expand_as(prior)
    prior = torch.where(valid[..., None], prior, fallback)
    prior = prior / prior.sum(-1, keepdim=True).clamp_min(epsilon)
    return prior, valid


def _reconstruction_cost(
    query: Tensor,
    candidates: Tensor,
    prior: Tensor,
    config: ContrastiveSupportConfig,
) -> Tensor:
    _, cost = reconstruction_solution(query, candidates, prior, config)
    return cost


def reconstruction_solution(
    query: Tensor,
    candidates: Tensor,
    prior: Tensor,
    config: ContrastiveSupportConfig,
) -> tuple[Tensor, Tensor]:
    """Return simplex coefficients and their regularized reconstruction cost."""
    coefficients = prior.clamp_min(config.epsilon)
    coefficients = coefficients / coefficients.sum(-1, keepdim=True)
    for _ in range(config.reconstruction_steps):
        reconstruction = torch.einsum("bqk,bqkd->bqd", coefficients, candidates)
        residual = reconstruction - query
        gradient = 2.0 * torch.einsum("bqd,bqkd->bqk", residual, candidates)
        gradient = gradient + config.reconstruction_kl * (
            (coefficients.clamp_min(config.epsilon) / prior.clamp_min(config.epsilon)).log() + 1.0
        )
        coefficients = coefficients * torch.exp(
            (-config.reconstruction_step_size * gradient).clamp(-12.0, 12.0)
        )
        coefficients = coefficients / coefficients.sum(-1, keepdim=True).clamp_min(config.epsilon)
    reconstruction = torch.einsum("bqk,bqkd->bqd", coefficients, candidates)
    reconstruction_error = (reconstruction - query).square().sum(-1)
    kl = coefficients * (
        coefficients.clamp_min(config.epsilon).log() - prior.clamp_min(config.epsilon).log()
    )
    cost = reconstruction_error + config.reconstruction_kl * kl.sum(-1)
    return coefficients, cost
