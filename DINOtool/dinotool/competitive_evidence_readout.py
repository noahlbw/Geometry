"""Matched-control competitive evidence rereading for frozen DINO.text.

Descriptions never vote for a class directly. They only propose members of a
fixed Geometry neighbourhood. A proposal may change Geometry attention when
it reconstructs the query better than deterministic, budget-matched spatial
rolls of the same description responses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contrastive_support import ContrastiveSupportConfig, reconstruction_solution
from .evidence_vocabulary import (
    EncodedEvidenceBank,
    EvidenceVocabulary,
    encode_evidence_vocabulary,
)
from .tcpr import (
    TCPRConfig,
    TCPRPreparedImage,
    TCPRSegmenter,
    TCPRTextBank,
    _finish_attention_block,
    _finish_head,
    _geometry_attended,
    _normalize_valid_mask,
    _tokens_to_map,
    aggregate_alias_scores,
    filter_aliases,
)
from .text_visual_intervention import _text_conditioned_relation


@dataclass(frozen=True)
class CompetitiveEvidenceConfig:
    """Frozen hyperparameters for the first matched-control mechanism screen."""

    alias_temperature: float = 0.07
    class_temperature: float = 0.07
    semantic_relation_strength: float = 4.0
    candidate_members: int = 24
    response_temperature: float = 0.07
    direction_temperature: float = 0.08
    anchor_prior: float = 0.04
    gain_temperature: float = 0.02
    evidence_strength: float = 1.0
    maximum_attention_kl: float = 0.03
    attention_bisection_steps: int = 10
    query_chunk: int = 64
    null_roll_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    shuffled_roll_fraction: float = 0.5
    epsilon: float = 1e-6
    reconstruction: ContrastiveSupportConfig = ContrastiveSupportConfig(
        candidate_members=24,
        candidate_classes=2,
        query_chunk=64,
    )

    def validate(self) -> None:
        positive = (
            self.alias_temperature,
            self.class_temperature,
            self.semantic_relation_strength,
            self.candidate_members,
            self.response_temperature,
            self.direction_temperature,
            self.gain_temperature,
            self.maximum_attention_kl,
            self.attention_bisection_steps,
            self.query_chunk,
            self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("Competitive-evidence temperatures and counts must be positive.")
        if self.anchor_prior < 0 or self.evidence_strength < 0:
            raise ValueError("anchor_prior and evidence_strength must be non-negative.")
        if not self.null_roll_fractions:
            raise ValueError("At least one matched null roll is required.")
        fractions = (*self.null_roll_fractions, self.shuffled_roll_fraction)
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("Spatial roll fractions must be strictly between zero and one.")
        self.reconstruction.validate()


@dataclass(frozen=True)
class EvidenceArmDiagnostics:
    changed_from_s0: float
    evidence_coverage: float
    mean_positive_gain: float
    real_better_fraction: float
    mean_attention_kl: float
    maximum_attention_kl: float

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CompetitiveEvidenceResult:
    s0_logits: Tensor
    image_self_logits: Tensor
    matched_logits: Tensor
    shuffled_logits: Tensor
    matched_evidence_map: Tensor
    shuffled_evidence_map: Tensor
    matched_diagnostics: EvidenceArmDiagnostics
    shuffled_diagnostics: EvidenceArmDiagnostics


@dataclass(frozen=True)
class _InitialReadout:
    geometry: Tensor
    valid: Tensor
    visual: Tensor
    text: Tensor
    alias_weights: Tensor
    class_scores: Tensor
    s0_scores: Tensor
    class_text: Tensor
    description_responses: Tensor


class CompetitiveEvidenceSegmenter:
    """One-round matched-control rereading around the immutable TextGraph S0."""

    method_name = "competitive evidence rereading"

    def __init__(
        self,
        backbone,
        config: CompetitiveEvidenceConfig = CompetitiveEvidenceConfig(),
        tcpr_config: TCPRConfig = TCPRConfig(),
    ) -> None:
        config.validate()
        self.backbone = backbone
        self.config = config
        self.base = TCPRSegmenter(backbone, tcpr_config)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def prepare_image(self, rgb: Tensor) -> TCPRPreparedImage:
        return self.base.prepare_image(rgb)

    def encode_text(self, classes, batch_size: int = 64) -> TCPRTextBank:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_evidence(
        self,
        vocabulary: EvidenceVocabulary,
        class_names,
        *,
        batch_size: int = 64,
    ) -> EncodedEvidenceBank:
        return encode_evidence_vocabulary(
            self.backbone, vocabulary, class_names, batch_size=batch_size
        )

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        evidence_bank: EncodedEvidenceBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> CompetitiveEvidenceResult:
        text_bank.validate()
        evidence_bank.validate()
        if tuple(text_bank.class_names) != tuple(evidence_bank.class_names):
            raise ValueError("Recognition and evidence banks must use the same class order.")
        initial = self._initial_readout(prepared, text_bank, evidence_bank, valid_mask)
        matched, matched_map, matched_diag = self._read_arm(
            prepared, text_bank, evidence_bank, initial, primary_roll=0.0
        )
        shuffled, shuffled_map, shuffled_diag = self._read_arm(
            prepared,
            text_bank,
            evidence_bank,
            initial,
            primary_roll=self.config.shuffled_roll_fraction,
        )
        image_context = F.normalize(initial.geometry @ initial.visual, dim=-1)
        image_features = F.normalize(0.5 * initial.visual + 0.5 * image_context, dim=-1)
        image_alias = image_features @ initial.text.T
        image_scores = aggregate_alias_scores(
            image_alias,
            initial.alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        height, width = prepared.grid_height, prepared.grid_width
        return CompetitiveEvidenceResult(
            s0_logits=_tokens_to_map(initial.s0_scores, height, width),
            image_self_logits=_tokens_to_map(image_scores, height, width),
            matched_logits=_tokens_to_map(matched, height, width),
            shuffled_logits=_tokens_to_map(shuffled, height, width),
            matched_evidence_map=matched_map.reshape(matched_map.shape[0], height, width),
            shuffled_evidence_map=shuffled_map.reshape(shuffled_map.shape[0], height, width),
            matched_diagnostics=matched_diag,
            shuffled_diagnostics=shuffled_diag,
        )

    def _initial_readout(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        evidence_bank: EncodedEvidenceBank,
        valid_mask: Tensor | None,
    ) -> _InitialReadout:
        visual = F.normalize(prepared.geometry_projected.float(), dim=-1)
        batch, patches, _ = visual.shape
        valid = _normalize_valid_mask(
            valid_mask,
            batch,
            prepared.grid_height,
            prepared.grid_width,
            visual.device,
        ).reshape(batch, patches)
        text = F.normalize(text_bank.features.float(), dim=-1)
        geometry_alias = visual @ text.T
        native_alias = prepared.native_projected.float() @ text.T
        alias_weights = filter_aliases(
            geometry_alias,
            native_alias,
            text_bank.parent_indices,
            text_bank.canonical_mask,
            text_bank.class_count,
            valid,
            self.base.config,
        )
        class_scores = aggregate_alias_scores(
            geometry_alias,
            alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        geometry = _normalized_geometry(prepared.geometry_patch_conditional.float(), valid)
        probability = torch.softmax(class_scores / self.config.class_temperature, dim=-1)
        relation = _text_conditioned_relation(
            geometry, probability, valid, self.config.semantic_relation_strength
        )
        s0 = relation @ class_scores
        canonical = []
        for class_index in range(text_bank.class_count):
            member = (text_bank.parent_indices == class_index) & text_bank.canonical_mask
            canonical.append(text[member][0])
        class_text = torch.stack(canonical)
        evidence_text = F.normalize(evidence_bank.features.float(), dim=-1)
        responses = visual @ evidence_text.T
        centered = responses.clone()
        for image_index in range(batch):
            if bool(valid[image_index].any()):
                median = responses[image_index, valid[image_index]].median(dim=0).values
                centered[image_index] = responses[image_index] - median
        centered = (centered / self.config.response_temperature).clamp(-12.0, 12.0)
        return _InitialReadout(
            geometry=geometry,
            valid=valid,
            visual=visual,
            text=text,
            alias_weights=alias_weights,
            class_scores=class_scores,
            s0_scores=s0,
            class_text=class_text,
            description_responses=centered,
        )

    def _read_arm(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        evidence_bank: EncodedEvidenceBank,
        initial: _InitialReadout,
        *,
        primary_roll: float,
    ) -> tuple[Tensor, Tensor, EvidenceArmDiagnostics]:
        pairs = _ordered_competitors(initial.s0_scores)
        indices, candidate_weights = _geometry_candidates(
            initial.geometry,
            initial.valid,
            self.config.candidate_members,
            self.config.epsilon,
        )
        primary = _roll_valid(initial.description_responses, initial.valid, primary_roll)
        null_responses = tuple(
            _roll_valid(primary, initial.valid, fraction)
            for fraction in self.config.null_roll_fractions
        )
        edge_bias, evidence_map, positive_gain, comparisons, real_better = (
            self._competitive_edge_bias(
                prepared,
                evidence_bank,
                initial,
                pairs,
                indices,
                candidate_weights,
                primary,
                null_responses,
            )
        )
        calibrated_bias = calibrate_edge_bias(
            edge_bias, evidence_map, self.config.epsilon
        )
        modified, row_kl = redistribute_candidate_attention(
            initial.geometry,
            indices,
            calibrated_bias * self.config.evidence_strength,
            initial.valid,
            maximum_kl=self.config.maximum_attention_kl,
            bisection_steps=self.config.attention_bisection_steps,
            epsilon=self.config.epsilon,
        )
        active = evidence_map > self.config.epsilon
        if self.config.evidence_strength == 0 or not bool(active.any()):
            scores = initial.s0_scores
        else:
            scores = self._frozen_reread(prepared, text_bank, initial, modified)
        valid_count = max(int(initial.valid.sum()), 1)
        changed = (
            (scores.argmax(-1) != initial.s0_scores.argmax(-1)) & initial.valid
        ).sum().item() / valid_count
        positive_count = max(int((positive_gain > 0).sum()), 1)
        diagnostics = EvidenceArmDiagnostics(
            changed_from_s0=float(changed),
            evidence_coverage=float((active & initial.valid).sum().item() / valid_count),
            mean_positive_gain=float(positive_gain.sum().item() / positive_count),
            real_better_fraction=float(real_better / max(comparisons, 1)),
            mean_attention_kl=float(row_kl[initial.valid].mean().item()),
            maximum_attention_kl=float(row_kl.max().item()),
        )
        return scores, evidence_map, diagnostics

    def _competitive_edge_bias(
        self,
        prepared: TCPRPreparedImage,
        evidence_bank: EncodedEvidenceBank,
        initial: _InitialReadout,
        pairs: Tensor,
        indices: Tensor,
        candidate_weights: Tensor,
        primary_responses: Tensor,
        null_responses: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        raw = F.normalize(prepared.raw_patch_tokens.float(), dim=-1)
        evidence_text = F.normalize(evidence_bank.features.float(), dim=-1)
        class_description = initial.class_text @ evidence_text.T
        first, second = pairs[..., 0], pairs[..., 1]
        first_relation = class_description[first]
        second_relation = class_description[second]
        anchors = evidence_bank.anchor_indices
        anchor_direction = (
            (anchors[None, None, :] == first[..., None]).float()
            - (anchors[None, None, :] == second[..., None]).float()
        )
        direction = torch.tanh(
            (first_relation - second_relation + self.config.anchor_prior * anchor_direction)
            / self.config.direction_temperature
        )
        phrase_weights = direction.abs() * evidence_bank.priors[None, None]
        pair_scores = torch.stack(
            (
                initial.s0_scores.gather(-1, first[..., None]).squeeze(-1),
                initial.s0_scores.gather(-1, second[..., None]).squeeze(-1),
            ),
            dim=-1,
        )
        side_probability = torch.softmax(pair_scores / self.config.class_temperature, dim=-1)

        batch, patches, count = indices.shape
        bias = torch.zeros((batch, patches, count), device=raw.device)
        gain_map = torch.zeros((batch, patches), device=raw.device)
        active_families = torch.zeros_like(gain_map)
        all_positive_gain: list[Tensor] = []
        comparisons = 0
        real_better = 0
        batch_index = torch.arange(batch, device=raw.device)[:, None, None]

        for start in range(0, patches, self.config.query_chunk):
            stop = min(start + self.config.query_chunk, patches)
            selected = indices[:, start:stop]
            candidates = raw[batch_index, selected]
            query = raw[:, start:stop]
            local_geometry = candidate_weights[:, start:stop]
            primary = primary_responses[batch_index, selected]
            nulls = tuple(value[batch_index, selected] for value in null_responses)
            local_bias = torch.zeros_like(local_geometry)
            local_gain = torch.zeros((batch, stop - start), device=raw.device)
            local_active = torch.zeros_like(local_gain)

            for family in range(evidence_bank.family_count):
                members = evidence_bank.family_indices == family
                if not bool(members.any()):
                    continue
                family_direction = direction[:, start:stop, members]
                family_weight = phrase_weights[:, start:stop, members]
                family_bias = torch.zeros_like(local_geometry)
                family_gain = torch.zeros_like(local_gain)
                family_available = torch.zeros_like(local_gain, dtype=torch.bool)
                for side_index, positive in enumerate((True, False)):
                    oriented = family_direction if positive else -family_direction
                    real_prior, real_valid = _description_prior(
                        primary[..., members], oriented, family_weight,
                        local_geometry, self.config.epsilon,
                    )
                    null_priors = []
                    null_valid = []
                    for null in nulls:
                        prior, available = _description_prior(
                            null[..., members], oriented, family_weight,
                            local_geometry, self.config.epsilon,
                        )
                        null_priors.append(prior)
                        null_valid.append(available)
                    side_valid = real_valid & torch.stack(null_valid).all(0)
                    edge, gain, raw_gain = validated_prior_difference(
                        query,
                        candidates,
                        real_prior,
                        tuple(null_priors),
                        side_valid & initial.valid[:, start:stop],
                        self.config.reconstruction,
                        gain_temperature=self.config.gain_temperature,
                    )
                    weight = side_probability[:, start:stop, side_index]
                    family_bias += weight[..., None] * edge
                    family_gain += weight * gain
                    family_available |= gain > 0
                    finite = side_valid & torch.isfinite(raw_gain)
                    comparisons += int(finite.sum())
                    real_better += int((finite & (raw_gain > 0)).sum())
                    all_positive_gain.append(gain)
                local_bias += family_bias
                local_gain += family_gain
                local_active += family_available.float()

            denominator = local_active.clamp_min(1.0)
            bias[:, start:stop] = local_bias / denominator[..., None]
            gain_map[:, start:stop] = local_gain / denominator
            active_families[:, start:stop] = local_active

        bias = torch.where(active_families[..., None] > 0, bias, torch.zeros_like(bias))
        bias = bias.masked_fill(~initial.valid[..., None], 0.0)
        positive_gain = (
            torch.cat([item.reshape(-1) for item in all_positive_gain])
            if all_positive_gain else torch.zeros(1, device=raw.device)
        )
        return bias, gain_map, positive_gain, comparisons, real_better

    def _frozen_reread(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        initial: _InitialReadout,
        geometry: Tensor,
    ) -> Tensor:
        head = self.backbone.model.visual_model.head
        with self.backbone._autocast():
            attended = _geometry_attended(
                prepared.native_attention,
                geometry,
                prepared.value_tokens,
                prepared.prefix_tokens,
            )
            block = _finish_attention_block(
                head.blocks[prepared.block_index], prepared.shared_tokens, attended
            )
            projected = _finish_head(head, block, prepared.block_index)
        visual = F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1)
        alias_scores = visual @ initial.text.T
        class_scores = aggregate_alias_scores(
            alias_scores,
            initial.alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        probability = torch.softmax(class_scores / self.config.class_temperature, dim=-1)
        relation = _text_conditioned_relation(
            geometry, probability, initial.valid, self.config.semantic_relation_strength
        )
        output = relation @ class_scores
        return torch.where(initial.valid[..., None], output, initial.s0_scores)


def validated_prior_difference(
    query: Tensor,
    candidates: Tensor,
    real_prior: Tensor,
    null_priors: tuple[Tensor, ...],
    valid: Tensor,
    reconstruction_config: ContrastiveSupportConfig,
    *,
    gain_temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return bounded edge evidence only when real correspondence beats nulls."""
    if not null_priors:
        raise ValueError("validated_prior_difference requires at least one null prior.")
    real_coefficients, real_cost = reconstruction_solution(
        query, candidates, real_prior, reconstruction_config
    )
    null_solutions = [
        reconstruction_solution(query, candidates, prior, reconstruction_config)
        for prior in null_priors
    ]
    null_coefficients = torch.stack([item[0] for item in null_solutions]).mean(0)
    null_cost = torch.stack([item[1] for item in null_solutions]).median(0).values
    raw_gain = null_cost - real_cost
    gain = torch.tanh(F.relu(raw_gain) / gain_temperature)
    gain = torch.where(valid, gain, torch.zeros_like(gain))
    edge = gain[..., None] * (real_coefficients - null_coefficients)
    return edge, gain, raw_gain


def redistribute_candidate_attention(
    geometry: Tensor,
    indices: Tensor,
    edge_bias: Tensor,
    valid: Tensor,
    *,
    maximum_kl: float,
    bisection_steps: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Redistribute only candidate mass under a per-row KL trust region."""
    if indices.shape != edge_bias.shape:
        raise ValueError("indices and edge_bias must have identical [B,N,K] shape.")
    base = geometry.gather(-1, indices)
    mass = base.sum(-1, keepdim=True)
    conditional = base / mass.clamp_min(epsilon)
    active = valid & (mass.squeeze(-1) > epsilon) & (edge_bias.abs().amax(-1) > epsilon)
    lower = torch.zeros_like(mass)
    upper = torch.ones_like(mass)
    log_base = conditional.clamp_min(epsilon).log()
    for _ in range(bisection_steps):
        middle = 0.5 * (lower + upper)
        candidate = torch.softmax(log_base + middle * edge_bias, dim=-1)
        kl = _row_kl(candidate, conditional, epsilon)[..., None]
        upper = torch.where(kl > maximum_kl, middle, upper)
        lower = torch.where(kl > maximum_kl, lower, middle)
    proposed = torch.softmax(log_base + lower * edge_bias, dim=-1)
    proposed_kl = _row_kl(proposed, conditional, epsilon)
    replacement = proposed * mass
    output = geometry.clone()
    current = output.gather(-1, indices)
    replacement = torch.where(active[..., None], replacement, current)
    output.scatter_(-1, indices, replacement)
    row_kl = torch.where(active, proposed_kl, torch.zeros_like(proposed_kl))
    return output, row_kl


def calibrate_edge_bias(edge_bias: Tensor, gain: Tensor, epsilon: float) -> Tensor:
    """Separate an edge direction from its bounded validation strength."""
    centered = edge_bias - edge_bias.mean(-1, keepdim=True)
    scale = centered.square().mean(-1, keepdim=True).sqrt()
    direction = centered / scale.clamp_min(epsilon)
    active = scale > epsilon
    calibrated = direction * gain.clamp(0.0, 1.0)[..., None]
    return torch.where(active, calibrated, torch.zeros_like(calibrated))


def _row_kl(first: Tensor, second: Tensor, epsilon: float) -> Tensor:
    return (
        first * (first.clamp_min(epsilon).log() - second.clamp_min(epsilon).log())
    ).sum(-1)


def _description_prior(
    response: Tensor,
    oriented_direction: Tensor,
    phrase_weight: Tensor,
    geometry: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    weight = phrase_weight * F.relu(oriented_direction)
    logits = response + weight.clamp_min(epsilon).log()[:, :, None, :]
    logits = logits.masked_fill(weight[:, :, None, :] <= 0, -torch.inf)
    description_score = torch.logsumexp(logits, dim=-1)
    prior_logits = geometry.clamp_min(epsilon).log() + description_score
    available = torch.isfinite(prior_logits).any(-1) & (weight.sum(-1) > epsilon)
    prior = torch.softmax(prior_logits, dim=-1)
    prior = torch.where(available[..., None], prior, geometry)
    prior = prior / prior.sum(-1, keepdim=True).clamp_min(epsilon)
    return prior, available


def _ordered_competitors(scores: Tensor) -> Tensor:
    count = min(2, scores.shape[-1])
    classes = torch.topk(scores, count, dim=-1).indices
    if count == 1:
        classes = torch.cat((classes, classes), dim=-1)
    return classes


def _geometry_candidates(
    geometry: Tensor,
    valid: Tensor,
    maximum: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    _, patches, _ = geometry.shape
    logits = geometry.clamp_min(epsilon).log()
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    if patches > 1:
        diagonal = torch.eye(patches, device=geometry.device, dtype=torch.bool)[None]
        logits = logits.masked_fill(diagonal, -torch.inf)
    count = min(maximum, patches - 1 if patches > 1 else 1)
    values, indices = torch.topk(logits, count, dim=-1)
    weights = torch.softmax(values, dim=-1)
    weights = torch.where(torch.isfinite(values), weights, torch.zeros_like(weights))
    weights = weights * valid[..., None]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(epsilon)
    return indices, weights


def _roll_valid(values: Tensor, valid: Tensor, fraction: float) -> Tensor:
    if fraction == 0.0:
        return values
    output = values.clone()
    for image_index in range(values.shape[0]):
        positions = valid[image_index].nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        shift = max(int(round(float(positions.numel()) * fraction)), 1)
        shift %= int(positions.numel())
        if shift:
            output[image_index, positions] = values[image_index, positions].roll(shift, dims=0)
    return output


def _normalized_geometry(geometry: Tensor, valid: Tensor) -> Tensor:
    masked = geometry.masked_fill(~valid[:, None, :], 0.0)
    return masked / masked.sum(-1, keepdim=True).clamp_min(1e-8)
