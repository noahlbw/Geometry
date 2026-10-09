"""Training-free discriminative evidence propagation over a shared geometry graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .hypothesis_readout import preserve_residual_partition, uniform_subset_scores
from .tcpr import (
    TCPRConfig,
    TCPRPreparedImage,
    TCPRSegmenter,
    TCPRTextBank,
    _finish_attention_block,
    _finish_head,
    _normalize_valid_mask,
    _tokens_to_map,
)


@dataclass(frozen=True)
class DGERConfig:
    consensus_correction: bool = False
    alias_temperature: float = 0.07
    alias_distinctiveness_temperature: float = 0.05
    canonical_alias_prior: float = 0.20
    self_weight: float = 0.75
    native_weight: float = 0.25
    evidence_temperature: float = 0.04
    disagreement_floor: float = 0.20
    evidence_topk: int = 32
    laplacian_strength: float = 1.50
    text_separation_scale: float = 0.25
    correction_strength: float = 1.00
    correction_clip: float = 0.10
    epsilon: float = 1e-6
    residual_class_names: tuple[str, ...] = ("background",)

    def validate(self) -> None:
        positive = (
            self.alias_temperature,
            self.alias_distinctiveness_temperature,
            self.evidence_temperature,
            self.text_separation_scale,
            self.correction_clip,
            self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("DGER temperatures, scales, clipping, and epsilon must be positive.")
        if self.evidence_topk < 1:
            raise ValueError("evidence_topk must be positive.")
        bounded = (self.canonical_alias_prior, self.self_weight, self.native_weight,
                   self.disagreement_floor)
        if min(bounded) < 0 or max(bounded) > 1:
            raise ValueError("DGER mixture values must be in [0, 1].")
        if abs(self.self_weight + self.native_weight - 1.0) > 1e-6:
            raise ValueError("self_weight and native_weight must sum to one.")
        if self.laplacian_strength < 0 or self.correction_strength < 0:
            raise ValueError("DGER propagation and correction strengths must be non-negative.")


@dataclass(frozen=True)
class DGERDiagnostics:
    mean_text_separation: float
    mean_local_reliability: float
    mean_neighbor_coverage: float
    mean_neighbor_consensus: float
    mean_gate: float
    mean_absolute_correction: float
    contradiction_fraction: float
    residual_reversion_fraction: float
    changed_from_geometry: float
    geometry_self_disagreement: float
    geometry_native_disagreement: float
    local_agreement_fraction: float = 0.0
    neighborhood_agreement_fraction: float = 0.0
    eligible_fraction: float = 0.0

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DGERResult:
    geometry_logits: Tensor
    logits: Tensor
    self_logits: Tensor
    native_logits: Tensor
    pair_ids: Tensor
    propagated_evidence: Tensor
    correction_gate: Tensor
    diagnostics: DGERDiagnostics
    legacy_logits: Tensor | None = None


class DGERSegmenter:
    """Geometry readout corrected only by discriminative local evidence consensus."""

    method_name = "DGER: discriminative geometry evidence rereading"

    def __init__(
        self,
        backbone,
        config: DGERConfig = DGERConfig(),
        tcpr_config: TCPRConfig = TCPRConfig(maximum_aliases_per_class=20),
    ) -> None:
        config.validate()
        self.backbone = backbone
        self.config = config
        self.base = TCPRSegmenter(backbone, tcpr_config)
        self._pair_text_cache: dict[int, tuple[TCPRTextBank, Tensor, Tensor]] = {}

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

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> DGERResult:
        text_bank.validate()
        if text_bank.class_count < 2:
            raise ValueError("DGER requires at least two competing classes.")
        batch, patches, _ = prepared.geometry_projected.shape
        valid = _normalize_valid_mask(
            valid_mask,
            batch,
            prepared.grid_height,
            prepared.grid_width,
            prepared.geometry_projected.device,
        ).reshape(batch, patches)
        text = F.normalize(text_bank.features.float(), dim=-1)
        all_aliases = torch.ones(len(text_bank.alias_names), device=text.device, dtype=torch.bool)

        geometry_features = F.normalize(prepared.geometry_projected.float(), dim=-1)
        native_features = F.normalize(prepared.native_projected.float(), dim=-1)
        self_features = self._self_projected(prepared)
        geometry_scores = _uniform_class_scores(
            geometry_features, text, text_bank, all_aliases, self.config.alias_temperature
        )
        native_scores = _uniform_class_scores(
            native_features, text, text_bank, all_aliases, self.config.alias_temperature
        )
        self_scores = _uniform_class_scores(
            self_features, text, text_bank, all_aliases, self.config.alias_temperature
        )

        cached = self._pair_text_cache.get(id(text_bank))
        if cached is None:
            alias_weights, text_separation = pairwise_alias_weights(text, text_bank, self.config)
            self._pair_text_cache[id(text_bank)] = (text_bank, alias_weights, text_separation)
        else:
            _, alias_weights, text_separation = cached
        self_pair_scores = pairwise_alias_scores(self_features @ text.T, alias_weights)
        native_pair_scores = pairwise_alias_scores(native_features @ text.T, alias_weights)
        corrected, pair_ids, propagated, gate, details = discriminative_geometry_correction(
            geometry_scores,
            self_pair_scores,
            native_pair_scores,
            prepared.geometry_patch_conditional.float(),
            text_separation,
            valid,
            self.config,
        )
        correction = preserve_residual_partition(
            corrected - geometry_scores,
            geometry_scores,
            text_bank.class_names,
            self.config.residual_class_names,
        )
        corrected = torch.where(valid[..., None], geometry_scores + correction, geometry_scores)
        corrected, residual_reversion = enforce_residual_partition(
            corrected,
            geometry_scores,
            text_bank.class_names,
            self.config.residual_class_names,
        )
        legacy = corrected
        if self.config.consensus_correction:
            corrected, pair_ids, propagated, gate, details = consensus_geometry_correction(
                geometry_scores, self_scores, native_scores,
                self_pair_scores, native_pair_scores,
                prepared.geometry_patch_conditional.float(), valid,
                text_bank.class_names, self.config,
            )
            corrected, residual_reversion = enforce_residual_partition(
                corrected, geometry_scores, text_bank.class_names,
                self.config.residual_class_names,
            )

        geometry_label = geometry_scores.argmax(-1)
        final_label = corrected.argmax(-1)
        self_label = self_scores.argmax(-1)
        native_label = native_scores.argmax(-1)
        count = max(int(valid.sum()), 1)
        diagnostics = DGERDiagnostics(
            mean_text_separation=_valid_mean(details["text_separation"], valid),
            mean_local_reliability=_valid_mean(details["local_reliability"], valid),
            mean_neighbor_coverage=_valid_mean(details["coverage"], valid),
            mean_neighbor_consensus=_valid_mean(details["consensus"], valid),
            mean_gate=_valid_mean(gate, valid),
            mean_absolute_correction=_valid_mean(details["absolute_correction"], valid),
            contradiction_fraction=float((details["contradiction"] & valid).sum().item() / count),
            residual_reversion_fraction=float((residual_reversion & valid).sum().item() / count),
            changed_from_geometry=float(((geometry_label != final_label) & valid).sum().item() / count),
            geometry_self_disagreement=float(((geometry_label != self_label) & valid).sum().item() / count),
            geometry_native_disagreement=float(((geometry_label != native_label) & valid).sum().item() / count),
            local_agreement_fraction=_valid_mean(details.get("local_agreement", gate * 0).float(), valid),
            neighborhood_agreement_fraction=_valid_mean(details.get("neighborhood_agreement", gate * 0).float(), valid),
            eligible_fraction=_valid_mean(details.get("eligible", gate * 0).float(), valid),
        )
        height, width = prepared.grid_height, prepared.grid_width
        return DGERResult(
            geometry_logits=_tokens_to_map(geometry_scores, height, width),
            logits=_tokens_to_map(corrected, height, width),
            self_logits=_tokens_to_map(self_scores, height, width),
            native_logits=_tokens_to_map(native_scores, height, width),
            pair_ids=pair_ids.transpose(1, 2).reshape(batch, 2, height, width),
            propagated_evidence=propagated.reshape(batch, height, width),
            correction_gate=gate.reshape(batch, height, width),
            diagnostics=diagnostics,
            legacy_logits=_tokens_to_map(legacy, height, width) if self.config.consensus_correction else None,
        )

    @torch.inference_mode()
    def _self_projected(self, prepared: TCPRPreparedImage) -> Tensor:
        attended = self_attended(
            prepared.native_attention,
            prepared.value_tokens,
            prepared.prefix_tokens,
        )
        head = self.backbone.model.visual_model.head
        with self.backbone._autocast():
            current = _finish_attention_block(
                head.blocks[prepared.block_index], prepared.shared_tokens, attended
            )
            projected = _finish_head(head, current, prepared.block_index)
        return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1)


def pairwise_alias_weights(
    text: Tensor,
    bank: TCPRTextBank,
    config: DGERConfig,
) -> tuple[Tensor, Tensor]:
    """Weight aliases by how specifically they separate one class from a competitor."""
    classes, aliases = bank.class_count, text.shape[0]
    canonical = torch.stack([
        text[(bank.parent_indices == class_index) & bank.canonical_mask][0]
        for class_index in range(classes)
    ])
    similarity = text @ canonical.T
    weights = torch.zeros((classes, classes, aliases), device=text.device, dtype=text.dtype)
    separation = torch.zeros((classes, classes), device=text.device, dtype=text.dtype)
    for positive in range(classes):
        members = bank.parent_indices == positive
        for competitor in range(classes):
            if positive == competitor:
                continue
            distinctiveness = similarity[members, positive] - similarity[members, competitor]
            semantic = torch.softmax(
                distinctiveness / config.alias_distinctiveness_temperature, dim=0
            )
            canonical_member = bank.canonical_mask[members].to(text.dtype)
            mixed = (1.0 - config.canonical_alias_prior) * semantic
            mixed = mixed + config.canonical_alias_prior * canonical_member
            weights[positive, competitor, members] = mixed / mixed.sum().clamp_min(config.epsilon)
            separation[positive, competitor] = (
                weights[positive, competitor, members] * distinctiveness.clamp_min(0.0)
            ).sum()
    symmetric = 0.5 * (separation + separation.T)
    return weights, symmetric


def pairwise_alias_scores(alias_scores: Tensor, weights: Tensor) -> Tensor:
    if alias_scores.ndim != 3 or weights.ndim != 3:
        raise ValueError("Expected alias_scores [B,N,M] and weights [C,C,M].")
    if alias_scores.shape[-1] != weights.shape[-1]:
        raise ValueError("Alias score and weight counts do not match.")
    return torch.einsum("bnm,cdm->bncd", alias_scores, weights)


def discriminative_geometry_correction(
    base_scores: Tensor,
    self_pair_scores: Tensor,
    native_pair_scores: Tensor,
    geometry: Tensor,
    text_separation: Tensor,
    valid: Tensor,
    config: DGERConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Propagate pair-specific evidence on one shared graph and conserve class mass."""
    batch, patches, classes = base_scores.shape
    expected = (batch, patches, classes, classes)
    if self_pair_scores.shape != expected or native_pair_scores.shape != expected:
        raise ValueError("Pairwise scores must have shape [B,N,C,C].")
    if geometry.shape != (batch, patches, patches) or valid.shape != (batch, patches):
        raise ValueError("Geometry or valid mask has incompatible shape.")
    top = torch.topk(base_scores, k=2, dim=-1)
    first, second = top.indices[..., 0], top.indices[..., 1]
    pair_ids = torch.stack((first, second), dim=-1)

    self_margin = _query_pair_key_margin(self_pair_scores, first, second)
    native_margin = _query_pair_key_margin(native_pair_scores, first, second)
    local_margin = config.self_weight * self_margin + config.native_weight * native_margin
    self_strength = torch.tanh(self_margin.abs() / config.evidence_temperature)
    native_strength = torch.tanh(native_margin.abs() / config.evidence_temperature)
    view_agreement = torch.where(
        self_margin * native_margin >= 0,
        torch.ones_like(local_margin),
        torch.full_like(local_margin, config.disagreement_floor),
    )
    local_reliability = torch.sqrt((self_strength * native_strength).clamp_min(0.0))
    local_reliability = local_reliability * view_agreement

    shared = geometry.masked_fill(~valid[:, None, :], 0.0)
    if patches > 1:
        diagonal = torch.eye(patches, device=geometry.device, dtype=torch.bool)[None]
        shared = shared.masked_fill(diagonal, 0.0)
    count = min(config.evidence_topk, max(patches - 1, 1))
    affinity, indices = torch.topk(shared, count, dim=-1)
    key_margin = local_margin.gather(-1, indices)
    key_reliability = local_reliability.gather(-1, indices)
    edge_weight = affinity * key_reliability
    denominator = edge_weight.sum(-1).clamp_min(config.epsilon)
    neighbor_margin = (edge_weight * key_margin).sum(-1) / denominator
    coverage = edge_weight.sum(-1) / affinity.sum(-1).clamp_min(config.epsilon)
    vote = (edge_weight * torch.tanh(key_margin / config.evidence_temperature)).sum(-1)
    consensus = (vote / denominator).abs().clamp(0.0, 1.0)

    local_anchor = config.self_weight * _pair_diagonal(self_pair_scores, first, second)
    local_anchor = local_anchor + config.native_weight * _pair_diagonal(
        native_pair_scores, first, second
    )
    propagated = (
        local_anchor + config.laplacian_strength * neighbor_margin
    ) / (1.0 + config.laplacian_strength)
    base_first = base_scores.gather(-1, first[..., None]).squeeze(-1)
    base_second = base_scores.gather(-1, second[..., None]).squeeze(-1)
    base_margin = base_first - base_second
    pair_separation = text_separation[first, second]
    text_quality = (pair_separation / config.text_separation_scale).clamp(0.0, 1.0)
    anchor_self = _pair_diagonal(self_pair_scores, first, second)
    anchor_native = _pair_diagonal(native_pair_scores, first, second)
    anchor_reliability = torch.sqrt(
        (torch.tanh(anchor_self.abs() / config.evidence_temperature)
         * torch.tanh(anchor_native.abs() / config.evidence_temperature)).clamp_min(0.0)
    )
    gate = text_quality * torch.sqrt((coverage * consensus).clamp_min(0.0))
    gate = gate * (0.5 + 0.5 * anchor_reliability)
    contradiction = propagated < 0
    gate = gate * contradiction * valid
    target_delta = 0.5 * config.correction_strength * gate * (propagated - base_margin)
    delta = config.correction_clip * torch.tanh(target_delta / config.correction_clip)

    correction = torch.zeros_like(base_scores)
    correction.scatter_add_(-1, first[..., None], delta[..., None])
    correction.scatter_add_(-1, second[..., None], -delta[..., None])
    details = {
        "text_separation": pair_separation,
        "local_reliability": local_reliability.mean(-1),
        "coverage": coverage,
        "consensus": consensus,
        "absolute_correction": delta.abs(),
        "contradiction": contradiction,
    }
    return base_scores + correction, pair_ids, propagated, gate, details


def consensus_geometry_correction(
    base_scores: Tensor,
    self_scores: Tensor,
    native_scores: Tensor,
    self_pair_scores: Tensor,
    native_pair_scores: Tensor,
    geometry: Tensor,
    valid: Tensor,
    class_names: tuple[str, ...],
    config: DGERConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Replace a pair margin only when every view contradicts it more strongly.

    All neighbors use the same Geometry support, including contrary evidence.
    Preserve the pair's log-partition instead of adding a shift to other classes.
    """
    batch, patches, classes = base_scores.shape
    if self_scores.shape != base_scores.shape or native_scores.shape != base_scores.shape:
        raise ValueError("Local class scores must match base scores.")
    if self_pair_scores.shape != (batch, patches, classes, classes) or native_pair_scores.shape != self_pair_scores.shape:
        raise ValueError("Pair scores must have shape [B,N,C,C].")
    if geometry.shape != (batch, patches, patches) or valid.shape != (batch, patches):
        raise ValueError("Geometry or valid mask has incompatible shape.")
    if len(class_names) != classes or classes < 2:
        raise ValueError("At least two matching class names are required.")
    top = torch.topk(base_scores, 2, dim=-1)
    first, second = top.indices.unbind(-1)
    base_margin = top.values[..., 0] - top.values[..., 1]
    b = torch.arange(batch, device=base_scores.device)[:, None, None]

    shared = geometry.masked_fill(~valid[:, None, :], 0.0)
    diagonal = torch.eye(patches, device=geometry.device, dtype=torch.bool)[None]
    shared = shared.masked_fill(diagonal, 0.0)
    affinity, indices = shared.topk(min(config.evidence_topk, max(patches - 1, 1)), dim=-1)
    mass = affinity.sum(-1)
    weights = affinity / mass[..., None].clamp_min(config.epsilon)

    def key_margin(scores: Tensor) -> Tensor:
        return (scores[b, indices, first[..., None], second[..., None]]
                - scores[b, indices, second[..., None], first[..., None]])

    self_keys, native_keys = key_margin(self_pair_scores), key_margin(native_pair_scores)
    neighbor_self = (weights * self_keys).sum(-1)
    neighbor_native = (weights * native_keys).sum(-1)
    same_sign = self_keys * native_keys > 0
    signed_keys = torch.where(
        same_sign, self_keys.sign() * torch.minimum(self_keys.abs(), native_keys.abs()),
        torch.zeros_like(self_keys),
    )
    neighbor_signed = (weights * signed_keys).sum(-1)
    local_self = _pair_diagonal(self_pair_scores, first, second)
    local_native = _pair_diagonal(native_pair_scores, first, second)
    class_self = self_scores.gather(-1, first[..., None]).squeeze(-1) - self_scores.gather(-1, second[..., None]).squeeze(-1)
    class_native = native_scores.gather(-1, first[..., None]).squeeze(-1) - native_scores.gather(-1, second[..., None]).squeeze(-1)

    local_agreement = (
        (self_scores.argmax(-1) == second) & (native_scores.argmax(-1) == second)
        & (local_self < 0) & (local_native < 0)
    )
    neighborhood_agreement = (
        (mass > config.epsilon) & (neighbor_self < 0)
        & (neighbor_native < 0) & (neighbor_signed < 0)
    )
    # The weakest contradicting measurement bounds the replacement margin.
    support = torch.stack((
        -local_self, -local_native, -class_self, -class_native,
        -neighbor_self, -neighbor_native, -neighbor_signed,
    ), dim=-1).amin(-1).clamp_min(0)
    residual = torch.tensor([
        name.casefold() in {item.casefold() for item in config.residual_class_names}
        for name in class_names
    ], device=base_scores.device, dtype=torch.bool)
    eligible = (
        valid & local_agreement & neighborhood_agreement
        & (support > base_margin + config.epsilon)
        & ~residual[first] & ~residual[second]
    )

    temperature = config.alias_temperature
    log_mass = torch.logsumexp(top.values / temperature, dim=-1, keepdim=True)
    target_pair = torch.stack((-0.5 * support, 0.5 * support), dim=-1)
    shifted = target_pair + temperature * (
        log_mass - torch.logsumexp(target_pair / temperature, dim=-1, keepdim=True)
    )
    proposed = base_scores.scatter(-1, top.indices, shifted)
    corrected = torch.where(eligible[..., None], proposed, base_scores)
    correction = (corrected - base_scores).abs().amax(-1)
    consensus = neighbor_signed.abs() / (weights * signed_keys.abs()).sum(-1).clamp_min(config.epsilon)
    details = {
        "text_separation": torch.zeros_like(support),
        "local_reliability": same_sign.float().mean(-1),
        "coverage": (weights * same_sign).sum(-1),
        "consensus": consensus,
        "absolute_correction": correction,
        "contradiction": (local_self < 0) & (local_native < 0),
        "local_agreement": local_agreement,
        "neighborhood_agreement": neighborhood_agreement,
        "eligible": eligible,
    }
    return corrected, top.indices, -support, eligible.float(), details


def _query_pair_key_margin(scores: Tensor, first: Tensor, second: Tensor) -> Tensor:
    batch = scores.shape[0]
    lookup = scores.permute(0, 2, 3, 1)
    b = torch.arange(batch, device=scores.device)[:, None]
    positive = lookup[b, first, second]
    negative = lookup[b, second, first]
    return positive - negative


def _pair_diagonal(scores: Tensor, first: Tensor, second: Tensor) -> Tensor:
    batch, patches = first.shape
    b = torch.arange(batch, device=scores.device)[:, None]
    n = torch.arange(patches, device=scores.device)[None, :]
    return scores[b, n, first, second] - scores[b, n, second, first]


def _uniform_class_scores(
    features: Tensor,
    text: Tensor,
    bank: TCPRTextBank,
    subset: Tensor,
    temperature: float,
) -> Tensor:
    return uniform_subset_scores(
        features @ text.T,
        bank.parent_indices,
        bank.class_count,
        subset,
        temperature,
    )


def self_attended(native_attention: Tensor, values: Tensor, prefix: int) -> Tensor:
    """Retain each patch value while preserving attention to special tokens."""
    prefix_attended = native_attention[..., :prefix, :] @ values
    special = native_attention[..., prefix:, :prefix] @ values[..., :prefix, :]
    patch_mass = native_attention[..., prefix:, prefix:].sum(-1, keepdim=True)
    patch = values[..., prefix:, :]
    return torch.cat((prefix_attended, special + patch_mass * patch), dim=2)


def enforce_residual_partition(
    corrected: Tensor,
    base_scores: Tensor,
    class_names: tuple[str, ...],
    residual_names: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    """Revert corrections that cross a named residual/foreground partition."""
    residual = [
        index for index, name in enumerate(class_names)
        if name.casefold() in {item.casefold() for item in residual_names}
    ]
    if not residual:
        return corrected, torch.zeros(corrected.shape[:-1], device=corrected.device, dtype=torch.bool)
    if len(residual) != 1:
        raise ValueError("DGER supports at most one residual class.")
    residual_index = residual[0]
    base_is_residual = base_scores.argmax(-1) == residual_index
    corrected_is_residual = corrected.argmax(-1) == residual_index
    crossing = base_is_residual != corrected_is_residual
    return torch.where(crossing[..., None], base_scores, corrected), crossing


def _valid_mean(values: Tensor, valid: Tensor) -> float:
    selected = values[valid]
    return float(selected.mean().item()) if selected.numel() else 0.0
