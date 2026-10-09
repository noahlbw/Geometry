"""Training-free competitive rereading constrained by frozen visual evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

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
class CVERConfig:
    alias_temperature: float = 0.07
    semantic_temperature: float = 0.08
    raw_temperature: float = 0.15
    native_weight: float = 0.25
    raw_weight: float = 0.50
    semantic_weight: float = 1.00
    maximum_route_fraction: float = 0.50
    route_kl_scale: float = 1.00
    pair_strength: float = 8.00
    correction_clip: float = 0.05
    epsilon: float = 1e-6
    residual_class_names: tuple[str, ...] = ("background",)

    def validate(self) -> None:
        positive = (
            self.alias_temperature,
            self.semantic_temperature,
            self.raw_temperature,
            self.route_kl_scale,
            self.correction_clip,
            self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("CVER temperatures, scales, clipping, and epsilon must be positive.")
        nonnegative = (
            self.native_weight,
            self.raw_weight,
            self.semantic_weight,
            self.maximum_route_fraction,
            self.pair_strength,
        )
        if min(nonnegative) < 0:
            raise ValueError("CVER evidence weights and strengths must be non-negative.")
        if self.maximum_route_fraction > 1:
            raise ValueError("maximum_route_fraction must be in [0, 1].")


@dataclass(frozen=True)
class CVERDiagnostics:
    mean_route_kl: float
    mean_raw_fit: float
    mean_semantic_fit: float
    mean_pair_quality: float
    mean_absolute_pair_evidence: float
    changed_from_geometry: float
    geometry_self_disagreement: float
    geometry_native_disagreement: float

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CVERResult:
    geometry_logits: Tensor
    logits: Tensor
    self_logits: Tensor
    native_logits: Tensor
    response_matrix: Tensor
    route_quality: Tensor
    pair_ids: Tensor
    pair_evidence: Tensor
    diagnostics: CVERDiagnostics


class CVERSegmenter:
    """Geometry prior plus visually constrained, all-class competitive rereading."""

    method_name = "CVER: competitive visual-evidence rereading"

    def __init__(
        self,
        backbone,
        config: CVERConfig = CVERConfig(),
        tcpr_config: TCPRConfig = TCPRConfig(maximum_aliases_per_class=20),
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

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> CVERResult:
        text_bank.validate()
        if text_bank.class_count < 2:
            raise ValueError("CVER requires at least two competing classes.")
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
        geometry_scores = _class_scores(
            geometry_features, text, text_bank, all_aliases, self.config.alias_temperature
        )
        native_scores = _class_scores(
            native_features, text, text_bank, all_aliases, self.config.alias_temperature
        )
        self_scores = _class_scores(
            self_features, text, text_bank, all_aliases, self.config.alias_temperature
        )

        evidence_scores = (geometry_scores + native_scores + self_scores) / 3.0
        semantic_contrast = class_contrast(evidence_scores)
        geometry = _normalize_relation(prepared.geometry_patch_conditional.float(), valid)
        native = _normalize_relation(
            prepared.native_patch_conditional.float().mean(1), valid
        )
        raw = F.normalize(prepared.raw_patch_tokens.float(), dim=-1)
        raw_similarity = raw @ raw.transpose(-1, -2)
        routes, route_quality, route_kl, raw_fit, semantic_fit = visual_constrained_relations(
            geometry,
            native,
            raw_similarity,
            semantic_contrast,
            valid,
            self.config,
        )
        reread = self._reread_hypotheses(prepared, routes)
        reread_alias = torch.einsum("bcnd,md->bcnm", reread, text)
        responses = all_class_hypothesis_scores(
            reread_alias,
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        corrected, pair_ids, pair_evidence, pair_quality = competitive_pair_correction(
            geometry_scores,
            responses,
            route_quality,
            valid,
            self.config,
        )
        correction = corrected - geometry_scores
        correction = preserve_residual_partition(
            correction,
            geometry_scores,
            text_bank.class_names,
            self.config.residual_class_names,
        )
        corrected = torch.where(valid[..., None], geometry_scores + correction, geometry_scores)

        geometry_label = geometry_scores.argmax(-1)
        final_label = corrected.argmax(-1)
        self_label = self_scores.argmax(-1)
        native_label = native_scores.argmax(-1)
        count = max(int(valid.sum()), 1)
        valid_classes = valid[:, None, :].expand_as(route_quality)
        diagnostics = CVERDiagnostics(
            mean_route_kl=float(route_kl[valid_classes].mean().item()),
            mean_raw_fit=float(raw_fit[valid_classes].mean().item()),
            mean_semantic_fit=float(semantic_fit[valid_classes].mean().item()),
            mean_pair_quality=float(pair_quality[valid].mean().item()),
            mean_absolute_pair_evidence=float(pair_evidence[valid].abs().mean().item()),
            changed_from_geometry=float(((geometry_label != final_label) & valid).sum().item() / count),
            geometry_self_disagreement=float(((geometry_label != self_label) & valid).sum().item() / count),
            geometry_native_disagreement=float(((geometry_label != native_label) & valid).sum().item() / count),
        )
        height, width = prepared.grid_height, prepared.grid_width
        return CVERResult(
            geometry_logits=_tokens_to_map(geometry_scores, height, width),
            logits=_tokens_to_map(corrected, height, width),
            self_logits=_tokens_to_map(self_scores, height, width),
            native_logits=_tokens_to_map(native_scores, height, width),
            response_matrix=responses,
            route_quality=route_quality.reshape(
                batch, text_bank.class_count, height, width
            ),
            pair_ids=pair_ids.transpose(1, 2).reshape(batch, 2, height, width),
            pair_evidence=pair_evidence.reshape(batch, height, width),
            diagnostics=diagnostics,
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

    @torch.inference_mode()
    def _reread_hypotheses(self, prepared: TCPRPreparedImage, routes: Tensor) -> Tensor:
        batch, classes, patches, _ = routes.shape
        prefix = prepared.prefix_tokens

        def expand(values: Tensor) -> Tensor:
            return values[:, None].expand(batch, classes, *values.shape[1:]).reshape(
                batch * classes, *values.shape[1:]
            )

        native_attention = expand(prepared.native_attention)
        values = expand(prepared.value_tokens)
        shared = expand(prepared.shared_tokens)
        attended = relation_attended(
            native_attention,
            routes.reshape(batch * classes, patches, patches),
            values,
            prefix,
        )
        head = self.backbone.model.visual_model.head
        with self.backbone._autocast():
            current = _finish_attention_block(head.blocks[prepared.block_index], shared, attended)
            projected = _finish_head(head, current, prepared.block_index)
        projected = F.normalize(projected[:, prefix:].float(), dim=-1)
        return projected.reshape(batch, classes, patches, projected.shape[-1])


def _class_scores(
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


def class_contrast(scores: Tensor) -> Tensor:
    """Return each class score relative to the strongest competing class."""
    if scores.ndim != 3 or scores.shape[-1] < 2:
        raise ValueError("scores must have shape [B,N,C] with C >= 2.")
    top = torch.topk(scores, k=2, dim=-1)
    class_ids = torch.arange(scores.shape[-1], device=scores.device)[None, None]
    competitor = torch.where(
        top.indices[..., :1] == class_ids,
        top.values[..., 1:2],
        top.values[..., :1],
    )
    return scores - competitor


def visual_constrained_relations(
    geometry: Tensor,
    native: Tensor,
    raw_similarity: Tensor,
    semantic_contrast: Tensor,
    valid: Tensor,
    config: CVERConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Solve an entropy-regularized visual/semantic rereading distribution."""
    batch, patches, _ = geometry.shape
    classes = semantic_contrast.shape[-1]
    if native.shape != geometry.shape or raw_similarity.shape != geometry.shape:
        raise ValueError("Geometry, native, and raw similarity must share [B,N,N].")
    if semantic_contrast.shape != (batch, patches, classes) or valid.shape != (batch, patches):
        raise ValueError("Semantic contrast or valid mask has incompatible shape.")

    key_valid = valid[:, None, :]
    candidate_prior = geometry.masked_fill(~key_valid, 0.0)
    native_prior = native.masked_fill(~key_valid, 0.0)
    if patches > 1:
        diagonal = torch.eye(patches, device=geometry.device, dtype=torch.bool)[None]
        candidate_prior = candidate_prior.masked_fill(diagonal, 0.0)
        native_prior = native_prior.masked_fill(diagonal, 0.0)
    candidate_prior = candidate_prior / candidate_prior.sum(-1, keepdim=True).clamp_min(config.epsilon)
    native_prior = native_prior / native_prior.sum(-1, keepdim=True).clamp_min(config.epsilon)
    empty = candidate_prior.sum(-1, keepdim=True) <= config.epsilon
    candidate_prior = torch.where(empty, geometry, candidate_prior)
    native_prior = torch.where(empty, native, native_prior)

    semantic_key = semantic_contrast.transpose(1, 2)[:, :, None, :]
    logits = candidate_prior.clamp_min(config.epsilon).log()[:, None]
    logits = logits + config.native_weight * native_prior.clamp_min(config.epsilon).log()[:, None]
    logits = logits + config.raw_weight * raw_similarity[:, None] / config.raw_temperature
    logits = logits + config.semantic_weight * semantic_key / config.semantic_temperature
    logits = logits.masked_fill(~key_valid[:, None], -torch.inf)
    if patches > 1:
        logits = logits.masked_fill(diagonal[:, None], -torch.inf)
    evidence = torch.softmax(logits, dim=-1)
    evidence = torch.where(valid[:, None, :, None], evidence, geometry[:, None])

    raw_unit = raw_similarity.add(1.0).mul(0.5).clamp(0.0, 1.0)
    raw_fit = (evidence * raw_unit[:, None]).sum(-1)
    semantic_probability = torch.sigmoid(semantic_key / config.semantic_temperature)
    semantic_fit = (evidence * semantic_probability).sum(-1)
    kl = evidence * (
        evidence.clamp_min(config.epsilon).log()
        - candidate_prior[:, None].clamp_min(config.epsilon).log()
    )
    kl = kl.sum(-1).clamp_min(0.0)
    quality = raw_fit * semantic_fit * torch.exp(-kl / config.route_kl_scale)
    quality = quality.clamp(0.0, 1.0) * valid[:, None]
    mixture = config.maximum_route_fraction * quality[..., None]
    routes = (1.0 - mixture) * geometry[:, None] + mixture * evidence
    routes = routes / routes.sum(-1, keepdim=True).clamp_min(config.epsilon)
    return routes, quality, kl, raw_fit, semantic_fit


def all_class_hypothesis_scores(
    hypothesis_alias_scores: Tensor,
    parents: Tensor,
    class_count: int,
    subset: Tensor,
    temperature: float,
) -> Tensor:
    """Score every rereading hypothesis against every semantic class."""
    if hypothesis_alias_scores.ndim != 4:
        raise ValueError("hypothesis_alias_scores must have shape [B,H,N,M].")
    output = []
    for class_index in range(class_count):
        members = (parents == class_index) & subset
        count = int(members.sum())
        if count < 1:
            raise ValueError(f"Alias subset is empty for class {class_index}.")
        values = hypothesis_alias_scores[..., members]
        output.append(
            temperature * (torch.logsumexp(values / temperature, dim=-1) - math.log(count))
        )
    return torch.stack(output, dim=-1)


def competitive_pair_correction(
    base_scores: Tensor,
    responses: Tensor,
    route_quality: Tensor,
    valid: Tensor,
    config: CVERConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply a zero-sum correction from complete top-two hypothesis responses."""
    batch, patches, classes = base_scores.shape
    if responses.shape != (batch, classes, patches, classes):
        raise ValueError("responses must have shape [B,C,N,C].")
    if route_quality.shape != (batch, classes, patches):
        raise ValueError("route_quality must have shape [B,C,N].")
    top = torch.topk(base_scores, k=2, dim=-1)
    first, second = top.indices[..., 0], top.indices[..., 1]
    response = responses.permute(0, 2, 1, 3)
    b = torch.arange(batch, device=base_scores.device)[:, None]
    n = torch.arange(patches, device=base_scores.device)[None, :]
    first_first = response[b, n, first, first]
    first_second = response[b, n, first, second]
    second_second = response[b, n, second, second]
    second_first = response[b, n, second, first]
    base_first = base_scores.gather(-1, first[..., None]).squeeze(-1)
    base_second = base_scores.gather(-1, second[..., None]).squeeze(-1)
    gain_first = (first_first - first_second) - (base_first - base_second)
    gain_second = (second_second - second_first) - (base_second - base_first)
    evidence = 0.5 * (gain_first - gain_second)
    evidence = config.correction_clip * torch.tanh(evidence / config.correction_clip)

    quality = route_quality.transpose(1, 2)
    quality_first = quality.gather(-1, first[..., None]).squeeze(-1)
    quality_second = quality.gather(-1, second[..., None]).squeeze(-1)
    pair_quality = torch.sqrt((quality_first * quality_second).clamp_min(0.0))
    delta = config.pair_strength * pair_quality * evidence * valid
    correction = torch.zeros_like(base_scores)
    correction.scatter_add_(-1, first[..., None], delta[..., None])
    correction.scatter_add_(-1, second[..., None], -delta[..., None])
    return base_scores + correction, torch.stack((first, second), dim=-1), delta, pair_quality


def self_attended(native_attention: Tensor, values: Tensor, prefix: int) -> Tensor:
    """Retain each patch's own value while preserving special-token attention."""
    prefix_attended = native_attention[..., :prefix, :] @ values
    special = native_attention[..., prefix:, :prefix] @ values[..., :prefix, :]
    patch_mass = native_attention[..., prefix:, prefix:].sum(-1, keepdim=True)
    patch = values[..., prefix:, :]
    return torch.cat((prefix_attended, special + patch_mass * patch), dim=2)


def relation_attended(native_attention: Tensor, relation: Tensor, values: Tensor, prefix: int) -> Tensor:
    prefix_attended = native_attention[..., :prefix, :] @ values
    special = native_attention[..., prefix:, :prefix] @ values[..., :prefix, :]
    patch_mass = native_attention[..., prefix:, prefix:].sum(-1, keepdim=True)
    patch = relation[:, None].to(values.dtype) @ values[..., prefix:, :]
    return torch.cat((prefix_attended, special + patch_mass * patch), dim=2)


def _normalize_relation(relation: Tensor, valid: Tensor) -> Tensor:
    masked = relation.masked_fill(~valid[:, None, :], 0.0)
    normalized = masked / masked.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.where(valid[:, :, None], normalized, relation)
