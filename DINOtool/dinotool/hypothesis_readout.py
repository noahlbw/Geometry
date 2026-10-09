"""Training-free class-hypothesis rereading for frozen DINO.text."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
import torch.nn.functional as F
from torch import Tensor

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
)


@dataclass(frozen=True)
class HEROConfig:
    alias_temperature: float = 0.07
    route_temperature: float = 0.07
    evidence_temperature: float = 0.10
    delta_strength: float = 1.0
    query_chunk: int = 64
    epsilon: float = 1e-6
    split_salt: str = "hero-ov-v1"
    residual_class_names: tuple[str, ...] = ("background",)

    def validate(self) -> None:
        positive = (
            self.alias_temperature,
            self.route_temperature,
            self.evidence_temperature,
            self.query_chunk,
            self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("HERO temperatures, chunk size, and epsilon must be positive.")
        if self.delta_strength < 0:
            raise ValueError("delta_strength must be non-negative.")
        if not self.split_salt:
            raise ValueError("split_salt must be non-empty.")


@dataclass(frozen=True)
class HERODiagnostics:
    mean_route_kl: float
    maximum_route_kl: float
    fold_sign_agreement: float
    mean_cross_delta: float
    mean_consensus_delta: float
    changed_cross_from_geometry: float
    changed_consensus_from_geometry: float

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HEROResult:
    geometry_logits: Tensor
    cross_logits: Tensor
    consensus_logits: Tensor
    same_group_logits: Tensor
    fold_deltas: Tensor
    diagnostics: HERODiagnostics


class HEROSegmenter:
    """Geometry readout followed by held-out text-verified hypothesis rereading."""

    method_name = "HERO-OV: hypothesis-evidence readout"

    def __init__(
        self,
        backbone,
        config: HEROConfig = HEROConfig(),
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
    ) -> HEROResult:
        text_bank.validate()
        if text_bank.class_count < 2:
            raise ValueError("HERO requires at least two competing classes.")
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
        alias_scores = visual @ text.T
        split_masks = deterministic_alias_split(text_bank, self.config.split_salt)
        all_mask = torch.ones(text.shape[0], device=text.device, dtype=torch.bool)
        geometry_scores = uniform_subset_scores(
            alias_scores,
            text_bank.parent_indices,
            text_bank.class_count,
            all_mask,
            self.config.alias_temperature,
        )
        geometry = _normalize_geometry(prepared.geometry_patch_conditional.float(), valid)

        cross_deltas: list[Tensor] = []
        same_deltas: list[Tensor] = []
        route_kls: list[Tensor] = []
        for route_fold in range(2):
            route_mask = split_masks[route_fold]
            heldout_mask = split_masks[1 - route_fold]
            route = hypothesis_conditioned_relations(
                geometry,
                alias_scores,
                text_bank.parent_indices,
                text_bank.class_count,
                route_mask,
                valid,
                self.config,
            )
            reread = self._reread_hypotheses(prepared, route)
            reread_alias = torch.einsum("bcnd,md->bcnm", reread, text)
            cross_after = hypothesis_subset_scores(
                reread_alias,
                text_bank.parent_indices,
                text_bank.class_count,
                heldout_mask,
                self.config.alias_temperature,
            )
            same_after = hypothesis_subset_scores(
                reread_alias,
                text_bank.parent_indices,
                text_bank.class_count,
                route_mask,
                self.config.alias_temperature,
            )
            cross_before = uniform_subset_scores(
                alias_scores,
                text_bank.parent_indices,
                text_bank.class_count,
                heldout_mask,
                self.config.alias_temperature,
            )
            same_before = uniform_subset_scores(
                alias_scores,
                text_bank.parent_indices,
                text_bank.class_count,
                route_mask,
                self.config.alias_temperature,
            )
            cross_deltas.append(cross_after - cross_before)
            same_deltas.append(same_after - same_before)
            route_kls.append(relation_kl(route, geometry, valid, self.config.epsilon))

        fold_delta = torch.stack(cross_deltas, dim=0)
        cross_delta = fold_delta.mean(0)
        same_delta = torch.stack(same_deltas, dim=0).mean(0)
        consensus_delta = consensus_evidence(fold_delta[0], fold_delta[1])
        cross_delta = preserve_residual_partition(
            cross_delta, geometry_scores, text_bank.class_names, self.config.residual_class_names
        )
        consensus_delta = preserve_residual_partition(
            consensus_delta,
            geometry_scores,
            text_bank.class_names,
            self.config.residual_class_names,
        )
        same_delta = preserve_residual_partition(
            same_delta, geometry_scores, text_bank.class_names, self.config.residual_class_names
        )
        strength = self.config.delta_strength
        cross_scores = geometry_scores + strength * cross_delta
        consensus_scores = geometry_scores + strength * consensus_delta
        same_scores = geometry_scores + strength * same_delta
        cross_scores = torch.where(valid[..., None], cross_scores, geometry_scores)
        consensus_scores = torch.where(valid[..., None], consensus_scores, geometry_scores)
        same_scores = torch.where(valid[..., None], same_scores, geometry_scores)

        valid_classes = valid[..., None].expand_as(cross_delta)
        agreement = ((fold_delta[0] * fold_delta[1]) > 0) & valid_classes
        geometry_label = geometry_scores.argmax(-1)
        cross_label = cross_scores.argmax(-1)
        consensus_label = consensus_scores.argmax(-1)
        valid_count = max(int(valid.sum()), 1)
        class_count = text_bank.class_count
        route_kl_values = torch.stack(route_kls)
        diagnostics = HERODiagnostics(
            mean_route_kl=float(route_kl_values.mean().item()),
            maximum_route_kl=float(route_kl_values.max().item()),
            fold_sign_agreement=float(agreement.sum().item() / max(valid_count * class_count, 1)),
            mean_cross_delta=float(cross_delta[valid_classes].abs().mean().item()),
            mean_consensus_delta=float(consensus_delta[valid_classes].abs().mean().item()),
            changed_cross_from_geometry=float(
                ((cross_label != geometry_label) & valid).sum().item() / valid_count
            ),
            changed_consensus_from_geometry=float(
                ((consensus_label != geometry_label) & valid).sum().item() / valid_count
            ),
        )
        height, width = prepared.grid_height, prepared.grid_width
        return HEROResult(
            geometry_logits=_tokens_to_map(geometry_scores, height, width),
            cross_logits=_tokens_to_map(cross_scores, height, width),
            consensus_logits=_tokens_to_map(consensus_scores, height, width),
            same_group_logits=_tokens_to_map(same_scores, height, width),
            fold_deltas=fold_delta,
            diagnostics=diagnostics,
        )

    @torch.inference_mode()
    def _reread_hypotheses(
        self,
        prepared: TCPRPreparedImage,
        relations: Tensor,
    ) -> Tensor:
        batch, classes, patches, _ = relations.shape
        prefix = prepared.prefix_tokens

        def expand(values: Tensor) -> Tensor:
            return values[:, None].expand(batch, classes, *values.shape[1:]).reshape(
                batch * classes, *values.shape[1:]
            )

        native_attention = expand(prepared.native_attention)
        value_tokens = expand(prepared.value_tokens)
        shared_tokens = expand(prepared.shared_tokens)
        attended = _geometry_attended(
            native_attention,
            relations.reshape(batch * classes, patches, patches),
            value_tokens,
            prefix,
        )
        head = self.backbone.model.visual_model.head
        with self.backbone._autocast():
            current = _finish_attention_block(
                head.blocks[prepared.block_index], shared_tokens, attended
            )
            projected = _finish_head(head, current, prepared.block_index)
        projected = F.normalize(projected[:, prefix:].float(), dim=-1)
        return projected.reshape(batch, classes, patches, projected.shape[-1])


def deterministic_alias_split(bank: TCPRTextBank, salt: str) -> Tensor:
    """Split each class deterministically without using image data or labels."""
    masks = torch.zeros((2, len(bank.alias_names)), device=bank.features.device, dtype=torch.bool)
    for class_index in range(bank.class_count):
        members = (bank.parent_indices == class_index).nonzero(as_tuple=False).flatten().tolist()
        ordered = sorted(
            members,
            key=lambda index: hashlib.sha256(
                f"{salt}\0{bank.alias_names[index]}".encode("utf-8")
            ).digest(),
        )
        if len(ordered) == 1:
            masks[:, ordered[0]] = True
            continue
        split = (len(ordered) + 1) // 2
        masks[0, ordered[:split]] = True
        masks[1, ordered[split:]] = True
    return masks


def uniform_subset_scores(
    alias_scores: Tensor,
    parents: Tensor,
    class_count: int,
    subset: Tensor,
    temperature: float,
) -> Tensor:
    if alias_scores.ndim != 3 or subset.shape != (alias_scores.shape[-1],):
        raise ValueError("Expected alias_scores [B,N,M] and subset [M].")
    output = []
    for class_index in range(class_count):
        members = (parents == class_index) & subset
        count = int(members.sum())
        if count < 1:
            raise ValueError(f"Alias subset is empty for class {class_index}.")
        values = alias_scores[..., members]
        output.append(
            temperature * (torch.logsumexp(values / temperature, dim=-1) - math.log(count))
        )
    return torch.stack(output, dim=-1)


def hypothesis_subset_scores(
    hypothesis_alias_scores: Tensor,
    parents: Tensor,
    class_count: int,
    subset: Tensor,
    temperature: float,
) -> Tensor:
    if hypothesis_alias_scores.ndim != 4:
        raise ValueError("hypothesis_alias_scores must have shape [B,C,N,M].")
    if hypothesis_alias_scores.shape[1] != class_count:
        raise ValueError("The hypothesis axis must match class_count.")
    output = []
    for class_index in range(class_count):
        members = (parents == class_index) & subset
        count = int(members.sum())
        if count < 1:
            raise ValueError(f"Alias subset is empty for class {class_index}.")
        values = hypothesis_alias_scores[:, class_index, :, members]
        output.append(
            temperature * (torch.logsumexp(values / temperature, dim=-1) - math.log(count))
        )
    return torch.stack(output, dim=-1)


def hypothesis_conditioned_relations(
    geometry: Tensor,
    alias_scores: Tensor,
    parents: Tensor,
    class_count: int,
    route_subset: Tensor,
    valid: Tensor,
    config: HEROConfig,
) -> Tensor:
    """Use query-specific text hypotheses to redistribute Geometry evidence."""
    batch, patches, _ = geometry.shape
    supports = torch.empty(
        (batch, class_count, patches, patches),
        device=geometry.device,
        dtype=torch.float32,
    )
    for start in range(0, patches, config.query_chunk):
        stop = min(start + config.query_chunk, patches)
        for class_index in range(class_count):
            members = (parents == class_index) & route_subset
            query = torch.softmax(
                alias_scores[:, start:stop, members] / config.route_temperature,
                dim=-1,
            )
            candidate = alias_scores[:, :, members]
            supports[:, class_index, start:stop] = torch.einsum(
                "bqm,bjm->bqj", query, candidate
            )

    top = torch.topk(supports, k=2, dim=1)
    class_ids = torch.arange(class_count, device=geometry.device)[None, :, None, None]
    competitor = torch.where(
        top.indices[:, :1] == class_ids,
        top.values[:, 1:2],
        top.values[:, :1],
    )
    relative = supports - competitor
    logits = geometry.clamp_min(config.epsilon).log()[:, None]
    logits = logits + relative / config.evidence_temperature
    logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
    conditioned = torch.softmax(logits, dim=-1)
    return torch.where(valid[:, None, :, None], conditioned, geometry[:, None])


def consensus_evidence(first: Tensor, second: Tensor) -> Tensor:
    """Retain only cross-fold effects with the same direction."""
    same_sign = first * second > 0
    magnitude = torch.minimum(first.abs(), second.abs())
    direction = (first + second).sign()
    return torch.where(same_sign, direction * magnitude, torch.zeros_like(first))


def preserve_residual_partition(
    delta: Tensor,
    base_scores: Tensor,
    class_names: tuple[str, ...],
    residual_names: tuple[str, ...],
) -> Tensor:
    """Let hypothesis rereading reorder named classes without inventing foreground."""
    residual = [
        index for index, name in enumerate(class_names)
        if name.casefold() in {item.casefold() for item in residual_names}
    ]
    if not residual:
        return delta
    if len(residual) != 1:
        raise ValueError("HERO currently supports at most one residual class.")
    residual_index = residual[0]
    base_label = base_scores.argmax(-1)
    active = base_label != residual_index
    anchor = delta.gather(-1, base_label[..., None])
    adjusted = delta - anchor
    adjusted[..., residual_index] = 0.0
    return torch.where(active[..., None], adjusted, torch.zeros_like(adjusted))


def relation_kl(conditioned: Tensor, geometry: Tensor, valid: Tensor, epsilon: float) -> Tensor:
    base = geometry[:, None].expand_as(conditioned)
    values = conditioned * (
        conditioned.clamp_min(epsilon).log() - base.clamp_min(epsilon).log()
    )
    values = values.sum(-1).masked_fill(~valid[:, None], 0.0)
    valid_count = max(int(valid.sum()) * conditioned.shape[1], 1)
    return values.sum() / valid_count


def _normalize_geometry(geometry: Tensor, valid: Tensor) -> Tensor:
    masked = geometry.masked_fill(~valid[:, None, :], 0.0)
    normalized = masked / masked.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.where(valid[:, :, None], normalized, geometry)
