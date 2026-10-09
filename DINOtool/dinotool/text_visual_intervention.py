"""Controlled training-free interventions between DINO visual and text states."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .tcpr import (
    TCPRPreparedImage,
    TCPRTextBank,
    aggregate_alias_scores,
    filter_aliases,
    _normalize_valid_mask,
    _tokens_to_map,
)


@dataclass(frozen=True)
class TextVisualInterventionConfig:
    alias_temperature: float = 0.07
    class_temperature: float = 0.07
    semantic_relation_strength: float = 4.0
    weak_update: float = 0.25
    strong_update: float = 0.50

    def validate(self) -> None:
        if min(self.alias_temperature, self.class_temperature, self.semantic_relation_strength) <= 0:
            raise ValueError("Temperatures and semantic relation strength must be positive.")
        if not 0.0 <= self.weak_update <= self.strong_update <= 1.0:
            raise ValueError("Update strengths must satisfy 0 <= weak <= strong <= 1.")


@dataclass(frozen=True)
class TextVisualInterventionResult:
    logits: dict[str, Tensor]
    feature_shift: dict[str, float]
    relation_kl_from_geometry: float
    alias_kept_mean: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "feature_shift": self.feature_shift,
            "relation_kl_from_geometry": self.relation_kl_from_geometry,
            "alias_kept_mean": self.alias_kept_mean,
        }


@torch.inference_mode()
def text_visual_intervention_readouts(
    prepared: TCPRPreparedImage,
    canonical_bank: TCPRTextBank,
    expanded_bank: TCPRTextBank,
    alias_config,
    config: TextVisualInterventionConfig = TextVisualInterventionConfig(),
    *,
    valid_mask: Tensor | None = None,
) -> TextVisualInterventionResult:
    """Return matched self-attention, cross-attention, serial, and parallel arms."""
    config.validate()
    canonical_bank.validate()
    expanded_bank.validate()
    visual = F.normalize(prepared.geometry_projected.float(), dim=-1)
    batch, patches, _ = visual.shape
    valid = _normalize_valid_mask(
        valid_mask,
        batch,
        prepared.grid_height,
        prepared.grid_width,
        visual.device,
    ).reshape(batch, patches)

    canonical_text = F.normalize(canonical_bank.features.float(), dim=-1)
    canonical_alias = visual @ canonical_text.T
    canonical_weights = torch.ones_like(canonical_alias[:, 0])
    canonical_scores = aggregate_alias_scores(
        canonical_alias,
        canonical_weights,
        canonical_bank.parent_indices,
        canonical_bank.class_count,
        config.alias_temperature,
    )

    expanded_text = F.normalize(expanded_bank.features.float(), dim=-1)
    geometry_alias = visual @ expanded_text.T
    native_alias = prepared.native_projected.float() @ expanded_text.T
    alias_weights = filter_aliases(
        geometry_alias,
        native_alias,
        expanded_bank.parent_indices,
        expanded_bank.canonical_mask,
        expanded_bank.class_count,
        valid,
        alias_config,
    )
    expanded_scores = aggregate_alias_scores(
        geometry_alias,
        alias_weights,
        expanded_bank.parent_indices,
        expanded_bank.class_count,
        config.alias_temperature,
    )
    prototypes = _class_prototypes(expanded_text, alias_weights, expanded_bank.parent_indices,
                                   expanded_bank.class_count)
    class_probability = torch.softmax(expanded_scores / config.class_temperature, dim=-1)

    geometry = prepared.geometry_patch_conditional.float()
    geometry = geometry.masked_fill(~valid[:, None, :], 0.0)
    geometry = geometry / geometry.sum(-1, keepdim=True).clamp_min(1e-8)
    image_context = F.normalize(geometry @ visual, dim=-1)
    text_relation = _text_conditioned_relation(
        geometry, class_probability, valid, config.semantic_relation_strength
    )
    text_context = F.normalize(text_relation @ visual, dim=-1)
    shuffled_probability = torch.roll(class_probability, shifts=max(patches // 3, 1), dims=1)
    shuffled_relation = _text_conditioned_relation(
        geometry, shuffled_probability, valid, config.semantic_relation_strength,
        query_probability=class_probability,
    )
    shuffled_context = F.normalize(shuffled_relation @ visual, dim=-1)
    cross_context = _cross_context(visual, prototypes, config.class_temperature)

    weak, strong = config.weak_update, config.strong_update
    features = {
        "ImageSelf_50": _blend(visual, image_context, strong),
        "TextSelf_25": _blend(visual, text_context, weak),
        "TextSelf_50": _blend(visual, text_context, strong),
        "Cross_25": _blend(visual, cross_context, weak),
        "Cross_50": _blend(visual, cross_context, strong),
        "Parallel_25": _parallel(visual, text_context, cross_context, weak),
        "Parallel_50": _parallel(visual, text_context, cross_context, strong),
        "TextSelf_shuffled_50": _blend(visual, shuffled_context, strong),
    }
    serial_self = _blend(visual, text_context, strong)
    serial_cross = _cross_context(serial_self, prototypes, config.class_temperature)
    features["Serial_50"] = _blend(serial_self, serial_cross, strong)

    logits = {
        "G_canonical": _tokens_to_map(canonical_scores, prepared.grid_height, prepared.grid_width),
        "G_expanded": _tokens_to_map(expanded_scores, prepared.grid_height, prepared.grid_width),
        "TextGraph_logits": _tokens_to_map(
            text_relation @ expanded_scores, prepared.grid_height, prepared.grid_width
        ),
    }
    shift: dict[str, float] = {}
    for name, values in features.items():
        alias_scores = values @ expanded_text.T
        class_scores = aggregate_alias_scores(
            alias_scores,
            alias_weights,
            expanded_bank.parent_indices,
            expanded_bank.class_count,
            config.alias_temperature,
        )
        logits[name] = _tokens_to_map(class_scores, prepared.grid_height, prepared.grid_width)
        displacement = 1.0 - F.cosine_similarity(values, visual, dim=-1)
        shift[name] = float(displacement[valid].mean().item()) if bool(valid.any()) else 0.0

    kl = text_relation * (
        text_relation.clamp_min(1e-8).log() - geometry.clamp_min(1e-8).log()
    )
    valid_rows = valid[:, :, None].expand_as(kl)
    relation_kl = float(kl.masked_fill(~valid_rows, 0.0).sum().item() / max(int(valid.sum()), 1))
    return TextVisualInterventionResult(
        logits=logits,
        feature_shift=shift,
        relation_kl_from_geometry=relation_kl,
        alias_kept_mean=float((alias_weights > 0).sum(-1).float().mean().item()),
    )


def _class_prototypes(text: Tensor, weights: Tensor, parents: Tensor, classes: int) -> Tensor:
    prototypes = []
    for class_index in range(classes):
        members = parents == class_index
        prototype = (text[members][None] * weights[:, members, None]).sum(dim=1)
        prototypes.append(F.normalize(prototype, dim=-1))
    return torch.stack(prototypes, dim=1)


def _cross_context(visual: Tensor, prototypes: Tensor, temperature: float) -> Tensor:
    attention = torch.softmax(torch.einsum("bnd,bcd->bnc", visual, prototypes) / temperature, dim=-1)
    return F.normalize(torch.einsum("bnc,bcd->bnd", attention, prototypes), dim=-1)


def _text_conditioned_relation(
    geometry: Tensor,
    key_probability: Tensor,
    valid: Tensor,
    strength: float,
    *,
    query_probability: Tensor | None = None,
) -> Tensor:
    query_probability = key_probability if query_probability is None else query_probability
    compatibility = query_probability @ key_probability.transpose(1, 2)
    logits = geometry.clamp_min(1e-8).log() + strength * compatibility
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    relation = torch.softmax(logits, dim=-1)
    return torch.where(valid[:, :, None], relation, geometry)


def _blend(base: Tensor, update: Tensor, strength: float) -> Tensor:
    return F.normalize((1.0 - strength) * base + strength * update, dim=-1)


def _parallel(base: Tensor, self_context: Tensor, cross_context: Tensor, strength: float) -> Tensor:
    base_weight = max(1.0 - 2.0 * strength, 0.0)
    return F.normalize(base_weight * base + strength * self_context + strength * cross_context, dim=-1)
