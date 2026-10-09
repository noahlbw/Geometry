"""Frozen, geometry-grounded competitive ownership of expanded text evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .gear_ov import GearObservations, GearText, class_scores
from .tcpr import TCPRTextBank


@dataclass(frozen=True)
class OwnershipConfig:
    temperature: float = 0.07
    family_cosine: float = 0.97
    semantic_temperature: float = 0.15
    graph_temperature: float = 0.15
    graph_strength: float = 0.4
    iterations: int = 5


def alias_families(bank: TCPRTextBank, threshold: float) -> list[list[int]]:
    """Collapse only nearly identical text directions, regardless of parent class."""
    features = F.normalize(bank.features.float(), dim=-1)
    similarities = features @ features.T
    count = len(bank.alias_names)
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(count):
        for right in range(left + 1, count):
            if bank.alias_names[left].casefold() == bank.alias_names[right].casefold() \
                    or float(similarities[left, right]) >= threshold:
                parent[root(right)] = root(left)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def positive_excess_readout(scaled: Tensor, reference: Tensor,
                            retention: Tensor, parents: Tensor,
                            class_count: int) -> Tensor:
    """Preserve weak slots and the original class denominator.

    All values are in units divided by the alias temperature. The class
    reference is indexed per alias and comes from a family-excluded bank.
    """
    if scaled.shape != reference.shape or scaled.shape != retention.shape:
        raise ValueError("Alias, reference and retention maps must match.")
    excess = (scaled - reference).clamp_min(0)
    corrected = scaled - excess + torch.log1p(
        retention.clamp(0, 1) * torch.expm1(excess)
    )
    return class_scores(corrected, parents, class_count)


class CompetitiveOwnershipReadout:
    def __init__(self, bank: TCPRTextBank, config: OwnershipConfig = OwnershipConfig()):
        bank.validate()
        self.config = config
        self.bank = bank
        self.families = alias_families(bank, config.family_cosine)
        self.parents = bank.parent_indices
        self.features = F.normalize(bank.features.float(), dim=-1)
        prototypes = []
        for class_index in range(bank.class_count):
            canonical = (self.parents == class_index) & bank.canonical_mask
            prototypes.append(self.features[canonical].mean(dim=0))
        prototypes = F.normalize(torch.stack(prototypes), dim=-1)
        semantic = (self.features @ prototypes.T / config.semantic_temperature).softmax(-1)
        source = F.one_hot(self.parents, bank.class_count).float()
        self.prior = 0.5 * semantic + 0.5 * source

    def _family_reference(self, scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Fixed leave-family-out logits, per-alias baseline and identifiability."""
        patches, aliases = scores.shape
        classes = self.bank.class_count
        teacher = scores.new_empty(patches, aliases, classes)
        reference = scores.new_empty(patches, aliases)
        identifiable = torch.ones(aliases, dtype=torch.bool, device=scores.device)
        for family in self.families:
            excluded = torch.zeros(aliases, dtype=torch.bool, device=scores.device)
            excluded[family] = True
            per_class = []
            for class_index in range(classes):
                subset = scores[:, (self.parents == class_index) & ~excluded]
                if subset.shape[1] == 0:
                    identifiable[family] = False
                    subset = scores[:, self.parents == class_index]
                per_class.append(torch.logsumexp(subset, dim=-1) - math.log(subset.shape[-1]))
            logits = torch.stack(per_class, dim=-1)
            teacher[:, family] = logits[:, None]
            for index in family:
                reference[:, index] = logits[:, int(self.parents[index])]
        return teacher, reference, identifiable

    def _graph(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.normalize(raw.float(), dim=-1)
        vertical = ((normalized[:-1] * normalized[1:]).sum(-1) - 1).div(
            self.config.graph_temperature).exp()
        horizontal = ((normalized[:, :-1] * normalized[:, 1:]).sum(-1) - 1).div(
            self.config.graph_temperature).exp()
        return vertical, horizontal

    def solve(self, views: GearObservations, text: GearText,
              multiscale_logits: Tensor) -> tuple[Tensor, dict[str, float]]:
        temperature = self.config.temperature
        if text.features.shape != self.features.shape:
            raise ValueError("Text bank differs from the frozen full vocabulary.")
        local = (views.local_aligned.float() @ self.features.T) / temperature
        detail = (views.detail_aligned.float() @ self.features.T) / temperature
        context = (views.context_aligned.float() @ self.features.T) / temperature
        local_fine = F.interpolate(local.permute(2, 0, 1)[None], size=(64, 64),
                                   mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        context_fine = F.interpolate(context.permute(2, 0, 1)[None], size=(64, 64),
                                     mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        scaled = ((local_fine + detail + context_fine) / 3).reshape(-1, len(self.bank.alias_names))
        teacher, reference, identifiable = self._family_reference(scaled)
        scores = teacher.softmax(-1)
        two = scores.topk(min(2, self.bank.class_count), dim=-1).values
        confidence = (two[..., 0] - two[..., -1]).clamp(0, 1)
        prior = self.prior[None].to(scores.device)
        assigned = (scores * prior).div((scores * prior).sum(-1, keepdim=True).clamp_min(1e-8))
        assigned = assigned * confidence[..., None]
        unresolved = (1 - confidence)[..., None]
        seed = torch.cat((assigned, unresolved), dim=-1)
        seed[:, ~identifiable, :-1] = 0
        seed[:, ~identifiable, -1] = 1

        # Support is relative to an alias's own image footprint, so rare
        # positive phrases are not penalized by their fraction of the image.
        valid_y = torch.arange(64, device=scaled.device) * 8 < views.actual_height
        valid_x = torch.arange(64, device=scaled.device) * 8 < views.actual_width
        valid = (valid_y[:, None] & valid_x[None, :]).flatten()
        quantile = torch.quantile(scaled.detach()[valid], 0.7, dim=0, keepdim=True)
        support = (scaled - quantile).relu()
        support[~valid] = 0
        support = support / support.amax(dim=0, keepdim=True).clamp_min(1e-6)
        current = seed.reshape(64, 64, -1, self.bank.class_count + 1)
        h = support.reshape(64, 64, -1)
        vertical, horizontal = self._graph(views.detail_raw)
        for _ in range(self.config.iterations):
            neighbors = torch.zeros_like(current)
            mass = torch.zeros_like(h)
            edge_v = self.config.graph_strength * vertical[..., None] * \
                torch.sqrt(h[:-1] * h[1:])
            edge_h = self.config.graph_strength * horizontal[..., None] * \
                torch.sqrt(h[:, :-1] * h[:, 1:])
            neighbors[:-1] += edge_v[..., None] * current[1:]
            neighbors[1:] += edge_v[..., None] * current[:-1]
            mass[:-1] += edge_v
            mass[1:] += edge_v
            neighbors[:, :-1] += edge_h[..., None] * current[:, 1:]
            neighbors[:, 1:] += edge_h[..., None] * current[:, :-1]
            mass[:, :-1] += edge_h
            mass[:, 1:] += edge_h
            current = (h[..., None] * seed.reshape_as(current) + neighbors) / \
                (h + mass).clamp_min(1e-8)[..., None]
            current = torch.where((h + mass)[..., None] > 0, current,
                                  seed.reshape_as(current))
        ownership = current.reshape(-1, len(self.bank.alias_names), self.bank.class_count + 1)
        own = ownership.gather(2, self.parents[None, :, None].expand(
            scaled.shape[0], -1, 1)).squeeze(-1)
        retention = (own + ownership[..., -1]).clamp(0, 1)
        corrected = positive_excess_readout(scaled, reference, retention, self.parents,
                                            self.bank.class_count)
        original = class_scores(scaled, self.parents, self.bank.class_count)
        correction = temperature * (corrected - original)
        correction = correction.reshape(64, 64, -1).permute(2, 0, 1)[None]
        correction = F.interpolate(correction, size=(512, 512), mode="bilinear",
                                   align_corners=False)
        logits = multiscale_logits + correction
        diagnostics = {
            "mean_rejection": float((1 - retention[valid]).mean()),
            "mean_abs_correction": float(correction.abs().mean()),
            "identifiable_alias_fraction": float(identifiable.float().mean()),
        }
        return logits, diagnostics
