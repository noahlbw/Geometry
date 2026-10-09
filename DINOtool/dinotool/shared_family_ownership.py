"""Frozen family-level ownership of alias evidence across Geometry views."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_ownership import alias_families, positive_excess_readout
from .gear_ov import GearObservations, GearText, class_scores
from .tcpr import TCPRTextBank


@dataclass(frozen=True)
class SharedOwnershipConfig:
    temperature: float = 0.07
    family_cosine: float = 0.97
    graph_temperature: float = 0.15
    graph_strength: float = 0.4
    iterations: int = 5


class SharedFamilyOwnershipReadout:
    def __init__(self, bank: TCPRTextBank,
                 config: SharedOwnershipConfig = SharedOwnershipConfig()):
        bank.validate()
        self.config = config
        self.parents = bank.parent_indices
        self.features = F.normalize(bank.features.float(), dim=-1)
        self.class_count = bank.class_count
        self.families = alias_families(bank, config.family_cosine)
        self.alias_to_family = torch.empty(len(bank.alias_names), dtype=torch.long,
                                           device=self.parents.device)
        self.class_families: list[list[tuple[int, Tensor]]] = []
        for family_index, members in enumerate(self.families):
            self.alias_to_family[members] = family_index
        for class_index in range(self.class_count):
            groups = []
            for family_index, members in enumerate(self.families):
                subset = [member for member in members
                          if int(self.parents[member]) == class_index]
                if subset:
                    groups.append((family_index, torch.tensor(subset, dtype=torch.long,
                                                               device=self.parents.device)))
            self.class_families.append(groups)

    def _reference(self, scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Leave one whole family out and give every remaining family one vote."""
        length, aliases = scores.shape
        families = len(self.families)
        teacher = scores.new_zeros(length, families, self.class_count)
        identifiable = torch.ones(families, dtype=torch.bool, device=scores.device)
        for class_index, groups in enumerate(self.class_families):
            family_scores = torch.stack([
                torch.logsumexp(scores[:, members], dim=-1) - math.log(len(members))
                for _, members in groups
            ], dim=-1)
            count = len(groups)
            if count == 1:
                identifiable[groups[0][0]] = False
                continue
            prefix = torch.logcumsumexp(family_scores, dim=-1)
            suffix = torch.logcumsumexp(family_scores.flip(-1), dim=-1).flip(-1)
            empty = torch.full_like(family_scores[:, :1], -torch.inf)
            excluded = torch.logaddexp(torch.cat((empty, prefix[:, :-1]), dim=-1),
                                       torch.cat((suffix[:, 1:], empty), dim=-1))
            excluded = excluded - math.log(count - 1)
            for position, (family_index, _) in enumerate(groups):
                teacher[:, family_index, class_index] = excluded[:, position]
        if not bool(identifiable.all()):
            teacher[:, ~identifiable] = 0
        reference = teacher[:, self.alias_to_family.to(scores.device), :].gather(
            2, self.parents[None, :, None].expand(length, aliases, 1)
        ).squeeze(-1)
        return teacher, reference, identifiable

    @staticmethod
    def _to_fine(values: Tensor, height: int, width: int) -> Tensor:
        return F.interpolate(values.permute(2, 0, 1)[None], size=(64, 64),
                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    @staticmethod
    def _to_native(values: Tensor, height: int, width: int) -> Tensor:
        return F.interpolate(values.permute(2, 0, 1)[None], size=(height, width),
                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    def _graph(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.normalize(raw.float(), dim=-1)
        vertical = ((normalized[:-1] * normalized[1:]).sum(-1) - 1).div(
            self.config.graph_temperature).exp()
        horizontal = ((normalized[:, :-1] * normalized[:, 1:]).sum(-1) - 1).div(
            self.config.graph_temperature).exp()
        return vertical, horizontal

    def _family_support(self, scores: Tensor, valid: Tensor) -> Tensor:
        responses = torch.stack([scores[:, members].mean(-1)
                                 for members in self.families], dim=-1)
        threshold = torch.quantile(responses.detach()[valid], 0.7, dim=0,
                                   keepdim=True)
        support = (responses - threshold).relu()
        support[~valid] = 0
        return support / support.amax(dim=0, keepdim=True).clamp_min(1e-6)

    def _ownership(self, views: GearObservations, per_view: list[Tensor],
                   teachers: list[Tensor], identifiable: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = []
        for scores, teacher in zip(per_view, teachers):
            height, width = scores.shape[:2]
            posterior = teacher.reshape(height, width, -1, self.class_count).softmax(-1)
            fine = F.interpolate(posterior.permute(2, 3, 0, 1).reshape(
                1, -1, height, width), size=(64, 64), mode="bilinear",
                align_corners=False)[0].reshape(-1, self.class_count, 64, 64)
            probabilities.append(fine.permute(2, 3, 0, 1))
        stacked = torch.stack(probabilities)
        mean = stacked.mean(0)
        entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
        view_entropies = -(stacked * stacked.clamp_min(1e-12).log()).sum(-1).mean(0)
        unknown = ((entropy - view_entropies) / math.log(self.class_count)).clamp(0, 1)
        seed = torch.cat((mean * (1 - unknown[..., None]), unknown[..., None]), -1)
        seed[:, :, ~identifiable, :-1] = 0
        seed[:, :, ~identifiable, -1] = 1

        valid_y = torch.arange(64, device=mean.device) * 8 < views.actual_height
        valid_x = torch.arange(64, device=mean.device) * 8 < views.actual_width
        valid = (valid_y[:, None] & valid_x[None, :]).flatten()
        average_alias = torch.stack([
            self._to_fine(scores, *scores.shape[:2]) for scores in per_view
        ]).mean(0).reshape(-1, self.features.shape[0])
        support = self._family_support(average_alias, valid).reshape(64, 64, -1)
        vertical, horizontal = self._graph(views.detail_raw)
        current = seed
        for _ in range(self.config.iterations):
            neighbors = torch.zeros_like(current)
            mass = torch.zeros_like(support)
            edge_v = self.config.graph_strength * vertical[..., None] * \
                torch.sqrt(support[:-1] * support[1:])
            edge_h = self.config.graph_strength * horizontal[..., None] * \
                torch.sqrt(support[:, :-1] * support[:, 1:])
            neighbors[:-1] += edge_v[..., None] * current[1:]
            neighbors[1:] += edge_v[..., None] * current[:-1]
            mass[:-1] += edge_v
            mass[1:] += edge_v
            neighbors[:, :-1] += edge_h[..., None] * current[:, 1:]
            neighbors[:, 1:] += edge_h[..., None] * current[:, :-1]
            mass[:, :-1] += edge_h
            mass[:, 1:] += edge_h
            current = (support[..., None] * seed + neighbors) / \
                (support + mass).clamp_min(1e-8)[..., None]
            current = torch.where((support + mass)[..., None] > 0, current, seed)
        return current, valid

    def solve(self, views: GearObservations, text: GearText,
              multiscale_logits: Tensor) -> tuple[Tensor, dict[str, float]]:
        if text.features.shape != self.features.shape or not torch.allclose(
                text.features, self.features, atol=1e-5):
            raise ValueError("Text features differ from the frozen full vocabulary.")
        if self.class_count == 1:
            return multiscale_logits, {"mean_rejection": 0.0,
                                       "mean_abs_correction": 0.0,
                                       "mean_unknown": 1.0,
                                       "identifiable_family_fraction": 0.0}
        per_view = [(features.float() @ self.features.T) / self.config.temperature
                    for features in (views.local_aligned, views.detail_aligned,
                                     views.context_aligned)]
        flattened = [scores.reshape(-1, scores.shape[-1]) for scores in per_view]
        references = [self._reference(scores) for scores in flattened]
        identifiable = torch.stack([item[2] for item in references]).all(0)
        ownership, valid = self._ownership(views, per_view,
                                            [item[0] for item in references],
                                            identifiable)
        corrected_views = []
        rejected = []
        for scores, (_, reference, _) in zip(per_view, references):
            height, width, aliases = scores.shape
            native = self._to_native(ownership.reshape(64, 64, -1), height, width)
            native = native.reshape(height * width, -1, self.class_count + 1)
            family = self.alias_to_family.to(native.device)
            parent = self.parents.to(native.device)
            own = native[:, family, :].gather(
                2, parent[None, :, None].expand(height * width, aliases, 1)
            ).squeeze(-1)
            retention = (own + native[:, family, -1]).clamp(0, 1)
            rejected.append((1 - retention).mean())
            corrected = positive_excess_readout(scores.reshape(-1, aliases),
                                                reference, retention, parent,
                                                self.class_count)
            corrected = corrected.reshape(height, width, self.class_count)
            output = F.interpolate((self.config.temperature * corrected).permute(
                2, 0, 1)[None], size=(512, 512), mode="bilinear",
                align_corners=False)
            corrected_views.append(output)
        logits = torch.stack(corrected_views).mean(0)
        return logits, {
            "mean_rejection": float(torch.stack(rejected).mean()),
            "mean_abs_correction": float((logits - multiscale_logits).abs().mean()),
            "mean_unknown": float(ownership.reshape(-1, len(self.families),
                                                    self.class_count + 1)[valid, :, -1].mean()),
            "identifiable_family_fraction": float(identifiable.float().mean()),
        }

    def config_dict(self) -> dict[str, float | int]:
        return asdict(self.config)
