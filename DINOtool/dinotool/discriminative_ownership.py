"""Fixed references and region-contrast rejection for frozen alias evidence.

No target masks enter this module. The contrast margin uses spatial blocks as
observations; it is a fixed dispersion heuristic, not a calibrated confidence
interval. Old shared-family results remain tied to their separate v2 module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_ownership import positive_excess_readout
from .gear_ov import GearObservations, GearText
from .shared_family_ownership import SharedFamilyOwnershipReadout, SharedOwnershipConfig
from .tcpr import TCPRTextBank


IMPLEMENTATION = "geometry-discriminative-ownership-v3-20260930"
METHODS = ("Geometry", "Multiscale", "COR_Fixed", "COR_Discriminative")
DIAGNOSTICS = (
    "fixed_mean_rejection", "discriminative_mean_rejection",
    "fixed_mean_abs_correction", "discriminative_mean_abs_correction",
    "ownership_mean_unknown", "identifiable_family_fraction",
    "supported_pair_fraction", "negative_contrast_pair_fraction",
)


@dataclass(frozen=True)
class DiscriminativeOwnershipConfig(SharedOwnershipConfig):
    anchor_probability: float = 0.75
    block_size: int = 4
    minimum_blocks: int = 2
    dispersion_multiplier: float = 1.96


def ownership_seed(probabilities: Tensor) -> Tensor:
    """Input [view,...,class]; predictive entropy includes shared uncertainty."""
    mean = probabilities.mean(0)
    classes = mean.shape[-1]
    if classes == 1:
        return torch.cat((torch.zeros_like(mean), torch.ones_like(mean)), -1)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
    unknown = (entropy / math.log(classes)).clamp(0, 1)
    return torch.cat((mean * (1 - unknown[..., None]), unknown[..., None]), -1)


def negative_class_contrast(response: Tensor, anchors: Tensor,
                            minimum_blocks: int, dispersion_multiplier: float
                            ) -> tuple[Tensor, Tensor]:
    """response [view,block,family], anchors [block,family,class].

    Return [family,parent,rival] suppression eligibility. Both classes need
    support, and each observed view must favor the rival beyond dispersion.
    """
    weights = anchors.float()
    count = weights.sum(0)
    means = torch.einsum("vbg,bgc->vgc", response, weights) / count.clamp_min(1)
    second = torch.einsum("vbg,bgc->vgc", response.square(), weights)
    variance = ((second - means.square() * count).clamp_min(0)
                / (count - 1).clamp_min(1))
    mean_variance = variance / count.clamp_min(1)
    # axis -2 is the original parent, -1 is the competing class.
    difference = means.unsqueeze(-2) - means.unsqueeze(-1)
    spread = dispersion_multiplier * torch.sqrt(
        mean_variance.unsqueeze(-2) + mean_variance.unsqueeze(-1))
    ratio = (difference - spread).clamp_min(0) / (
        difference.abs() + spread).clamp_min(1e-6)
    classes = count.shape[-1]
    supported = ((count >= minimum_blocks).unsqueeze(-1)
                 & (count >= minimum_blocks).unsqueeze(-2))
    supported &= ~torch.eye(classes, dtype=torch.bool, device=count.device)[None]
    harm = ratio.amin(0) * supported.float()
    return harm, supported


class DiscriminativeOwnershipReadout(SharedFamilyOwnershipReadout):
    def __init__(self, bank: TCPRTextBank,
                 config: DiscriminativeOwnershipConfig = DiscriminativeOwnershipConfig()):
        super().__init__(bank, config)
        if 64 % config.block_size or config.minimum_blocks < 2:
            raise ValueError("Invalid fixed contrast block configuration.")

    def _reference(self, scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        length, aliases = scores.shape
        families = len(self.families)
        teacher = scores.new_empty(length, families, self.class_count)
        identifiable = torch.ones(families, dtype=torch.bool, device=scores.device)
        for class_index, groups in enumerate(self.class_families):
            family_scores = torch.stack([
                torch.logsumexp(scores[:, members], -1) - math.log(len(members))
                for _, members in groups
            ], -1)
            count = len(groups)
            # Families absent from this class must receive its FULL evidence.
            full = torch.logsumexp(family_scores, -1) - math.log(count)
            teacher[:, :, class_index] = full[:, None]
            if count == 1:
                identifiable[groups[0][0]] = False
                continue
            prefix = torch.logcumsumexp(family_scores, -1)
            suffix = torch.logcumsumexp(family_scores.flip(-1), -1).flip(-1)
            empty = torch.full_like(family_scores[:, :1], -torch.inf)
            excluded = torch.logaddexp(torch.cat((empty, prefix[:, :-1]), -1),
                                       torch.cat((suffix[:, 1:], empty), -1))
            excluded -= math.log(count - 1)
            for position, (family_index, _) in enumerate(groups):
                teacher[:, family_index, class_index] = excluded[:, position]
        teacher[:, ~identifiable] = 0
        reference = teacher[:, self.alias_to_family, :].gather(
            2, self.parents[None, :, None].expand(length, aliases, 1)).squeeze(-1)
        return teacher, reference, identifiable

    @staticmethod
    def _resize(values: Tensor, size: tuple[int, int]) -> Tensor:
        height, width = values.shape[:2]
        tail = values.shape[2:]
        channels = values.reshape(height, width, -1).permute(2, 0, 1)[None]
        resized = F.interpolate(channels, size=size, mode="bilinear", align_corners=False)
        return resized[0].permute(1, 2, 0).reshape(*size, *tail)

    def _propagate(self, seed: Tensor, support: Tensor, raw: Tensor) -> Tensor:
        vertical, horizontal = self._graph(raw)
        edge_v = self.config.graph_strength * vertical[..., None] * \
            torch.sqrt(support[:-1] * support[1:])
        edge_h = self.config.graph_strength * horizontal[..., None] * \
            torch.sqrt(support[:, :-1] * support[:, 1:])
        mass = torch.zeros_like(support)
        mass[:-1] += edge_v
        mass[1:] += edge_v
        mass[:, :-1] += edge_h
        mass[:, 1:] += edge_h
        current = seed
        for _ in range(self.config.iterations):
            neighbors = torch.zeros_like(current)
            neighbors[:-1] += edge_v[..., None] * current[1:]
            neighbors[1:] += edge_v[..., None] * current[:-1]
            neighbors[:, :-1] += edge_h[..., None] * current[:, 1:]
            neighbors[:, 1:] += edge_h[..., None] * current[:, :-1]
            updated = (support[..., None] * seed + neighbors) / \
                (support + mass).clamp_min(1e-8)[..., None]
            current = torch.where((support + mass)[..., None] > 0, updated, seed)
        return current

    def _block_contrast(self, fine_scores: Tensor, probabilities: Tensor,
                        valid: Tensor, identifiable: Tensor) -> tuple[Tensor, Tensor]:
        # Every tensor below is image-only and independent of the target mask.
        block = self.config.block_size
        count = 64 // block
        def pool(values):
            return values.reshape(values.shape[0], count, block, count, block,
                                  *values.shape[3:]).mean((2, 4)).flatten(1, 2)
        q = pool(probabilities)
        anchor = (q >= self.config.anchor_probability).all(0)
        block_valid = valid.reshape(count, block, count, block).all(3).all(1).flatten()
        anchor &= block_valid[:, None, None] & identifiable[None, :, None]
        # Remove a pixel's common response to the full fixed vocabulary.
        centered = fine_scores - fine_scores.mean(-1, keepdim=True)
        family_response = torch.stack([
            torch.logsumexp(centered[..., members], -1) - math.log(len(members))
            for members in self.families
        ], -1)
        return negative_class_contrast(pool(family_response), anchor,
                                       self.config.minimum_blocks,
                                       self.config.dispersion_multiplier)

    @torch.no_grad()
    def solve_all(self, views: GearObservations, text: GearText,
                  multiscale_logits: Tensor) -> tuple[dict[str, Tensor], dict[str, float]]:
        if (text.features.shape != self.features.shape
                or not torch.allclose(text.features, self.features, atol=1e-5)
                or not torch.equal(text.parents, self.parents)):
            raise ValueError("Frozen vocabulary differs from the observation bank.")
        if self.class_count == 1:
            diagnostics = {key: 0.0 for key in DIAGNOSTICS}
            diagnostics["ownership_mean_unknown"] = 1.0
            return {name: multiscale_logits for name in METHODS[2:]}, diagnostics
        native_scores = [(features.float() @ self.features.T) / self.config.temperature
                         for features in (views.local_aligned, views.detail_aligned,
                                          views.context_aligned)]
        references = [self._reference(scores.flatten(0, 1)) for scores in native_scores]
        identifiable = references[0][2]
        fine_scores = torch.stack([self._resize(scores, (64, 64)) for scores in native_scores])
        probabilities = torch.stack([
            self._resize(teacher.reshape(*scores.shape[:2], len(self.families),
                                         self.class_count).softmax(-1), (64, 64))
            for scores, (teacher, _, _) in zip(native_scores, references)
        ])
        seed = ownership_seed(probabilities)
        seed[:, :, ~identifiable, :-1] = 0
        seed[:, :, ~identifiable, -1] = 1
        y = torch.arange(64, device=seed.device) * 8 < views.actual_height
        x = torch.arange(64, device=seed.device) * 8 < views.actual_width
        valid = y[:, None] & x[None, :]
        support = self._family_support(fine_scores.mean(0).flatten(0, 1),
                                       valid.flatten()).reshape(64, 64, -1)
        ownership = self._propagate(seed, support, views.detail_raw)
        # No graph may turn an unidentifiable family into asserted evidence.
        ownership[:, :, ~identifiable, :-1] = 0
        ownership[:, :, ~identifiable, -1] = 1
        harm, supported = self._block_contrast(fine_scores, probabilities, valid, identifiable)
        corrected_views = {name: [] for name in METHODS[2:]}
        rejection = {name: [] for name in METHODS[2:]}
        families = self.alias_to_family
        parents = self.parents
        pair_harm = harm[families, parents, :]
        for scores, (_, reference, _) in zip(native_scores, references):
            height, width, aliases = scores.shape
            native = self._resize(ownership, (height, width)).reshape(
                -1, len(self.families), self.class_count + 1)
            alias_ownership = native[:, families, :]
            own = alias_ownership.gather(
                2, parents[None, :, None].expand(height * width, aliases, 1)).squeeze(-1)
            fixed = (own + alias_ownership[..., -1]).clamp(0, 1)
            disputed = (alias_ownership[..., :-1] * pair_harm[None]).sum(-1)
            contrast = (1 - disputed).clamp(0, 1)
            valid_y = torch.arange(height, device=scores.device) * (512 / height) < views.actual_height
            valid_x = torch.arange(width, device=scores.device) * (512 / width) < views.actual_width
            native_valid = (valid_y[:, None] & valid_x[None, :]).flatten()
            for method, retention in zip(METHODS[2:], (fixed, contrast)):
                rejection[method].append((1 - retention[native_valid]).mean())
                corrected = positive_excess_readout(scores.reshape(-1, aliases), reference,
                                                    retention, parents, self.class_count)
                corrected = corrected.reshape(height, width, self.class_count)
                corrected_views[method].append(self.config.temperature * self._resize(
                    corrected, (512, 512)).permute(2, 0, 1)[None])
        logits = {method: torch.stack(values).mean(0)
                  for method, values in corrected_views.items()}
        diagnostics = {
            "ownership_mean_unknown": float(ownership[valid][..., -1].mean()),
            "identifiable_family_fraction": float(identifiable.float().mean()),
            "supported_pair_fraction": float(supported.float().mean()),
            "negative_contrast_pair_fraction": float((harm > 0).float().mean()),
        }
        for prefix, method in zip(("fixed", "discriminative"), METHODS[2:]):
            diagnostics[prefix + "_mean_rejection"] = float(torch.stack(rejection[method]).mean())
            diagnostics[prefix + "_mean_abs_correction"] = float(
                (logits[method] - multiscale_logits)[..., :views.actual_height,
                                                    :views.actual_width].abs().mean())
        return logits, diagnostics
