"""Actual support crops verify signed alias replacements in Geometry readouts.

The native global branch verifies ownership only. Dense proposal scores use
the same frozen Geometry head and text bank as the unchanged Multiscale path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_ownership import alias_families
from .gear_ov import GearObservations, GearOVSegmenter, _crop_at, class_scores
from .prompts import REMOTE_SENSING_TEMPLATES
from .tcpr import TCPRTextBank


IMPLEMENTATION = "geometry-support-conditioned-signed-alias-v1-20261001"
METHODS = ("Geometry", "Multiscale", "SupportConditioned")
DIAGNOSTICS = (
    "regions_proposed", "regions_observed", "extra_backbone_calls",
    "support_coverage", "resolved_alias_fraction", "positive_replacement_mean",
    "negative_replacement_mean", "mean_abs_class_delta", "changed_patch_fraction",
    "partition_seconds", "crop_seconds", "replacement_seconds",
)


@dataclass(frozen=True)
class SupportConfig:
    segments: int = 32
    maximum_regions: int = 4
    minimum_region_patches: int = 4
    minimum_crop_pixels: int = 64
    crop_margin_patches: int = 2
    crop_resolution: int = 512
    wide_extent_multiplier: int = 2
    family_cosine: float = 0.97
    alias_temperature: float = 0.07
    geometry_temperature: float = 0.1
    transport_radius_patches: int = 2
    query_chunk: int = 256

    def validate(self) -> None:
        counts = (self.segments, self.maximum_regions, self.minimum_region_patches,
                  self.minimum_crop_pixels, self.crop_resolution,
                  self.wide_extent_multiplier, self.query_chunk)
        if min(counts) < 1 or self.maximum_regions > self.segments:
            raise ValueError("Invalid support or crop budget.")
        if self.crop_resolution != 512 or self.minimum_crop_pixels % 16:
            raise ValueError("Support crops use the fixed 512/patch16 Geometry reader.")
        if self.crop_margin_patches < 0 or self.transport_radius_patches < 1:
            raise ValueError("Invalid crop margin or transport radius.")
        if not 0 <= self.family_cosine <= 1 or min(
                self.alias_temperature, self.geometry_temperature) <= 0:
            raise ValueError("Invalid text or geometry parameters.")


@dataclass(frozen=True)
class SupportRegion:
    label: int
    mask: Tensor
    top: int
    left: int
    extent: int


@dataclass(frozen=True)
class SupportObservation:
    region: SupportRegion
    proposal_features: Tensor
    usable: Tensor
    tight_global: Tensor
    wide_global: Tensor


def resize_grid(values: Tensor, shape: tuple[int, int], *, mode: str = "bilinear") -> Tensor:
    options = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(values.permute(2, 0, 1)[None].float(), size=shape,
                         mode=mode, **options)[0].permute(1, 2, 0)


def crop_geometry(mask: Tensor, config: SupportConfig, stride: int = 8
                  ) -> tuple[int, int, int]:
    """Square pixel crop enclosing the actual support, not its fixed quadrant."""
    positions = mask.nonzero(as_tuple=False)
    if not len(positions):
        raise ValueError("Cannot crop an empty support.")
    lower, upper = positions.amin(0), positions.amax(0) + 1
    size = (upper - lower) * stride + 2 * config.crop_margin_patches * stride
    extent = max(config.minimum_crop_pixels, int(size.max()))
    extent = math.ceil(extent / 16) * 16
    center = (lower + upper).float() * (stride / 2)
    return math.floor(float(center[0]) - extent / 2), \
        math.floor(float(center[1]) - extent / 2), extent


def partition_supports(raw: Tensor, valid: Tensor, config: SupportConfig) -> Tensor:
    """SLIC-zero partitions the original normalized DINO feature grid."""
    from skimage.segmentation import slic

    if raw.ndim != 3 or valid.shape != raw.shape[:2] or not bool(valid.any()):
        raise ValueError("Expected a nonempty raw feature grid and valid support.")
    normalized = F.normalize(raw.detach().float(), dim=-1)
    labels = slic(normalized.cpu().numpy(), n_segments=min(config.segments, int(valid.sum())),
                  compactness=0.1, slic_zero=True, convert2lab=False,
                  channel_axis=-1, start_label=1, enforce_connectivity=True,
                  mask=valid.cpu().numpy(), max_num_iter=10)
    labels = torch.as_tensor(np.asarray(labels).copy(), device=raw.device,
                             dtype=torch.long) - 1
    return labels.masked_fill(~valid, -1)


def choose_regions(labels: Tensor, logits: Tensor, config: SupportConfig
                   ) -> list[SupportRegion]:
    """Mean ambiguity, with one support per predicted class before duplicates.

    No area prior or fixed top-confidence class prototype is constructed.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Support labels and class grid must match.")
    probability = (logits.float() / config.alias_temperature).softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    predicted = probability.argmax(-1)
    candidates = []
    for label in torch.unique(labels[labels >= 0]).tolist():
        mask = labels == label
        if int(mask.sum()) < config.minimum_region_patches:
            continue
        count = torch.bincount(predicted[mask], minlength=logits.shape[-1])
        candidates.append((float(entropy[mask].mean()), int(count.argmax()), label, mask))
    candidates.sort(key=lambda item: (-item[0], item[2]))
    selected, duplicate, seen = [], [], set()
    for item in candidates:
        if item[1] in seen:
            duplicate.append(item)
        else:
            selected.append(item)
            seen.add(item[1])
    selected = (selected + duplicate)[:config.maximum_regions]
    return [SupportRegion(label, mask, *crop_geometry(mask, config))
            for _, _, label, mask in selected]


def family_exclusion_layout(parents: Tensor, families: list[list[int]], classes: int
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    family_index = torch.full_like(parents, -1)
    for index, family in enumerate(families):
        family_index[family] = index
    if bool((family_index < 0).any()):
        raise ValueError("Every alias must belong to a verifier family.")
    keep_family = torch.arange(len(families), device=parents.device)[:, None] != family_index[None]
    keep_class = torch.arange(classes, device=parents.device)[:, None] == parents[None]
    keep = keep_family[:, None] & keep_class[None]
    counts = keep.sum(-1)
    available = (counts > 0).all(-1)
    return keep, counts.clamp_min(1).float().log(), family_index, available


def family_excluded_margins(global_scores: Tensor, parents: Tensor,
                            families: list[list[int]], classes: int,
                            layout=None) -> tuple[Tensor, Tensor]:
    """Parent versus strongest rival, excluding the judged family in every class.

    A family that removes any entire class has no identifiable verifier margin.
    The logsumexp implementation avoids subtracting nearly equal exp sums.
    """
    if global_scores.shape[-1] != parents.numel() or classes < 2:
        raise ValueError("Invalid family-verifier bank.")
    keep, log_count, family_index, available = (
        family_exclusion_layout(parents, families, classes) if layout is None else layout)
    scores = torch.logsumexp(global_scores[..., None, None, :].masked_fill(~keep, -torch.inf), -1)
    scores = torch.where(available[:, None], scores - log_count, 0)
    reference = scores[..., family_index, :]
    parent_index = parents[:, None].expand(*reference.shape[:-2], -1, 1)
    own = reference.gather(-1, parent_index)[..., 0]
    rivals = reference.masked_fill(F.one_hot(parents, classes).bool(), -torch.inf).amax(-1)
    return (own - rivals) * available[family_index], available[family_index]


def consensus_states(tight_margin: Tensor, wide_margin: Tensor,
                     identifiable: Tensor) -> tuple[Tensor, Tensor]:
    """Positive, negative and implicit unresolved evidence; no forced top-K."""
    if tight_margin.shape != wide_margin.shape or tight_margin.shape[-1] != identifiable.numel():
        raise ValueError("Tight/wide family margins must match.")
    positive = torch.minimum(tight_margin, wide_margin).clamp_min(0).tanh()
    negative = (-torch.maximum(tight_margin, wide_margin)).clamp_min(0).tanh()
    return positive * identifiable, negative * identifiable


def signed_replacement(original: Tensor, proposal: Tensor, positive: Tensor,
                       negative: Tensor, usable: Tensor) -> Tensor:
    """A supported increase or decrease; unresolved aliases retain their slots."""
    if original.shape != proposal.shape or usable.shape != original.shape[:-1]:
        raise ValueError("Proposal, original and spatial support must match.")
    if positive.shape != negative.shape or positive.shape[-1] != original.shape[-1]:
        raise ValueError("Alias verifier state must match the readout bank.")
    difference = proposal - original
    confidence = torch.where(difference >= 0, positive, negative).clamp(0, 1)
    return original + torch.where(usable[..., None], confidence * difference, 0)


def transport_crop_features(raw: Tensor, region: SupportRegion, source_raw: Tensor,
                            source_aligned: Tensor, actual_shape: tuple[int, int],
                            config: SupportConfig) -> tuple[Tensor, Tensor]:
    """Match near the exact re-encoded coordinate, retaining only region donors."""
    if source_raw.shape[:2] != source_aligned.shape[:2] or raw.ndim != 3:
        raise ValueError("Invalid cross-view feature grids.")
    device = raw.device
    sh, sw = source_raw.shape[:2]
    gy, gx = torch.meshgrid(torch.arange(sh, device=device),
                            torch.arange(sw, device=device), indexing="ij")
    py = region.top + (gy.flatten() + 0.5) * region.extent / sh
    px = region.left + (gx.flatten() + 0.5) * region.extent / sw
    iy, ix = (py / 8).floor().long(), (px / 8).floor().long()
    valid_source = ((py >= 0) & (py < actual_shape[0]) & (px >= 0)
                    & (px < actual_shape[1]))
    inside = ((iy >= 0) & (iy < raw.shape[0]) & (ix >= 0) & (ix < raw.shape[1]))
    safe_y, safe_x = iy.clamp(0, raw.shape[0] - 1), ix.clamp(0, raw.shape[1] - 1)
    valid_source &= inside & region.mask[safe_y, safe_x]
    source_unit = F.normalize(source_raw.reshape(-1, source_raw.shape[-1]).float(), dim=-1)
    aligned = source_aligned.reshape(-1, source_aligned.shape[-1]).float()
    query_indices = region.mask.nonzero(as_tuple=False)
    proposal = raw.new_zeros((*raw.shape[:2], aligned.shape[-1]), dtype=torch.float32)
    usable = torch.zeros(raw.shape[:2], dtype=torch.bool, device=device)
    radius = config.transport_radius_patches * region.extent / sh
    for start in range(0, len(query_indices), config.query_chunk):
        coords = query_indices[start:start + config.query_chunk]
        qy, qx = (coords[:, 0] + 0.5) * 8, (coords[:, 1] + 0.5) * 8
        dy, dx = qy[:, None] - py[None], qx[:, None] - px[None]
        allowed = valid_source[None] & (dy.abs() <= radius) & (dx.abs() <= radius)
        unit = F.normalize(raw[coords[:, 0], coords[:, 1]].float(), dim=-1)
        score = unit @ source_unit.T / config.geometry_temperature
        score -= (dy.square() + dx.square()) / (2 * radius**2)
        score = score.masked_fill(~allowed, -torch.inf)
        available = allowed.any(-1)
        # An absent donor never becomes a uniform softmax over padded features.
        score = torch.where(available[:, None], score, torch.zeros_like(score))
        weights = score.softmax(-1) * allowed
        transported = F.normalize(weights @ aligned, dim=-1)
        proposal[coords[:, 0], coords[:, 1]] = transported
        usable[coords[:, 0], coords[:, 1]] = available
    return proposal, usable


class SupportConditionedReadout:
    def __init__(self, bank: TCPRTextBank, backbone,
                 config: SupportConfig = SupportConfig()):
        config.validate()
        bank.validate()
        self.bank, self.config = bank, config
        self.families = alias_families(bank, config.family_cosine)
        self.family_layout = family_exclusion_layout(bank.parent_indices, self.families, bank.class_count)
        self.full_text = self._encode_full_text(backbone)

    def _encode_full_text(self, backbone) -> Tensor:
        prompts = [template.format(label=alias) for alias in self.bank.alias_names
                   for template in REMOTE_SENSING_TEMPLATES]
        encoded = []
        for start in range(0, len(prompts), 64):
            tokens = backbone.tokenize(prompts[start:start + 64]).to(backbone.device)
            with torch.inference_mode(), backbone._autocast():
                features = backbone.model.encode_text(tokens, normalize=False)
            encoded.append(F.normalize(features.float(), dim=-1))
        vectors = torch.cat(encoded).reshape(len(self.bank.alias_names),
                                             len(REMOTE_SENSING_TEMPLATES), -1).mean(1)
        return F.normalize(vectors, dim=-1)

    @torch.inference_mode()
    def solve(self, views: GearObservations, baseline: Tensor,
              observations: list[SupportObservation]) -> tuple[Tensor, dict[str, float]]:
        started = time.perf_counter()
        if not observations:
            return baseline, {name: 0.0 for name in (
                "support_coverage", "resolved_alias_fraction", "positive_replacement_mean",
                "negative_replacement_mean", "mean_abs_class_delta", "changed_patch_fraction",
                "replacement_seconds")}
        if any(obs.tight_global.numel() != self.full_text.shape[-1]
               or obs.wide_global.numel() != self.full_text.shape[-1] for obs in observations):
            raise ValueError("Native global verification must use the full text embedding.")
        temperature = self.config.alias_temperature
        states = []
        for obs in observations:
            scores = torch.stack((obs.tight_global, obs.wide_global)) @ self.full_text.T / temperature
            margins, identifiable = family_excluded_margins(
                scores, self.bank.parent_indices, self.families, self.bank.class_count,
                self.family_layout)
            states.append(consensus_states(margins[0], margins[1], identifiable))
        final_delta = torch.zeros_like(baseline)
        positive_sum, negative_sum, count = 0.0, 0.0, 0
        for features in (views.local_aligned, views.detail_aligned, views.context_aligned):
            original = features.float() @ self.bank.features.float().T
            corrected = original.clone()
            for obs, (positive, negative) in zip(observations, states):
                mass = resize_grid(obs.usable[..., None].float(), original.shape[:2])
                proposal = resize_grid((obs.proposal_features @ self.bank.features.float().T)
                                       * obs.usable[..., None], original.shape[:2])
                proposal = proposal / mass.clamp_min(1e-12)
                usable = resize_grid(obs.usable[..., None].float(), original.shape[:2],
                                     mode="nearest-exact")[..., 0].bool() & (mass[..., 0] > 0)
                # Supports are disjoint; each alias slot gets at most one observation.
                updated = signed_replacement(original, proposal, positive, negative, usable)
                corrected = torch.where(usable[..., None], updated, corrected)
            difference = corrected - original
            positive_sum += float(difference.clamp_min(0).sum())
            negative_sum += float((-difference).clamp_min(0).sum())
            count += difference.numel()
            class_delta = temperature * (
                class_scores(corrected / temperature, self.bank.parent_indices, self.bank.class_count)
                - class_scores(original / temperature, self.bank.parent_indices, self.bank.class_count))
            final_delta += resize_grid(class_delta, (512, 512)).permute(2, 0, 1)[None] / 3
        final = baseline + final_delta
        valid_y = torch.arange(64, device=baseline.device) * 8 < views.actual_height
        valid_x = torch.arange(64, device=baseline.device) * 8 < views.actual_width
        valid = valid_y[:, None] & valid_x[None]
        union = torch.stack([obs.usable for obs in observations]).any(0)
        before = F.interpolate(baseline, size=(64, 64), mode="bilinear", align_corners=False).argmax(1)
        after = F.interpolate(final, size=(64, 64), mode="bilinear", align_corners=False).argmax(1)
        return final, {
            "support_coverage": float((union & valid).sum() / valid.sum()),
            "resolved_alias_fraction": float(torch.stack([p + n for p, n in states]).gt(0).float().mean()),
            "positive_replacement_mean": positive_sum / count,
            "negative_replacement_mean": negative_sum / count,
            "mean_abs_class_delta": float(final_delta.abs().mean()),
            "changed_patch_fraction": float(((before != after) & valid[None]).sum() / valid.sum()),
            "replacement_seconds": time.perf_counter() - started,
        }


class SupportConditionedGeometry:
    def __init__(self, backbone, config: SupportConfig = SupportConfig()):
        config.validate()
        self.backbone, self.config = backbone, config
        self.base = GearOVSegmenter(backbone)

    def config_dict(self) -> dict[str, object]:
        return {"support": asdict(self.config), "gear": self.base.config_dict(),
                "verifier_templates": list(REMOTE_SENSING_TEMPLATES),
                "verifier_representation": "full native CLS + pooled patch / full text"}

    @torch.inference_mode()
    def observe(self, image: Tensor, top: int, left: int, views: GearObservations,
                priority_logits: Tensor) -> tuple[list[SupportObservation], dict[str, float]]:
        started = time.perf_counter()
        valid_y = torch.arange(64, device=views.detail_raw.device) * 8 < views.actual_height
        valid_x = torch.arange(64, device=views.detail_raw.device) * 8 < views.actual_width
        valid = valid_y[:, None] & valid_x[None]
        labels = partition_supports(views.detail_raw, valid, self.config)
        logits = F.interpolate(priority_logits.float(), size=(64, 64), mode="bilinear",
                               align_corners=False)[0].permute(1, 2, 0)
        regions = choose_regions(labels, logits, self.config)
        partition_seconds = time.perf_counter() - started
        observations = []
        for region in regions:
            rgb = _crop_at(image, top + region.top, left + region.left, region.extent)
            tight = self.base.base.prepare_image(F.interpolate(
                rgb.to(self.backbone.device), size=(512, 512), mode="bilinear", align_corners=False))
            wide_extent = self.config.wide_extent_multiplier * region.extent
            wide_top = top + region.top - (wide_extent - region.extent) // 2
            wide_left = left + region.left - (wide_extent - region.extent) // 2
            wide_rgb = _crop_at(image, wide_top, wide_left, wide_extent)
            wide = self.base.base.prepare_image(F.interpolate(
                wide_rgb.to(self.backbone.device), size=(512, 512), mode="bilinear", align_corners=False))
            features, usable = transport_crop_features(
                views.detail_raw, region,
                tight.raw_patch_tokens[0].reshape(32, 32, -1),
                tight.geometry_projected[0].reshape(32, 32, -1),
                (views.actual_height, views.actual_width), self.config)
            observations.append(SupportObservation(region, features, usable,
                                                    tight.native_global[0], wide.native_global[0]))
            del tight, wide
        return observations, {
            "regions_proposed": float((torch.unique(labels[labels >= 0])).numel()),
            "regions_observed": float(len(observations)),
            "extra_backbone_calls": float(2 * len(observations)),
            "partition_seconds": partition_seconds,
            "crop_seconds": time.perf_counter() - started - partition_seconds,
        }
