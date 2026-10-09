"""Connected-context OVSS with two frozen DINO.text observations.

The local route reads the original high-resolution tile. The context route
re-encodes RGB crops around visually connected supports. Raw DINO features only
define where evidence may travel; they never create class scores themselves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import CheckpointConfig
from .model import DINOTextSegmenter
from .prompts import ClassSpec
from .structure_interfaces import StructureInterfaceConfig, StructurePartition, build_structure_interfaces


@dataclass(frozen=True)
class ConnectedContextOVConfig:
    atom_similarity_threshold: float = 0.85
    minimum_context_patches: int = 4
    maximum_context_regions: int = 32
    context_size: int = 256
    context_margin: float = 0.25
    context_batch_size: int = 8
    context_anchor_weight: float = 0.25
    branch_temperature: float = 0.07
    local_route_weight: float = 1.0
    context_route_weight: float = 1.0
    region_prior_strength: float = 0.75
    continuity_strength: float = 0.35
    continuity_iterations: int = 3
    structure_temperature: float = 0.10
    cross_support_weight: float = 0.02

    def validate(self, patch_size: int = 16) -> None:
        if not -1.0 <= self.atom_similarity_threshold <= 1.0:
            raise ValueError("atom_similarity_threshold must be in [-1, 1].")
        if min(self.minimum_context_patches, self.maximum_context_regions, self.context_batch_size) < 1:
            raise ValueError("Context region counts and batch size must be positive.")
        if self.context_size < patch_size or self.context_size % patch_size:
            raise ValueError("context_size must be divisible by the DINO patch size.")
        if self.context_margin < 0:
            raise ValueError("context_margin must be non-negative.")
        if not 0.0 <= self.context_anchor_weight <= 1.0:
            raise ValueError("context_anchor_weight must be in [0, 1].")
        if self.branch_temperature <= 0 or self.structure_temperature <= 0:
            raise ValueError("Temperatures must be positive.")
        if self.local_route_weight <= 0 or self.context_route_weight <= 0:
            raise ValueError("Both semantic route weights must be positive.")
        if self.region_prior_strength < 0 or self.continuity_strength < 0:
            raise ValueError("Regional and continuity strengths must be non-negative.")
        if self.continuity_iterations < 0:
            raise ValueError("continuity_iterations must be non-negative.")
        if not 0.0 <= self.cross_support_weight <= 1.0:
            raise ValueError("cross_support_weight must be in [0, 1].")


@dataclass(frozen=True)
class RouteComplementarityDiagnostics:
    supports: int
    context_regions: int
    context_coverage: float
    branch_disagreement: float
    mean_js_divergence: float
    final_changed_from_local: float
    final_kept_local_against_context: float
    local_visual_coherence: float
    context_visual_coherence: float
    final_visual_coherence: float

    def summary(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SupervisedRouteComplementarity:
    valid_pixels: int
    context_coverage: float
    both_correct: float
    local_only_correct: float
    context_only_correct: float
    both_wrong: float
    local_accuracy: float
    context_filled_accuracy: float
    final_accuracy: float
    oracle_accuracy: float
    oracle_gain_over_best_branch: float
    local_miou: float
    context_filled_miou: float
    final_miou: float
    oracle_miou: float

    def summary(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ConnectedContextOVResult:
    logits: Tensor
    local_logits: Tensor
    context_logits: Tensor
    context_valid: Tensor
    support_ids: Tensor
    anchors: Tensor
    diagnostics: tuple[RouteComplementarityDiagnostics, ...]


class ConnectedContextOVSegmenter:
    """Frozen DINO.text with local and connected RGB-context observations."""

    method_name = "Connected-Context OV"
    implementation_note = (
        "Two frozen DINO.text observations: dense local pixels and re-encoded RGB "
        "context crops. Raw DINO continuity limits regional evidence propagation."
    )

    def __init__(self, checkpoints: CheckpointConfig | None = None, *,
                 backbone: DINOTextSegmenter | None = None,
                 config: ConnectedContextOVConfig = ConnectedContextOVConfig(),
                 device: str = "cuda", amp: bool = True) -> None:
        if backbone is None:
            if checkpoints is None:
                raise ValueError("Either checkpoints or an initialized DINOTextSegmenter is required.")
            backbone = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=False)
        elif checkpoints is not None:
            raise ValueError("Pass checkpoints or backbone, not both.")
        config.validate(backbone.patch_size)
        self.backbone = backbone
        self.config = config

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.backbone.encode_text(classes, batch_size=batch_size)

    def segment(self, rgb: Tensor, classes: Sequence[ClassSpec], *,
                valid_mask: Tensor | None = None,
                text_batch_size: int = 64) -> ConnectedContextOVResult:
        text = self.encode_text(classes, batch_size=text_batch_size)
        return self.segment_with_text_features(rgb, text, valid_mask=valid_mask)

    def segment_with_text_features(self, rgb: Tensor, text_features: Tensor, *,
                                   valid_mask: Tensor | None = None) -> ConnectedContextOVResult:
        local_features, raw_features, anchors = self.backbone.encode_image_with_structure(rgb)
        local_logits = self.backbone.similarity_logits(local_features, text_features).float()
        batch, _, height, width = local_logits.shape
        masks = _normalize_valid_mask(valid_mask, batch, height, width, local_logits.device)
        partitions = tuple(
            build_structure_interfaces(
                raw_features[index],
                StructureInterfaceConfig(self.config.atom_similarity_threshold),
                masks[index],
            )
            for index in range(batch)
        )
        context_logits, context_valid = self._context_observations(rgb, text_features, partitions)
        final_logits, diagnostics = support_conditioned_decision(
            local_logits, context_logits, context_valid, raw_features, partitions, self.config
        )
        return ConnectedContextOVResult(
            logits=final_logits,
            local_logits=local_logits,
            context_logits=context_logits,
            context_valid=context_valid,
            support_ids=torch.stack([item.atom_ids for item in partitions]),
            anchors=anchors,
            diagnostics=diagnostics,
        )

    def _context_observations(
        self,
        rgb: Tensor,
        text_features: Tensor,
        partitions: tuple[StructurePartition, ...],
    ) -> tuple[Tensor, Tensor]:
        batch = rgb.shape[0]
        classes = text_features.shape[0]
        height, width = partitions[0].height, partitions[0].width
        context_logits = torch.zeros((batch, classes, height, width), device=rgb.device)
        context_valid = torch.zeros((batch, height, width), device=rgb.device, dtype=torch.bool)
        observations: list[tuple[int, int, Tensor, Tensor]] = []

        for image_index, partition in enumerate(partitions):
            for atom_id in _rank_context_atoms(partition.atom_ids, self.config):
                crop, crop_mask = _extract_context_crop(
                    rgb[image_index],
                    partition.atom_ids == atom_id,
                    self.patch_size,
                    self.config.context_size,
                    self.config.context_margin,
                )
                observations.append((image_index, atom_id, crop, crop_mask))

        normalized_text = F.normalize(text_features.float(), dim=-1)
        for start in range(0, len(observations), self.config.context_batch_size):
            items = observations[start : start + self.config.context_batch_size]
            crops = torch.stack([item[2] for item in items])
            masks = torch.stack([item[3] for item in items])[:, None]
            patch_features, anchors = self.backbone.encode_image(crops)
            patch_masks = F.interpolate(masks.float(), size=patch_features.shape[-2:], mode="area")
            weights = patch_masks.clamp_min(0.0)
            pooled = (patch_features * weights).flatten(2).sum(dim=-1)
            pooled = F.normalize(pooled / weights.flatten(2).sum(dim=-1).clamp_min(1e-6), dim=-1)
            patch_scores = pooled @ normalized_text.T
            anchor_scores = F.normalize(anchors.float(), dim=-1) @ normalized_text.T
            scores = (
                (1.0 - self.config.context_anchor_weight) * patch_scores
                + self.config.context_anchor_weight * anchor_scores
            )
            for offset, (image_index, atom_id, _, _) in enumerate(items):
                support = partitions[image_index].atom_ids == atom_id
                context_logits[image_index, :, support] = scores[offset, :, None]
                context_valid[image_index, support] = True
        return context_logits, context_valid


def support_conditioned_decision(
    local_logits: Tensor,
    context_logits: Tensor,
    context_valid: Tensor,
    raw_features: Tensor,
    partitions: Sequence[StructurePartition],
    config: ConnectedContextOVConfig,
) -> tuple[Tensor, tuple[RouteComplementarityDiagnostics, ...]]:
    """Combine branch evidence per class, then enforce visual continuity."""
    _validate_decision_inputs(local_logits, context_logits, context_valid, raw_features, partitions)
    local_logp = F.log_softmax(local_logits.float() / config.branch_temperature, dim=1)
    context_logp = F.log_softmax(context_logits.float() / config.branch_temperature, dim=1)
    outputs: list[Tensor] = []
    diagnostics: list[RouteComplementarityDiagnostics] = []
    for index, partition in enumerate(partitions):
        local = local_logp[index]
        context = context_logp[index]
        valid = partition.valid_mask
        support = partition.atom_ids
        regional = torch.zeros_like(local)
        for atom_id in range(partition.atom_count):
            members = support == atom_id
            local_region = local[:, members].mean(dim=1)
            observed = context_valid[index] & members
            if bool(observed.any()):
                context_region = context[:, observed].mean(dim=1)
                regional_score = torch.logaddexp(
                    local_region + math.log(config.local_route_weight),
                    context_region + math.log(config.context_route_weight),
                ) - math.log(config.local_route_weight + config.context_route_weight)
            else:
                regional_score = local_region
            regional[:, members] = regional_score[:, None]

        regional = regional - regional.mean(dim=0, keepdim=True)
        unary = local + config.region_prior_strength * regional
        current = unary
        for _ in range(config.continuity_iterations):
            message = _visual_neighbor_message(
                torch.softmax(current, dim=0), raw_features[index], support, valid, config
            )
            current = unary + config.continuity_strength * message.clamp_min(1e-6).log()
        current[:, ~valid] = local[:, ~valid]
        outputs.append(current)
        diagnostics.append(
            _route_diagnostics(
                local,
                context,
                current,
                context_valid[index] & valid,
                raw_features[index],
                support,
                valid,
                partition.atom_count,
                config,
            )
        )
    return torch.stack(outputs), tuple(diagnostics)


def supervised_route_complementarity(
    local_logits: Tensor,
    context_logits: Tensor,
    final_logits: Tensor,
    context_valid: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
) -> SupervisedRouteComplementarity:
    """Measure unique branch corrections and the pixel-oracle upper bound."""
    if local_logits.shape != context_logits.shape or local_logits.shape != final_logits.shape:
        raise ValueError("All logits must share shape [B,C,H,W].")
    if target.shape != local_logits.shape[:1] + local_logits.shape[-2:]:
        raise ValueError("target must have shape [B,H,W].")
    if context_valid.shape != target.shape:
        raise ValueError("context_valid must have shape [B,H,W].")
    valid = target != ignore_index
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("No valid target pixels were provided.")
    local = local_logits.argmax(dim=1)
    context_raw = context_logits.argmax(dim=1)
    context = torch.where(context_valid, context_raw, local)
    final = final_logits.argmax(dim=1)
    local_ok = valid & (local == target)
    context_ok = valid & (context == target)
    oracle = torch.where(local_ok | context_ok, target, local)

    def fraction(mask: Tensor) -> float:
        return float(mask.sum().item() / count)

    local_accuracy = fraction(local_ok)
    context_accuracy = fraction(context_ok)
    final_accuracy = fraction(valid & (final == target))
    oracle_accuracy = fraction(valid & (oracle == target))
    classes = local_logits.shape[1]
    return SupervisedRouteComplementarity(
        valid_pixels=count,
        context_coverage=fraction(valid & context_valid),
        both_correct=fraction(local_ok & context_ok),
        local_only_correct=fraction(local_ok & ~context_ok),
        context_only_correct=fraction(~local_ok & context_ok),
        both_wrong=fraction(valid & ~local_ok & ~context_ok),
        local_accuracy=local_accuracy,
        context_filled_accuracy=context_accuracy,
        final_accuracy=final_accuracy,
        oracle_accuracy=oracle_accuracy,
        oracle_gain_over_best_branch=oracle_accuracy - max(local_accuracy, context_accuracy),
        local_miou=_mean_iou(local, target, valid, classes),
        context_filled_miou=_mean_iou(context, target, valid, classes),
        final_miou=_mean_iou(final, target, valid, classes),
        oracle_miou=_mean_iou(oracle, target, valid, classes),
    )


def _rank_context_atoms(atom_ids: Tensor, config: ConnectedContextOVConfig) -> list[int]:
    valid = atom_ids[atom_ids >= 0]
    if not valid.numel():
        return []
    counts = torch.bincount(valid, minlength=int(valid.max().item()) + 1)
    candidates = [
        (int(counts[atom].item()), atom)
        for atom in range(counts.numel())
        if int(counts[atom].item()) >= config.minimum_context_patches
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [atom for _, atom in candidates[: config.maximum_context_regions]]


def _extract_context_crop(
    rgb: Tensor,
    patch_mask: Tensor,
    patch_size: int,
    output_size: int,
    margin: float,
) -> tuple[Tensor, Tensor]:
    rows, columns = patch_mask.nonzero(as_tuple=True)
    if not rows.numel():
        raise ValueError("Cannot crop an empty support.")
    image_h, image_w = rgb.shape[-2:]
    y0 = int(rows.min().item()) * patch_size
    y1 = min((int(rows.max().item()) + 1) * patch_size, image_h)
    x0 = int(columns.min().item()) * patch_size
    x1 = min((int(columns.max().item()) + 1) * patch_size, image_w)
    expansion = int(round(max(y1 - y0, x1 - x0) * margin))
    y0, y1 = max(0, y0 - expansion), min(image_h, y1 + expansion)
    x0, x1 = max(0, x0 - expansion), min(image_w, x1 + expansion)
    pixel_mask = F.interpolate(
        patch_mask[None, None].float(), size=(image_h, image_w), mode="nearest"
    )[0, 0]
    crop = rgb[:, y0:y1, x0:x1]
    mask = pixel_mask[y0:y1, x0:x1]
    source_h, source_w = crop.shape[-2:]
    scale = output_size / max(source_h, source_w)
    target_h = max(1, min(output_size, int(round(source_h * scale))))
    target_w = max(1, min(output_size, int(round(source_w * scale))))
    crop = F.interpolate(crop[None], size=(target_h, target_w), mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(mask[None, None], size=(target_h, target_w), mode="nearest")[0, 0]
    pad_h, pad_w = output_size - target_h, output_size - target_w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(crop, padding, mode="replicate"), F.pad(mask, padding, value=0.0)


def _visual_neighbor_message(
    probabilities: Tensor,
    raw_features: Tensor,
    support_ids: Tensor,
    valid: Tensor,
    config: ConnectedContextOVConfig,
) -> Tensor:
    features = F.normalize(raw_features.float(), dim=0)
    weighted = probabilities.clone()
    normalizer = torch.ones_like(valid, dtype=torch.float32)

    if probabilities.shape[-1] > 1:
        similarity = (features[:, :, :-1] * features[:, :, 1:]).sum(dim=0)
        weights = torch.exp((similarity - 1.0) / config.structure_temperature)
        weights *= torch.where(
            support_ids[:, :-1] == support_ids[:, 1:], 1.0, config.cross_support_weight
        )
        weights *= valid[:, :-1] & valid[:, 1:]
        weighted[:, :, :-1] += probabilities[:, :, 1:] * weights
        weighted[:, :, 1:] += probabilities[:, :, :-1] * weights
        normalizer[:, :-1] += weights
        normalizer[:, 1:] += weights
    if probabilities.shape[-2] > 1:
        similarity = (features[:, :-1, :] * features[:, 1:, :]).sum(dim=0)
        weights = torch.exp((similarity - 1.0) / config.structure_temperature)
        weights *= torch.where(
            support_ids[:-1, :] == support_ids[1:, :], 1.0, config.cross_support_weight
        )
        weights *= valid[:-1, :] & valid[1:, :]
        weighted[:, :-1, :] += probabilities[:, 1:, :] * weights
        weighted[:, 1:, :] += probabilities[:, :-1, :] * weights
        normalizer[:-1, :] += weights
        normalizer[1:, :] += weights
    return weighted / normalizer.clamp_min(1e-6)


def _route_diagnostics(
    local_logp: Tensor,
    context_logp: Tensor,
    final_logits: Tensor,
    context_valid: Tensor,
    raw_features: Tensor,
    support_ids: Tensor,
    valid: Tensor,
    supports: int,
    config: ConnectedContextOVConfig,
) -> RouteComplementarityDiagnostics:
    valid_count = max(int(valid.sum().item()), 1)
    context_count = int(context_valid.sum().item())
    local_probability = local_logp.exp()
    context_probability = context_logp.exp()
    local_prediction = local_logp.argmax(dim=0)
    context_prediction = context_logp.argmax(dim=0)
    final_prediction = final_logits.argmax(dim=0)
    if context_count:
        mixture = 0.5 * (local_probability + context_probability)
        js = 0.5 * (
            (local_probability * (local_logp - mixture.clamp_min(1e-8).log())).sum(dim=0)
            + (context_probability * (context_logp - mixture.clamp_min(1e-8).log())).sum(dim=0)
        )
        disagreement = float(
            (local_prediction[context_valid] != context_prediction[context_valid]).float().mean().item()
        )
        mean_js = float(js[context_valid].mean().item())
        kept = context_valid & (local_prediction != context_prediction) & (final_prediction == local_prediction)
        kept_fraction = float(kept.sum().item() / context_count)
    else:
        disagreement = mean_js = kept_fraction = 0.0
    changed = valid & (final_prediction != local_prediction)
    return RouteComplementarityDiagnostics(
        supports=supports,
        context_regions=int(torch.unique(support_ids[context_valid]).numel()) if context_count else 0,
        context_coverage=context_count / valid_count,
        branch_disagreement=disagreement,
        mean_js_divergence=mean_js,
        final_changed_from_local=float(changed.sum().item() / valid_count),
        final_kept_local_against_context=kept_fraction,
        local_visual_coherence=_visual_coherence(local_prediction, raw_features, valid, config),
        context_visual_coherence=_visual_coherence(context_prediction, raw_features, context_valid, config),
        final_visual_coherence=_visual_coherence(final_prediction, raw_features, valid, config),
    )


def _visual_coherence(
    labels: Tensor,
    raw_features: Tensor,
    valid: Tensor,
    config: ConnectedContextOVConfig,
) -> float:
    features = F.normalize(raw_features.float(), dim=0)
    agreements: list[Tensor] = []
    weights_all: list[Tensor] = []
    if labels.shape[1] > 1:
        weights = torch.exp(
            ((features[:, :, :-1] * features[:, :, 1:]).sum(0) - 1.0)
            / config.structure_temperature
        )
        weights *= valid[:, :-1] & valid[:, 1:]
        agreements.append((labels[:, :-1] == labels[:, 1:]).float() * weights)
        weights_all.append(weights)
    if labels.shape[0] > 1:
        weights = torch.exp(
            ((features[:, :-1, :] * features[:, 1:, :]).sum(0) - 1.0)
            / config.structure_temperature
        )
        weights *= valid[:-1, :] & valid[1:, :]
        agreements.append((labels[:-1, :] == labels[1:, :]).float() * weights)
        weights_all.append(weights)
    denominator = sum(
        (item.sum() for item in weights_all),
        start=labels.new_tensor(0.0, dtype=torch.float32),
    )
    if float(denominator.item()) <= 0:
        return 0.0
    numerator = sum(
        (item.sum() for item in agreements),
        start=labels.new_tensor(0.0, dtype=torch.float32),
    )
    return float((numerator / denominator).item())


def _mean_iou(prediction: Tensor, target: Tensor, valid: Tensor, classes: int) -> float:
    scores = []
    for class_index in range(classes):
        intersection = (valid & (prediction == class_index) & (target == class_index)).sum()
        union = (valid & ((prediction == class_index) | (target == class_index))).sum()
        if int(union.item()):
            scores.append(intersection.float() / union.float())
    return float(torch.stack(scores).mean().item()) if scores else 0.0


def _normalize_valid_mask(
    valid_mask: Tensor | None,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones((batch, height, width), device=device, dtype=torch.bool)
    if valid_mask.ndim == 2 and batch == 1:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape != (batch, height, width):
        raise ValueError("valid_mask must have shape [B,H,W] on the patch grid.")
    return valid_mask.to(device=device, dtype=torch.bool)


def _validate_decision_inputs(
    local_logits: Tensor,
    context_logits: Tensor,
    context_valid: Tensor,
    raw_features: Tensor,
    partitions: Sequence[StructurePartition],
) -> None:
    if local_logits.ndim != 4 or context_logits.shape != local_logits.shape:
        raise ValueError("local_logits and context_logits must share shape [B,C,H,W].")
    if raw_features.ndim != 4 or raw_features.shape[0] != local_logits.shape[0]:
        raise ValueError("raw_features must have shape [B,D,H,W].")
    if raw_features.shape[-2:] != local_logits.shape[-2:]:
        raise ValueError("Raw features and logits must share grid dimensions.")
    if context_valid.shape != local_logits.shape[:1] + local_logits.shape[-2:]:
        raise ValueError("context_valid must have shape [B,H,W].")
    if len(partitions) != local_logits.shape[0]:
        raise ValueError("One structure partition is required per batch image.")
