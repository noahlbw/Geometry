from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import find_objects as _scipy_find_objects
    from scipy.ndimage import label as _scipy_label
except ImportError:  # Keep the package usable in minimal inference environments.
    _scipy_find_objects = None
    _scipy_label = None


@dataclass(frozen=True)
class RegionVerifierConfig:
    """Training-free, conservative controls for DINO region verification."""

    enabled: bool = True
    minimum_region_area: int = 64
    maximum_regions_per_class: int = 128
    sam_confident_score: float = 0.72
    sam_ambiguous_margin: float = 0.12
    sam_candidate_count: int = 3
    minimum_dino_margin: float = 0.015
    minimum_boundary_contrast: float = -0.01
    global_top_k: int = 4
    boundary_width_patches: int = 1
    minimum_patch_coverage: float = 0.10
    use_per_image_prototypes: bool = True
    prototype_minimum_regions: int = 2
    prototype_score_bias: float = 0.01

    def validate(self) -> None:
        if self.minimum_region_area < 1 or self.maximum_regions_per_class < 1:
            raise ValueError("Region component limits must be positive.")
        if not 0.0 <= self.sam_confident_score <= 1.0:
            raise ValueError("sam_confident_score must be in [0, 1].")
        if self.sam_ambiguous_margin < 0.0 or self.sam_candidate_count < 2:
            raise ValueError("SAM region uncertainty settings are invalid.")
        if self.minimum_dino_margin < 0.0:
            raise ValueError("minimum_dino_margin must be non-negative.")
        if self.global_top_k < 1 or self.boundary_width_patches < 0:
            raise ValueError("Global top-k and boundary width must be non-negative/positive.")
        if not 0.0 < self.minimum_patch_coverage <= 1.0:
            raise ValueError("minimum_patch_coverage must be in (0, 1].")
        if self.prototype_minimum_regions < 1:
            raise ValueError("prototype_minimum_regions must be positive.")

    def signature(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegionVerificationDiagnostics:
    regions_total: int
    regions_ambiguous: int
    regions_relabelled: int
    prototype_classes: int
    skipped_no_dino_candidate: int
    skipped_low_dino_margin: int
    skipped_low_boundary_contrast: int
    skipped_global_presence: int

    def summary(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _Region:
    class_index: int
    mask: np.ndarray
    area: int


@dataclass(frozen=True)
class _Evidence:
    region: _Region
    sam_scores: torch.Tensor
    sam_candidates: torch.Tensor
    sam_score: float
    sam_margin: float
    visual_feature: torch.Tensor
    dino_scores: torch.Tensor
    dino_candidate: int
    dino_margin: float
    boundary_contrast: float
    ring_scores: torch.Tensor | None
    ambiguous: bool


class DINORegionVerifier:
    """Use DINOv3 semantics to verify SAM3 regions without pixel-logit blending."""

    def __init__(self, config: RegionVerifierConfig) -> None:
        config.validate()
        self.config = config

    def verify(
        self,
        sam_scores: torch.Tensor,
        patch_features: torch.Tensor,
        text_features: torch.Tensor,
        global_anchor: torch.Tensor,
        *,
        dino_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RegionVerificationDiagnostics]:
        """Return a conservative region-level relabeling of SAM3 scores.

        DINO can only select a class already present in the local SAM candidate
        set.  It never paints a new dense mask and it cannot replace a region
        whose SAM score and margin are both confident.
        """

        self._validate_inputs(sam_scores, patch_features, text_features, global_anchor, dino_logits)
        classes, height, width = sam_scores.shape
        device = sam_scores.device
        labels = sam_scores.argmax(dim=0)
        regions = _extract_regions(
            labels.detach().cpu().numpy(),
            classes,
            minimum_area=self.config.minimum_region_area,
            maximum_regions_per_class=self.config.maximum_regions_per_class,
        )
        normalized_text = F.normalize(text_features.float(), dim=-1)
        normalized_patches = F.normalize(patch_features[0].float(), dim=0)
        normalized_anchor = F.normalize(global_anchor.float().reshape(-1), dim=0)
        global_scores = normalized_anchor @ normalized_text.T
        global_k = min(classes, max(self.config.global_top_k, int(math.ceil(classes * 0.50))))
        globally_present = set(global_scores.topk(global_k).indices.tolist())

        evidence = [
            self._region_evidence(
                region,
                sam_scores,
                normalized_patches,
                normalized_text,
                dino_logits,
            )
            for region in regions
        ]
        prototypes = self._build_prototypes(evidence, classes)

        relabelled = 0
        ambiguous = 0
        skipped_no_candidate = 0
        skipped_margin = 0
        skipped_contrast = 0
        skipped_presence = 0
        for item in evidence:
            if not item.ambiguous:
                continue
            ambiguous += 1
            candidate = item.dino_candidate
            dino_scores = item.dino_scores
            if prototypes is not None:
                prototype_scores = self._prototype_scores(item.visual_feature, prototypes, classes)
                dino_scores = torch.maximum(dino_scores, prototype_scores - self.config.prototype_score_bias)
                candidate = int(dino_scores.argmax().item())
            if candidate not in item.sam_candidates.tolist():
                skipped_no_candidate += 1
                continue
            top_values = dino_scores.topk(min(2, classes)).values
            dino_margin = float((top_values[0] - top_values[-1]).item()) if classes > 1 else float("inf")
            if dino_margin < self.config.minimum_dino_margin:
                skipped_margin += 1
                continue
            boundary_contrast = item.boundary_contrast
            if item.ring_scores is not None:
                boundary_contrast = float((item.dino_scores[candidate] - item.ring_scores[candidate]).item())
            if boundary_contrast < self.config.minimum_boundary_contrast:
                skipped_contrast += 1
                continue
            if candidate not in globally_present:
                skipped_presence += 1
                continue
            if candidate == item.region.class_index:
                continue
            mask = torch.from_numpy(item.region.mask).to(device=device, dtype=torch.bool)
            labels[mask] = candidate
            relabelled += 1

        diagnostics = RegionVerificationDiagnostics(
            regions_total=len(evidence),
            regions_ambiguous=ambiguous,
            regions_relabelled=relabelled,
            prototype_classes=0 if prototypes is None else int((prototypes.norm(dim=1) > 0).sum().item()),
            skipped_no_dino_candidate=skipped_no_candidate,
            skipped_low_dino_margin=skipped_margin,
            skipped_low_boundary_contrast=skipped_contrast,
            skipped_global_presence=skipped_presence,
        )
        return labels, diagnostics

    def _region_evidence(
        self,
        region: _Region,
        sam_scores: torch.Tensor,
        patch_features: torch.Tensor,
        text_features: torch.Tensor,
        dino_logits: torch.Tensor | None,
    ) -> _Evidence:
        device = sam_scores.device
        mask = torch.from_numpy(region.mask).to(device=device, dtype=torch.bool)
        region_sam_scores = sam_scores[:, mask].mean(dim=1)
        candidate_count = min(self.config.sam_candidate_count, region_sam_scores.numel())
        sam_values, sam_candidates = region_sam_scores.topk(candidate_count)
        sam_score = float(sam_values[0].item())
        sam_margin = float((sam_values[0] - sam_values[1]).item()) if candidate_count > 1 else float("inf")

        low_mask = F.interpolate(
            mask.float().view(1, 1, *mask.shape),
            size=patch_features.shape[-2:],
            mode="area",
        )[0, 0]
        weights = low_mask.clamp_min(0.0)
        visual_feature = F.normalize(
            (patch_features * weights.unsqueeze(0)).flatten(1).sum(dim=1) / weights.sum().clamp_min(1e-6),
            dim=0,
        )
        feature_scores = visual_feature @ text_features.T
        if dino_logits is None:
            dino_scores = feature_scores
        else:
            dino_scores = 0.5 * (feature_scores + _weighted_mean(dino_logits, weights))

        patch_mask = low_mask >= self.config.minimum_patch_coverage
        ring = _boundary_ring(patch_mask, self.config.boundary_width_patches)
        ring_scores: torch.Tensor | None = None
        if ring.any():
            ring_feature = F.normalize(
                (patch_features * ring.float().unsqueeze(0)).flatten(1).sum(dim=1)
                / ring.sum().clamp_min(1),
                dim=0,
            )
            ring_scores = ring_feature @ text_features.T
            candidate = int(dino_scores.argmax().item())
            boundary_contrast = float((dino_scores[candidate] - ring_scores[candidate]).item())
        else:
            boundary_contrast = 0.0

        top_values, top_indices = dino_scores.topk(min(2, dino_scores.numel()))
        dino_margin = float((top_values[0] - top_values[-1]).item()) if dino_scores.numel() > 1 else float("inf")
        return _Evidence(
            region=region,
            sam_scores=region_sam_scores,
            sam_candidates=sam_candidates,
            sam_score=sam_score,
            sam_margin=sam_margin,
            visual_feature=visual_feature,
            dino_scores=dino_scores,
            dino_candidate=int(top_indices[0].item()),
            dino_margin=dino_margin,
            boundary_contrast=boundary_contrast,
            ring_scores=ring_scores,
            ambiguous=(
                sam_score < self.config.sam_confident_score
                or sam_margin < self.config.sam_ambiguous_margin
            ),
        )

    def _build_prototypes(self, evidence: list[_Evidence], classes: int) -> torch.Tensor | None:
        if not self.config.use_per_image_prototypes or not evidence:
            return None
        grouped: list[list[torch.Tensor]] = [[] for _ in range(classes)]
        for item in evidence:
            if item.ambiguous or item.dino_candidate != item.region.class_index:
                continue
            if item.dino_margin < self.config.minimum_dino_margin:
                continue
            grouped[item.region.class_index].append(item.visual_feature)
        prototypes = torch.zeros((classes, evidence[0].visual_feature.numel()), device=evidence[0].visual_feature.device)
        valid = False
        for class_index, features in enumerate(grouped):
            if len(features) < self.config.prototype_minimum_regions:
                continue
            prototypes[class_index] = F.normalize(torch.stack(features).mean(dim=0), dim=0)
            valid = True
        return prototypes if valid else None

    @staticmethod
    def _prototype_scores(feature: torch.Tensor, prototypes: torch.Tensor, classes: int) -> torch.Tensor:
        scores = feature @ prototypes.T
        missing = prototypes.norm(dim=1) == 0
        scores[missing] = torch.finfo(scores.dtype).min
        return scores

    @staticmethod
    def _validate_inputs(
        sam_scores: torch.Tensor,
        patch_features: torch.Tensor,
        text_features: torch.Tensor,
        global_anchor: torch.Tensor,
        dino_logits: torch.Tensor | None,
    ) -> None:
        if sam_scores.ndim != 3:
            raise ValueError("SAM scores must have shape [classes, height, width].")
        if patch_features.ndim != 4 or patch_features.shape[0] != 1:
            raise ValueError("DINO patch features must have shape [1, features, height, width].")
        if text_features.ndim != 2 or text_features.shape[0] != sam_scores.shape[0]:
            raise ValueError("DINO text features must have one row per SAM class.")
        if patch_features.shape[1] != text_features.shape[1] or global_anchor.numel() != text_features.shape[1]:
            raise ValueError("DINO patch, global, and text feature dimensions must match.")
        if dino_logits is not None and dino_logits.shape != (sam_scores.shape[0], *patch_features.shape[-2:]):
            raise ValueError("DINO logits must match the class and patch-grid dimensions.")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights.unsqueeze(0)).flatten(1).sum(dim=1) / weights.sum().clamp_min(1e-6)


def _boundary_ring(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    values = mask.float().view(1, 1, *mask.shape)
    kernel = 2 * width + 1
    dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=width).squeeze() > 0
    return dilated & ~mask


def _extract_regions(
    labels: np.ndarray,
    classes: int,
    *,
    minimum_area: int,
    maximum_regions_per_class: int,
) -> list[_Region]:
    if labels.ndim != 2:
        raise ValueError("Region labels must be two-dimensional.")
    regions: list[_Region] = []
    for class_index in range(classes):
        mask = labels == class_index
        class_regions = _connected_components(mask, class_index, minimum_area)
        class_regions.sort(key=lambda region: region.area, reverse=True)
        regions.extend(class_regions[:maximum_regions_per_class])
    return regions


def _connected_components(mask: np.ndarray, class_index: int, minimum_area: int) -> list[_Region]:
    """Extract 8-connected regions with a compiled fast path when available."""

    if _scipy_label is not None and _scipy_find_objects is not None:
        return _connected_components_scipy(mask, class_index, minimum_area)
    return _connected_components_python(mask, class_index, minimum_area)


def _connected_components_scipy(mask: np.ndarray, class_index: int, minimum_area: int) -> list[_Region]:
    component_labels, component_count = _scipy_label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return []

    regions: list[_Region] = []
    # ``find_objects`` bounds each dense comparison so we do not repeatedly
    # scan an entire image while rebuilding retained component masks.
    for component_index, bounds in enumerate(_scipy_find_objects(component_labels), start=1):
        if bounds is None:
            continue
        component = component_labels[bounds] == component_index
        area = int(component.sum())
        if area < minimum_area:
            continue
        region_mask = np.zeros_like(mask, dtype=bool)
        region_mask[bounds] = component
        regions.append(_Region(class_index=class_index, mask=region_mask, area=area))
    return regions


def _connected_components_python(mask: np.ndarray, class_index: int, minimum_area: int) -> list[_Region]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[_Region] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y, next_x = y + dy, x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
        if len(pixels) < minimum_area:
            continue
        region_mask = np.zeros_like(mask, dtype=bool)
        region_y, region_x = zip(*pixels)
        region_mask[np.asarray(region_y), np.asarray(region_x)] = True
        regions.append(_Region(class_index=class_index, mask=region_mask, area=len(pixels)))
    return regions
