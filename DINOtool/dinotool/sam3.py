from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .loveda import LOVEDA_CLASS_NAMES, LoveDAConfusionMatrix, discover_loveda_samples


OFFICIAL_LOVEDA_PROMPTS: tuple[str, ...] = (
    "background",
    "building,house",
    "road",
    "water",
    "barren,bareland,soil",
    "forest,tree",
    "agricultural",
)


@dataclass(frozen=True)
class Sam3GeometryConfig:
    """Label-free geometric re-prompting controls for SAM 3 query masks.

    The defaults are deliberately generic: no class IDs or dataset-specific
    routing rules are encoded here.  A candidate must be compact enough to be
    useful as a box prompt and the refined response must remain consistent
    with the text-only response before it is accepted.
    """

    enabled: bool = False
    probability_threshold: float = 0.60
    minimum_component_area: int = 80
    maximum_mask_area_ratio: float = 0.25
    maximum_components: int = 2
    box_expand_ratio: float = 0.04
    boundary_width: int = 3
    blend: float = 0.25
    minimum_consistency_iou: float = 0.50
    refined_area_ratio_low: float = 0.50
    refined_area_ratio_high: float = 1.80
    reprompt_background: bool = False
    maximum_prompt_calls: int | None = None

    def validate(self) -> None:
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("Geometry probability_threshold must be in (0, 1).")
        if self.minimum_component_area < 1 or self.maximum_components < 1:
            raise ValueError("Geometry component limits must be positive.")
        if not 0.0 < self.maximum_mask_area_ratio <= 1.0:
            raise ValueError("Geometry maximum_mask_area_ratio must be in (0, 1].")
        if not 0.0 <= self.box_expand_ratio <= 1.0:
            raise ValueError("Geometry box_expand_ratio must be in [0, 1].")
        if self.boundary_width < 0 or not 0.0 <= self.blend <= 1.0:
            raise ValueError("Geometry boundary_width or blend is invalid.")
        if not 0.0 <= self.minimum_consistency_iou <= 1.0:
            raise ValueError("Geometry minimum_consistency_iou must be in [0, 1].")
        if self.refined_area_ratio_low <= 0 or self.refined_area_ratio_high < self.refined_area_ratio_low:
            raise ValueError("Geometry refined area ratio limits are invalid.")
        if self.maximum_prompt_calls is not None and self.maximum_prompt_calls < 1:
            raise ValueError("Geometry maximum_prompt_calls must be positive when provided.")


@dataclass(frozen=True)
class Sam3PromptComponents:
    """Prompt-conditioned SAM 3 score maps at the original image size.

    The official processor calls the source fields ``*_logits``, but applies
    sigmoid before placing them in its state dictionary.  The field names are
    retained for compatibility with the official implementation; values here
    are probabilities in ``[0, 1]``.
    """

    query_index: int
    class_index: int
    prompt: str
    semantic_logits: torch.Tensor
    instance_mask_logits: torch.Tensor
    instance_scores: torch.Tensor
    instance_fused_logits: torch.Tensor
    base_logits: torch.Tensor
    presence_score: torch.Tensor
    fused_logits: torch.Tensor
    geometry_attempted: bool = False
    geometry_accepted: bool = False


@dataclass(frozen=True)
class Sam3ImageComponents:
    """All prompt outputs and their class-level baseline aggregation."""

    prompt_outputs: tuple[Sam3PromptComponents, ...]
    class_logits: torch.Tensor
    geometry_attempted: int = 0
    geometry_accepted: int = 0


@dataclass
class Sam3ImageContext:
    """Reusable SAM 3 image state for sequential text-grounding requests.

    The processor mutates this state while setting and clearing prompts.  It is
    intentionally session-local and must not be used concurrently.
    """

    state: dict[str, Any]
    width: int
    height: int


@dataclass(frozen=True)
class Sam3Config:
    """Paths and inference settings for the official SegEarth-OV3 SAM 3 model."""

    root: Path
    checkpoint_path: Path
    bpe_path: Path
    class_file: Path
    device: str = "cuda"
    resolution: int = 1008
    confidence_threshold: float = 0.5
    probability_threshold: float = 0.5
    amp: bool = True

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        bpe_path: str | Path | None = None,
        class_file: str | Path | None = None,
        device: str = "cuda",
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        probability_threshold: float = 0.5,
        amp: bool = True,
    ) -> "Sam3Config":
        resolved_root = Path(root).expanduser().resolve()
        return cls(
            root=resolved_root,
            checkpoint_path=(
                Path(checkpoint_path).expanduser().resolve()
                if checkpoint_path
                else resolved_root / "weights" / "sam3" / "sam3.pt"
            ),
            bpe_path=(
                Path(bpe_path).expanduser().resolve()
                if bpe_path
                else resolved_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            ),
            class_file=(
                Path(class_file).expanduser().resolve()
                if class_file
                else resolved_root / "configs" / "cls_loveda.txt"
            ),
            device=device,
            resolution=resolution,
            confidence_threshold=confidence_threshold,
            probability_threshold=probability_threshold,
            amp=amp,
        )

    def validate(self) -> None:
        missing = [path for path in (self.root, self.checkpoint_path, self.bpe_path, self.class_file) if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing SegEarth-OV3 asset(s):\n{formatted}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("SAM 3 confidence_threshold must be in [0, 1].")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("SAM 3 probability_threshold must be in [0, 1].")
        if self.resolution < 32:
            raise ValueError("SAM 3 resolution must be at least 32 pixels.")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    def signature(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "checkpoint": _file_manifest(self.checkpoint_path),
            "bpe": _file_manifest(self.bpe_path),
            "class_file": _file_manifest(self.class_file),
            "device": self.device,
            "resolution": self.resolution,
            "confidence_threshold": self.confidence_threshold,
            "probability_threshold": self.probability_threshold,
            "amp": self.amp,
        }


class SegEarthOV3Predictor:
    """Standalone execution of the official SegEarth-OV3 image inference path.

    MMSegmentation is intentionally not imported here. Its runner only provides
    dataset plumbing; this class preserves the repository's SAM 3 image setup,
    query loop, dual-head max fusion, presence filtering, synonym aggregation,
    and background thresholding.
    """

    def __init__(self, config: Sam3Config) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        self.query_words, query_indices = _read_query_file(config.class_file)
        self.query_indices = torch.tensor(query_indices, dtype=torch.int64, device=self.device)
        self.num_classes = max(query_indices) + 1
        if self.num_classes != len(LOVEDA_CLASS_NAMES):
            raise ValueError(
                f"LoveDA benchmark requires {len(LOVEDA_CLASS_NAMES)} class lines, "
                f"but {config.class_file} defines {self.num_classes}."
            )
        self.use_amp = config.amp and self.device.type == "cuda"

        root_string = str(config.root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as error:
            raise ImportError(
                f"Could not import SAM 3 from {config.root}. Supply the official SegEarth-OV3 repository via --sam3-root."
            ) from error

        model = build_sam3_image_model(
            bpe_path=str(config.bpe_path),
            checkpoint_path=str(config.checkpoint_path),
            device=str(self.device),
        )
        self.processor = Sam3Processor(
            model,
            resolution=config.resolution,
            confidence_threshold=config.confidence_threshold,
            device=self.device,
        )

    def predict_logits(self, image: Image.Image) -> torch.Tensor:
        """Return official dual-head class confidence maps at the original image resolution.

        This remains the reference SegEarth-OV3 path.  It is implemented using
        ``predict_components`` so the raw head outputs consumed by the new
        SAM3+DINO predictor cannot silently diverge from the benchmark baseline.
        """

        return self.predict_components(image).class_logits

    def predict_components(
        self,
        image: Image.Image,
        *,
        query_words: Sequence[str] | None = None,
        query_class_indices: Sequence[int] | None = None,
        geometry: Sam3GeometryConfig | None = None,
        class_count: int | None = None,
    ) -> Sam3ImageComponents:
        """Return raw SAM 3 prompt outputs plus official class aggregation.

        ``query_words`` permits a cached, constrained prompt bank to add
        variants without mutating the predictor's baseline prompt contract.
        Geometry is optional and always operates after text grounding within
        the same encoded image state, so no extra image encoding is required.
        """

        return self.predict_components_from_context(
            self.prepare_image(image),
            query_words=query_words,
            query_class_indices=query_class_indices,
            geometry=geometry,
            class_count=class_count,
        )

    def prepare_image(self, image: Image.Image) -> Sam3ImageContext:
        """Encode an image once for a session that may issue multiple queries."""

        image = image.convert("RGB")
        width, height = image.size
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            state = self.processor.set_image(image)
        return Sam3ImageContext(state=state, width=width, height=height)

    def predict_components_from_context(
        self,
        context: Sam3ImageContext,
        *,
        query_words: Sequence[str] | None = None,
        query_class_indices: Sequence[int] | None = None,
        geometry: Sam3GeometryConfig | None = None,
        class_count: int | None = None,
    ) -> Sam3ImageComponents:
        """Ground prompts against a context returned by :meth:`prepare_image`.

        ``class_count`` decouples agent vocabularies from the LoveDA prompt
        file used to bootstrap the official SegEarth model.  It only changes
        aggregation shape; model weights and image encoding remain unchanged.
        """

        height = context.height
        width = context.width
        active_class_count = self.num_classes if class_count is None else class_count
        if active_class_count < 1:
            raise ValueError("SAM 3 class_count must be positive.")
        words = list(self.query_words if query_words is None else query_words)
        indices = list(
            self.query_indices.detach().cpu().tolist()
            if query_class_indices is None
            else query_class_indices
        )
        if not words or len(words) != len(indices):
            raise ValueError("SAM 3 query words and class indices must be non-empty and have the same length.")
        if any(index < 0 or index >= active_class_count for index in indices):
            raise ValueError("SAM 3 query class indices are outside the configured class range.")
        if geometry is not None:
            geometry.validate()

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )
        outputs: list[Sam3PromptComponents] = []
        geometry_attempted = 0
        geometry_accepted = 0
        geometry_prompt_calls = 0
        with torch.no_grad(), autocast:
            state = context.state
            for query_position, (query_word, class_index) in enumerate(zip(words, indices)):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(state=state, prompt=query_word)
                components = self._components_from_state(
                    state,
                    height=height,
                    width=width,
                    query_index=query_position,
                    class_index=class_index,
                    prompt=query_word,
                )
                use_geometry = geometry is not None and geometry.enabled and (
                    geometry.reprompt_background or class_index != 0
                )
                within_geometry_budget = (
                    geometry is None
                    or geometry.maximum_prompt_calls is None
                    or geometry_prompt_calls < geometry.maximum_prompt_calls
                )
                if use_geometry and within_geometry_budget:
                    geometry_prompt_calls += 1
                    components = self._geometric_reprompt(
                        state,
                        components,
                        height=height,
                        width=width,
                        config=geometry,
                    )
                    geometry_attempted += int(components.geometry_attempted)
                    geometry_accepted += int(components.geometry_accepted)
                outputs.append(components)
            self.processor.reset_all_prompts(state)
            context.state = state

        class_logits = self._aggregate_class_logits(outputs, height, width, active_class_count)
        return Sam3ImageComponents(
            prompt_outputs=tuple(outputs),
            class_logits=class_logits,
            geometry_attempted=geometry_attempted,
            geometry_accepted=geometry_accepted,
        )

    def _components_from_state(
        self,
        state: dict[str, Any],
        *,
        height: int,
        width: int,
        query_index: int,
        class_index: int,
        prompt: str,
    ) -> Sam3PromptComponents:
        """Extract one prompt's heads while preserving the official max path."""

        # Sam3Processor has already applied sigmoid to this field.  Preserve
        # those probabilities exactly so this standalone path matches its
        # official dual-head aggregation.
        semantic_logits = self._resize_semantic_logits(state["semantic_mask_logits"], height, width).float()
        base_logits = torch.zeros((height, width), device=self.device, dtype=torch.float32)

        raw_masks = state.get("masks_logits")
        raw_scores = state.get("object_score")
        if isinstance(raw_masks, torch.Tensor) and raw_masks.shape[0] > 0 and isinstance(raw_scores, torch.Tensor):
            raw_masks = raw_masks.squeeze(1)
            raw_scores = raw_scores.reshape(-1)
            if raw_masks.shape[0] != raw_scores.shape[0]:
                raise RuntimeError("SAM 3 instance mask and object-score counts do not match.")
            weighted_masks = raw_masks * raw_scores.view(-1, 1, 1)
            if weighted_masks.shape[-2:] != (height, width):
                weighted_masks = F.interpolate(
                    weighted_masks.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            instance_fused_logits = weighted_masks.max(dim=0).values.float()
            if raw_masks.shape[-2:] != (height, width):
                raw_masks = F.interpolate(
                    raw_masks.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            instance_mask_logits = raw_masks.float()
            instance_scores = raw_scores.float()
            base_logits = torch.maximum(base_logits, instance_fused_logits)
        else:
            instance_mask_logits = torch.empty((0, height, width), device=self.device, dtype=torch.float32)
            instance_scores = torch.empty((0,), device=self.device, dtype=torch.float32)
            instance_fused_logits = torch.zeros((height, width), device=self.device, dtype=torch.float32)

        # This is the exact SegEarth-OV3 dual-head aggregation order.
        base_logits = torch.maximum(base_logits, semantic_logits)
        presence = torch.as_tensor(state.get("presence_score", 1.0), device=self.device).reshape(-1)[0].float()
        fused_logits = base_logits * presence
        return Sam3PromptComponents(
            query_index=query_index,
            class_index=class_index,
            prompt=prompt,
            semantic_logits=semantic_logits,
            instance_mask_logits=instance_mask_logits,
            instance_scores=instance_scores,
            instance_fused_logits=instance_fused_logits,
            base_logits=base_logits,
            presence_score=presence,
            fused_logits=fused_logits,
        )

    @staticmethod
    def _resize_semantic_logits(logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # The official implementation compares the complete tensor shape to
        # (H, W), causing its [N, C, H, W] tensor to be interpolated.
        if logits.shape != (height, width):
            if logits.ndim == 2:
                logits = logits.view(1, 1, *logits.shape)
            elif logits.ndim == 3:
                logits = logits.unsqueeze(1)
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return logits.squeeze()

    def _aggregate_class_logits(
        self,
        outputs: Sequence[Sam3PromptComponents],
        height: int,
        width: int,
        class_count: int | None = None,
    ) -> torch.Tensor:
        active_class_count = self.num_classes if class_count is None else class_count
        class_logits = torch.zeros((active_class_count, height, width), device=self.device)
        for class_index in range(active_class_count):
            matching = [output.fused_logits for output in outputs if output.class_index == class_index]
            if matching:
                class_logits[class_index] = torch.stack(matching, dim=0).max(dim=0).values
        return class_logits

    def _geometric_reprompt(
        self,
        state: dict[str, Any],
        components: Sam3PromptComponents,
        *,
        height: int,
        width: int,
        config: Sam3GeometryConfig,
    ) -> Sam3PromptComponents:
        """Refine compact text-grounded components using SAM 3 box prompts."""

        coarse_probabilities = components.base_logits.clamp(0.0, 1.0)
        coarse_mask = coarse_probabilities > config.probability_threshold
        coarse_area = int(coarse_mask.sum().item())
        if (
            coarse_area < config.minimum_component_area
            or coarse_area / max(coarse_mask.numel(), 1) > config.maximum_mask_area_ratio
            or not hasattr(self.processor, "add_geometric_prompt")
        ):
            return components

        boxes = _component_boxes(
            coarse_mask.detach().cpu().numpy(),
            minimum_area=config.minimum_component_area,
            maximum_components=config.maximum_components,
            score_map=coarse_probabilities.detach().cpu().numpy(),
            expand_ratio=config.box_expand_ratio,
        )
        if not boxes:
            return components

        refined_state = state
        for box in boxes:
            normalized = _xyxy_to_normalized_cxcywh(box, width, height)
            result = self.processor.add_geometric_prompt(state=refined_state, box=normalized, label=True)
            if result is not None:
                refined_state = result
        refined = self._components_from_state(
            refined_state,
            height=height,
            width=width,
            query_index=components.query_index,
            class_index=components.class_index,
            prompt=components.prompt,
        )
        refined_probabilities = refined.base_logits.clamp(0.0, 1.0)
        refined_mask = refined_probabilities > config.probability_threshold
        refined_area = int(refined_mask.sum().item())
        if refined_area < config.minimum_component_area:
            return replace(components, geometry_attempted=True)

        intersection = torch.logical_and(coarse_mask, refined_mask).sum().float()
        union = torch.logical_or(coarse_mask, refined_mask).sum().float()
        consistency_iou = float((intersection / union.clamp_min(1.0)).item())
        area_ratio = refined_area / max(coarse_area, 1)
        if (
            consistency_iou < config.minimum_consistency_iou
            or area_ratio < config.refined_area_ratio_low
            or area_ratio > config.refined_area_ratio_high
        ):
            return replace(components, geometry_attempted=True)

        boundary = _boundary_band(coarse_mask, config.boundary_width)
        useful = boundary & (refined_probabilities > coarse_probabilities)
        blended = components.base_logits.clone()
        blended[useful] = (
            (1.0 - config.blend) * components.base_logits[useful]
            + config.blend * refined.base_logits[useful]
        )
        return replace(
            refined,
            base_logits=blended,
            # Keep text-only presence: re-prompting must not manufacture a
            # concept absent from the original prompt-conditioned image state.
            presence_score=components.presence_score,
            fused_logits=blended * components.presence_score,
            geometry_attempted=True,
            geometry_accepted=True,
        )

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return label IDs and winning SAM 3 logits for one image."""

        class_logits = self.predict_logits(image)
        maximum, prediction = class_logits.max(dim=0)
        prediction[maximum < self.config.probability_threshold] = 0
        labels = prediction.to(torch.uint8).cpu().numpy()
        confidence = maximum.float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        return labels, confidence


def _component_boxes(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_components: int,
    score_map: np.ndarray,
    expand_ratio: float,
) -> list[tuple[float, float, float, float]]:
    """Return top connected-component boxes without adding an OpenCV dependency."""

    if mask.ndim != 2 or score_map.shape != mask.shape:
        raise ValueError("Component mask and score map must be matching two-dimensional arrays.")
    height, width = mask.shape
    mask = mask.astype(bool, copy=False)
    visited = np.zeros_like(mask, dtype=bool)
    candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        pending: deque[tuple[int, int]] = deque([(start_y, start_x)])
        count = 0
        total_score = 0.0
        min_y = max_y = start_y
        min_x = max_x = start_x
        while pending:
            y, x = pending.pop()
            count += 1
            total_score += float(score_map[y, x])
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
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
                        pending.append((next_y, next_x))
        if count < minimum_area:
            continue
        x1, y1 = float(min_x), float(min_y)
        x2, y2 = float(max_x + 1), float(max_y + 1)
        pad_x = (x2 - x1) * expand_ratio
        pad_y = (y2 - y1) * expand_ratio
        candidates.append(
            (
                total_score / count,
                count,
                (
                    max(0.0, x1 - pad_x),
                    max(0.0, y1 - pad_y),
                    min(float(width), x2 + pad_x),
                    min(float(height), y2 + pad_y),
                ),
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [box for _, _, box in candidates[:maximum_components]]


def _xyxy_to_normalized_cxcywh(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = box
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    return [
        (x1 + box_width / 2.0) / width,
        (y1 + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    ]


def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return torch.ones_like(mask, dtype=torch.bool)
    values = mask.float().view(1, 1, *mask.shape)
    kernel = 2 * width + 1
    dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - values, kernel_size=kernel, stride=1, padding=width)
    return (dilated - eroded).squeeze() > 0


class Sam3LoveDABenchmark:
    """Resumable full LoveDA validation benchmark for SegEarth-OV3."""

    def __init__(self, predictor: SegEarthOV3Predictor) -> None:
        self.predictor = predictor

    def run(
        self,
        data_root: str | Path,
        output_dir: str | Path,
        *,
        max_images: int | None = None,
        progress_every: int = 10,
    ) -> dict[str, object]:
        if max_images is not None and max_images < 1:
            raise ValueError("max_images must be positive when provided.")
        if progress_every < 1:
            raise ValueError("progress_every must be positive.")

        samples = discover_loveda_samples(data_root)
        if max_images is not None:
            samples = samples[:max_images]
        output_dir = Path(output_dir).expanduser().resolve()
        predictions_dir = output_dir / "predictions" / "segearth-ov3"
        output_dir.mkdir(parents=True, exist_ok=True)
        signature = self._signature(data_root, samples)
        _ensure_signature(output_dir / "signature.json", signature)
        if self.predictor.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.predictor.device)

        started = time.perf_counter()
        matrix = LoveDAConfusionMatrix()
        new_images = 0
        resumed_images = 0
        inference_seconds = 0.0

        for index, sample in enumerate(samples, start=1):
            with Image.open(sample.mask_path) as mask_image:
                target = np.asarray(mask_image).copy()
            prediction_path = predictions_dir / sample.key
            prediction = _load_prediction(prediction_path, target.shape)
            if prediction is None:
                with Image.open(sample.image_path) as image:
                    image_started = time.perf_counter()
                    prediction, _ = self.predictor.predict(image)
                    inference_seconds += time.perf_counter() - image_started
                _save_prediction(prediction_path, prediction)
                new_images += 1
            else:
                resumed_images += 1
            matrix.update(prediction, target)

            if index % progress_every == 0 or index == len(samples):
                _write_json_atomic(
                    output_dir / "results.json",
                    self._result(
                        "running" if index < len(samples) else "complete",
                        data_root,
                        len(samples),
                        index,
                        matrix,
                        new_images,
                        resumed_images,
                        inference_seconds,
                        time.perf_counter() - started,
                        output_dir,
                    ),
                )

        return self._result(
            "complete",
            data_root,
            len(samples),
            len(samples),
            matrix,
            new_images,
            resumed_images,
            inference_seconds,
            time.perf_counter() - started,
            output_dir,
        )

    def _signature(self, data_root: str | Path, samples: Sequence[Any]) -> dict[str, object]:
        sample_digest = hashlib.sha256("\n".join(sample.key for sample in samples).encode("utf-8")).hexdigest()
        return {
            "method": "SegEarth-OV3 official SAM 3 dual-head fusion",
            "dataset": "LoveDA validation",
            "data_root": str(Path(data_root).expanduser().resolve()),
            "sample_count": len(samples),
            "sample_keys_sha256": sample_digest,
            "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
            "prompt_lines": list(OFFICIAL_LOVEDA_PROMPTS),
            "sam3": self.predictor.config.signature(),
        }

    def _result(
        self,
        status: str,
        data_root: str | Path,
        total_images: int,
        processed_images: int,
        matrix: LoveDAConfusionMatrix,
        new_images: int,
        resumed_images: int,
        inference_seconds: float,
        wall_seconds: float,
        output_dir: Path,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "status": status,
            "dataset": {
                "name": "LoveDA",
                "split": "validation",
                "data_root": str(Path(data_root).expanduser().resolve()),
                "total_images": total_images,
                "processed_images": processed_images,
                "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
            },
            "method": "SegEarth-OV3 official SAM 3 dual-head fusion",
            "classes": list(LOVEDA_CLASS_NAMES),
            "prompt_lines": list(OFFICIAL_LOVEDA_PROMPTS),
            "metrics": matrix.summary(),
            "timing": {
                "wall_seconds": round(wall_seconds, 4),
                "new_inference_seconds": round(inference_seconds, 4),
                "new_images": new_images,
                "resumed_images": resumed_images,
                "mean_new_image_seconds": round(inference_seconds / new_images, 4) if new_images else None,
            },
            "config": asdict(self.predictor.config) | {
                "root": str(self.predictor.config.root),
                "checkpoint_path": str(self.predictor.config.checkpoint_path),
                "bpe_path": str(self.predictor.config.bpe_path),
                "class_file": str(self.predictor.config.class_file),
            },
            "checkpoints": self.predictor.config.signature(),
            "outputs": {
                "root": str(output_dir),
                "predictions": str(output_dir / "predictions" / "segearth-ov3"),
                "results": str(output_dir / "results.json"),
            },
            "protocol_note": (
                "This evaluates the labeled LoveDA validation split using the official SegEarth-OV3 SAM 3 "
                "query, dual-head fusion, presence filtering, synonym aggregation, and background threshold logic. "
                "Ground-truth IDs 0 and 255 are ignored."
            ),
        }
        if self.predictor.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.predictor.device) / (1024 * 1024), 2
            )
        return result


def _read_query_file(path: Path) -> tuple[list[str], list[int]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No SAM 3 prompts found in {path}.")
    query_words: list[str] = []
    query_indices: list[int] = []
    for class_index, line in enumerate(lines):
        aliases = [alias.strip() for alias in line.split(",") if alias.strip()]
        if not aliases:
            raise ValueError(f"Empty prompt line for class {class_index} in {path}.")
        query_words.extend(aliases)
        query_indices.extend([class_index] * len(aliases))
    return query_words, query_indices


def _file_manifest(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_prediction(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            prediction = np.asarray(image).copy()
    except (OSError, ValueError):
        return None
    if prediction.ndim != 2 or prediction.shape != expected_shape:
        return None
    return prediction


def _save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(temporary)
    temporary.replace(path)


def _ensure_signature(path: Path, signature: dict[str, object]) -> None:
    normalized = json.loads(json.dumps(signature, ensure_ascii=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"Existing benchmark configuration at {path} differs from this run. "
                "Use a new output directory to avoid mixing predictions."
            )
        return
    _write_json_atomic(path, normalized)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
