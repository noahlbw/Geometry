from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import CheckpointConfig, TLPConfig
from .inference import pil_to_tensor
from .model import DINOTextSegmenter, checkpoint_manifest
from .prompts import ClassSpec, serialize_class_specs
from .region_verifier import DINORegionVerifier, RegionVerifierConfig
from .sam3 import (
    Sam3Config,
    Sam3GeometryConfig,
    Sam3ImageContext,
    SegEarthOV3Predictor,
    _write_json_atomic,
)
from .sam3_dino import Sam3PromptBank, _class_probabilities_from_baseline, _pgrf_class_scores
from .tlp import text_aware_laplacian_propagation


@dataclass(frozen=True)
class AgentOVSSConfig:
    """Fixed, label-free quality profiles for dynamic-vocabulary OVSS."""

    profile: str = "fast"
    dino_input_resolution: int = 1024
    # Preserve every caller-supplied concept by default.  A bounded router is
    # an explicit latency/recall trade-off, never a silent default.
    maximum_active_classes: int = 0
    maximum_prompts_per_class: int = 1
    include_aerial_variants: bool = False
    # PGRF is a historical score attenuation experiment.  The agent default
    # must preserve SAM3's official dual-head confidence before DINO verifies
    # only ambiguous regions.
    use_pgrf: bool = False
    geometry: Sam3GeometryConfig = field(default_factory=Sam3GeometryConfig)
    region_verifier: RegionVerifierConfig = field(default_factory=RegionVerifierConfig)
    use_dino_tlp: bool = False
    use_satellite_structure: bool = False
    image_cache_size: int = 1

    @classmethod
    def for_profile(cls, profile: str) -> "AgentOVSSConfig":
        if profile == "fast":
            return cls(
                profile=profile,
                maximum_active_classes=0,
                maximum_prompts_per_class=1,
                use_pgrf=False,
                geometry=Sam3GeometryConfig(enabled=False),
                region_verifier=RegionVerifierConfig(enabled=True),
                use_dino_tlp=False,
                use_satellite_structure=False,
            )
        if profile == "balanced":
            return cls(
                profile=profile,
                maximum_active_classes=0,
                maximum_prompts_per_class=1,
                use_pgrf=False,
                geometry=Sam3GeometryConfig(
                    enabled=True,
                    maximum_components=1,
                    maximum_prompt_calls=2,
                ),
                region_verifier=RegionVerifierConfig(enabled=True),
                use_dino_tlp=False,
                use_satellite_structure=False,
            )
        if profile == "accurate":
            return cls(
                profile=profile,
                maximum_active_classes=0,
                maximum_prompts_per_class=2,
                include_aerial_variants=True,
                use_pgrf=False,
                geometry=Sam3GeometryConfig(
                    enabled=True,
                    maximum_components=2,
                    maximum_prompt_calls=4,
                ),
                region_verifier=RegionVerifierConfig(enabled=True),
                use_dino_tlp=True,
                use_satellite_structure=True,
            )
        raise ValueError("Agent OVSS profile must be one of: fast, balanced, accurate.")

    def validate(self) -> None:
        if self.profile not in {"fast", "balanced", "accurate"}:
            raise ValueError("Agent OVSS profile must be one of: fast, balanced, accurate.")
        if self.dino_input_resolution < 16 or self.dino_input_resolution % 16:
            raise ValueError("DINO input resolution must be a multiple of 16 and at least 16.")
        if self.maximum_active_classes < 0:
            raise ValueError("maximum_active_classes must be non-negative; zero keeps every supplied class.")
        if self.maximum_prompts_per_class < 1:
            raise ValueError("maximum_prompts_per_class must be positive.")
        if self.image_cache_size < 0:
            raise ValueError("image_cache_size must be non-negative.")
        self.geometry.validate()
        self.region_verifier.validate()

    def signature(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _ImageCacheEntry:
    sam3_context: Sam3ImageContext
    dino_rgb: torch.Tensor
    patch_features: torch.Tensor
    anchor: torch.Tensor
    satellite_structure: torch.Tensor | None = None


class AgentSam3DinoSession:
    """Persistent, dynamic-vocabulary SAM3+DINO OVSS session for an agent.

    A session owns model weights and can cache one or more image encodings. It
    processes calls serially because SAM3's prompt state is mutable.
    """

    def __init__(
        self,
        sam3_config: Sam3Config,
        dino_checkpoints: CheckpointConfig,
        config: AgentOVSSConfig,
        tlp_config: TLPConfig,
    ) -> None:
        config.validate()
        tlp_config.validate()
        self.sam3 = SegEarthOV3Predictor(sam3_config)
        self.dino = DINOTextSegmenter(
            dino_checkpoints,
            device=str(self.sam3.device),
            amp=sam3_config.amp,
            use_satellite=config.use_satellite_structure,
        )
        self.config = config
        self.tlp_config = tlp_config
        self.region_verifier = DINORegionVerifier(config.region_verifier)
        self._text_cache: dict[tuple[tuple[str, tuple[str, ...]], ...], torch.Tensor] = {}
        self._class_text_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}
        self._image_cache: OrderedDict[str, _ImageCacheEntry] = OrderedDict()

    @property
    def device(self) -> torch.device:
        return self.sam3.device

    def close(self) -> None:
        self._image_cache.clear()
        self._text_cache.clear()
        self._class_text_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """Run the complete path once before a persistent service accepts work.

        This intentionally does not populate either caller-visible cache.  It
        moves CUDA kernel initialization and allocator growth out of the first
        real image request while exercising both SAM3 and DINO verification.
        """

        side = max(16, self.config.dino_input_resolution)
        image = Image.new("RGB", (side, side), color=(96, 112, 88))
        self.segment(
            image,
            (
                ClassSpec("background", ("background", "other")),
                ClassSpec("terrain", ("terrain", "land cover")),
            ),
        )

    def segment(
        self,
        image: Image.Image,
        classes: Sequence[ClassSpec],
        *,
        image_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Segment one image using only dynamically routed supplied classes."""

        normalized_classes, background_added = _normalize_classes(classes)
        started = time.perf_counter()
        entry, image_cache_hit, image_prepare_seconds = self._image_entry(image, image_key)

        routing_started = time.perf_counter()
        text_features, text_cache_hit, text_class_cache_hits = self._text_features(normalized_classes)
        global_scores = entry.anchor @ text_features.T
        active_indices = _active_indices(global_scores, len(normalized_classes), self.config.maximum_active_classes)
        active_classes = tuple(normalized_classes[index] for index in active_indices)
        inactive_indices = tuple(index for index in range(len(normalized_classes)) if index not in active_indices)
        active_text_features = text_features[list(active_indices)]
        prompt_bank = _build_agent_prompt_bank(
            active_classes,
            maximum_prompts_per_class=self.config.maximum_prompts_per_class,
            include_aerial_variants=self.config.include_aerial_variants,
        )
        routing_seconds = time.perf_counter() - routing_started

        sam3_started = time.perf_counter()
        components = self.sam3.predict_components_from_context(
            entry.sam3_context,
            query_words=prompt_bank.query_words,
            query_class_indices=prompt_bank.query_class_indices,
            geometry=self.config.geometry,
            class_count=len(active_classes),
        )
        sam_scores = (
            _pgrf_class_scores(components, len(active_classes))
            if self.config.use_pgrf
            else _class_probabilities_from_baseline(components, len(active_classes))
        )
        maximum, local_labels = sam_scores.max(dim=0)
        sam3_seconds = time.perf_counter() - sam3_started

        verification_started = time.perf_counter()
        region_diagnostics: dict[str, int] | None = None
        if self.config.region_verifier.enabled:
            dino_logits = self.dino.similarity_logits(entry.patch_features, active_text_features)
            if self.config.use_dino_tlp:
                structure = self._satellite_structure(entry)
                dino_logits, _ = text_aware_laplacian_propagation(
                    dino_logits,
                    F.interpolate(entry.dino_rgb, size=dino_logits.shape[-2:], mode="area"),
                    active_text_features,
                    self.tlp_config,
                    structure,
                )
            local_labels, diagnostics = self.region_verifier.verify(
                sam_scores,
                entry.patch_features,
                active_text_features,
                entry.anchor,
                dino_logits=dino_logits.squeeze(0).float(),
            )
            region_diagnostics = diagnostics.summary()
        verification_seconds = time.perf_counter() - verification_started

        local_labels = local_labels.clone()
        local_labels[maximum < self.sam3.config.probability_threshold] = 0
        index_map = torch.tensor(active_indices, dtype=torch.long, device=local_labels.device)
        labels = index_map[local_labels].to(torch.uint8).cpu().numpy()
        confidence = maximum.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        aggregation = "PGRF" if self.config.use_pgrf else "official SAM3 dual-head aggregation"
        metadata: dict[str, object] = {
            "method": f"dynamic {aggregation} + bounded geometry + DINOv3 region verification",
            "profile": self.config.profile,
            "background_added": background_added,
            "input_class_count": len(normalized_classes),
            "active_class_count": len(active_classes),
            "active_class_indices": list(active_indices),
            "active_class_names": [spec.name for spec in active_classes],
            "inactive_class_indices": list(inactive_indices),
            "inactive_class_names": [normalized_classes[index].name for index in inactive_indices],
            "global_class_scores": [round(float(score), 6) for score in global_scores.detach().cpu().tolist()],
            "prompt_bank": prompt_bank.signature(),
            "geometry_attempted": components.geometry_attempted,
            "geometry_accepted": components.geometry_accepted,
            "region_verification": region_diagnostics,
            "cache": {
                "image_hit": image_cache_hit,
                "text_hit": text_cache_hit,
                "text_class_hits": text_class_cache_hits,
                "image_entries": len(self._image_cache),
            },
            "timing_seconds": {
                "image_prepare": round(image_prepare_seconds, 4),
                "class_routing": round(routing_seconds, 4),
                "sam3_grounding": round(sam3_seconds, 4),
                "dino_verification": round(verification_seconds, 4),
                "total": round(time.perf_counter() - started, 4),
            },
        }
        return labels, confidence, metadata

    def run_file(
        self,
        image_path: str | Path,
        classes: Sequence[ClassSpec],
        output_dir: str | Path,
        *,
        image_key: str | None = None,
    ) -> dict[str, object]:
        request_started = time.perf_counter()
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        stat = path.stat()
        resolved_key = image_key or f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        labels, confidence, diagnostics = self.segment(image, classes, image_key=resolved_key)
        normalized_classes, _ = _normalize_classes(classes)
        labels_path = output / "labels.png"
        confidence_path = output / "confidence.png"
        _save_uint8(labels_path, labels)
        _save_uint8(confidence_path, confidence)
        timing = diagnostics.get("timing_seconds")
        if isinstance(timing, dict):
            timing["end_to_end"] = round(time.perf_counter() - request_started, 4)
        result: dict[str, object] = {
            "image": str(path),
            "width": image.width,
            "height": image.height,
            "classes": serialize_class_specs(normalized_classes),
            "class_id_mapping": [
                {"id": index, "name": spec.name}
                for index, spec in enumerate(normalized_classes)
            ],
            "diagnostics": diagnostics,
            "config": {
                "agent_ovss": self.config.signature(),
                "tlp": asdict(self.tlp_config),
            },
            "checkpoints": {
                "sam3": self.sam3.config.signature(),
                "dino": checkpoint_manifest(self.dino.checkpoints),
            },
            "outputs": {
                "labels": str(labels_path),
                "confidence": str(confidence_path),
                "metadata": str(output / "metadata.json"),
            },
            "protocol_note": (
                "Training-free dynamic-vocabulary OVSS. DINO routes classes and verifies SAM3 regions; "
                "it does not paint a dense posterior into the output."
            ),
        }
        if self.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2
            )
        _write_json_atomic(output / "metadata.json", result)
        return result

    def _text_features(self, classes: Sequence[ClassSpec]) -> tuple[torch.Tensor, bool, int]:
        key = tuple((spec.name, spec.synonyms) for spec in classes)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached, True, len(classes)

        missing = [
            (class_key, spec)
            for class_key, spec in zip(key, classes)
            if class_key not in self._class_text_cache
        ]
        class_cache_hits = len(classes) - len(missing)
        if missing:
            encoded = self.dino.encode_text([spec for _, spec in missing])
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise RuntimeError("DINO text encoder returned an unexpected class feature shape.")
            for (class_key, _), feature in zip(missing, encoded):
                # Rows otherwise retain the temporary batch storage in the
                # long-lived cache.  Each prototype is small, so copy it once.
                self._class_text_cache[class_key] = feature.clone()

        features = torch.stack([self._class_text_cache[class_key] for class_key in key], dim=0)
        self._text_cache[key] = features
        return features, False, class_cache_hits

    def _image_entry(
        self,
        image: Image.Image,
        image_key: str | None,
    ) -> tuple[_ImageCacheEntry, bool, float]:
        if image_key is not None and image_key in self._image_cache:
            entry = self._image_cache.pop(image_key)
            self._image_cache[image_key] = entry
            return entry, True, 0.0

        started = time.perf_counter()
        dino_image = _resize_for_dino(image, self.config.dino_input_resolution)
        rgb = pil_to_tensor(dino_image, self.dino.device)
        patch_features, anchor = self.dino.encode_image(rgb)
        entry = _ImageCacheEntry(
            sam3_context=self.sam3.prepare_image(image),
            dino_rgb=rgb,
            patch_features=patch_features.float(),
            anchor=anchor.squeeze(0).float(),
        )
        elapsed = time.perf_counter() - started
        if image_key is not None and self.config.image_cache_size:
            self._image_cache[image_key] = entry
            while len(self._image_cache) > self.config.image_cache_size:
                self._image_cache.popitem(last=False)
        return entry, False, elapsed

    def _satellite_structure(self, entry: _ImageCacheEntry) -> torch.Tensor | None:
        if not self.config.use_satellite_structure or self.dino.satellite is None:
            return None
        if entry.satellite_structure is None:
            entry.satellite_structure = self.dino.encode_satellite_structure(entry.dino_rgb)
        return entry.satellite_structure


def _normalize_classes(classes: Sequence[ClassSpec]) -> tuple[tuple[ClassSpec, ...], bool]:
    if not classes:
        raise ValueError("At least one foreground class is required.")
    background = [spec for spec in classes if spec.name.casefold() == "background"]
    if len(background) > 1:
        raise ValueError("Only one class may be named 'background'.")
    foreground = [spec for spec in classes if spec.name.casefold() != "background"]
    if not foreground:
        raise ValueError("At least one non-background class is required.")
    if background:
        return tuple(background + foreground), False
    return (ClassSpec("background", ("background", "other")), *foreground), True


def _active_indices(
    global_scores: torch.Tensor,
    class_count: int,
    maximum_active_classes: int,
) -> tuple[int, ...]:
    if global_scores.ndim != 1 or global_scores.numel() != class_count:
        raise ValueError("DINO global scores must contain one value for each class.")
    foreground = class_count - 1
    if maximum_active_classes == 0 or maximum_active_classes >= foreground:
        return tuple(range(class_count))
    selected = torch.topk(global_scores[1:], k=maximum_active_classes).indices + 1
    return (0, *(int(index) for index in selected.tolist()))


def _build_agent_prompt_bank(
    classes: Sequence[ClassSpec],
    *,
    maximum_prompts_per_class: int,
    include_aerial_variants: bool,
) -> Sam3PromptBank:
    words: list[str] = []
    indices: list[int] = []
    for class_index, spec in enumerate(classes):
        candidates = [spec.name, *spec.synonyms]
        if include_aerial_variants:
            candidates.extend(
                (
                    f"aerial view of {spec.name}",
                    f"overhead view of {spec.name}",
                    f"{spec.name} seen from above",
                )
            )
        selected = list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))[
            :maximum_prompts_per_class
        ]
        if not selected:
            raise ValueError(f"No SAM3 prompt remains for class '{spec.name}'.")
        words.extend(selected)
        indices.extend([class_index] * len(selected))
    return Sam3PromptBank(
        query_words=tuple(words),
        query_class_indices=tuple(indices),
        source="dynamic-agent-vocabulary",
    )


def _resize_for_dino(image: Image.Image, target_long_side: int) -> Image.Image:
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("DINO region verification requires a non-empty image.")
    scale = target_long_side / max(width, height)
    resized_width = max(16, int(round(width * scale / 16.0)) * 16)
    resized_height = max(16, int(round(height * scale / 16.0)) * 16)
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def _save_uint8(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("OVSS output must be a two-dimensional uint8 array.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(values, mode="L").save(temporary)
    temporary.replace(path)
