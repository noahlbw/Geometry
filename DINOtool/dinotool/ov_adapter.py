from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CheckpointConfig
from .model import DINOTextSegmenter, checkpoint_manifest
from .prompts import ClassSpec


ADAPTER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DenseAdapterConfig:
    """Architecture of the small trainable bridge from DINO features to text space."""

    feature_dim: int
    hidden_dim: int = 256
    context_blocks: int = 2
    use_satellite_features: bool = True

    def validate(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if self.hidden_dim < 16:
            raise ValueError("hidden_dim must be at least 16.")
        if self.context_blocks < 0:
            raise ValueError("context_blocks must be non-negative.")


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        update = self.pointwise(self.activation(self.depthwise(self.norm(features))))
        return features + update


class DenseTextAdapter(nn.Module):
    """Map frozen DINO patch features into the dino.txt text embedding space.

    The output projection starts at zero, so the initial model exactly matches
    the LVD dino.txt visual features.  Supervised external data can then learn a
    remote-sensing correction without replacing the text-conditioned classifier.
    """

    def __init__(self, config: DenseAdapterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_dim = config.feature_dim * (2 if config.use_satellite_features else 1)
        groups = _group_count(config.hidden_dim)
        self.input_projection = nn.Conv2d(input_dim, config.hidden_dim, kernel_size=1, bias=False)
        self.input_norm = nn.GroupNorm(groups, config.hidden_dim)
        self.blocks = nn.Sequential(*[_SpatialResidualBlock(config.hidden_dim) for _ in range(config.context_blocks)])
        self.output_projection = nn.Conv2d(config.hidden_dim, config.feature_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, lvd_features: Tensor, satellite_features: Tensor | None = None) -> Tensor:
        if lvd_features.ndim != 4 or lvd_features.shape[1] != self.config.feature_dim:
            raise ValueError("lvd_features must be BCHW with the configured feature dimension.")
        if self.config.use_satellite_features:
            if satellite_features is None:
                raise ValueError("satellite_features are required by this adapter checkpoint.")
            if satellite_features.shape != lvd_features.shape:
                raise ValueError("LVD and SAT features must have identical BCHW shapes.")
            inputs = torch.cat((lvd_features, satellite_features), dim=1)
        else:
            if satellite_features is not None:
                raise ValueError("This adapter checkpoint was trained without satellite features.")
            inputs = lvd_features
        update = self.input_projection(inputs)
        update = self.blocks(F.gelu(self.input_norm(update)))
        return F.normalize(lvd_features + self.output_projection(update), dim=1)


class AdaptedDINOTextSegmenter:
    """DINOv3-SAT visual adaptation with an unchanged dino.txt text interface."""

    patch_size = DINOTextSegmenter.patch_size
    method_name = "DINOv3-SAT plus dino.txt dense open-vocabulary adapter"
    implementation_note = (
        "The DINOv3 LVD dino.txt visual backbone and SAT-493M backbone are frozen. "
        "Only the dense feature bridge was trained on an external remote-sensing source; "
        "class scores remain cosine similarities to dino.txt text prompts."
    )

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        adapter_checkpoint: str | Path,
        *,
        device: str = "cuda",
        amp: bool = True,
    ) -> None:
        self.adapter_checkpoint = Path(adapter_checkpoint).expanduser().resolve()
        payload = load_adapter_checkpoint(self.adapter_checkpoint)
        config = DenseAdapterConfig(**payload["adapter_config"])
        config.validate()
        self.base = DINOTextSegmenter(
            checkpoints,
            device=device,
            amp=amp,
            use_satellite=config.use_satellite_features,
        )
        if self.base.text_feature_dim != config.feature_dim:
            raise ValueError(
                "Adapter feature dimension does not match the loaded DINOv3 dino.txt checkpoint: "
                f"{config.feature_dim} != {self.base.text_feature_dim}."
            )
        self.adapter = DenseTextAdapter(config)
        self.adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
        self.adapter.to(self.base.device).eval().requires_grad_(False)
        self.adapter_config = config
        self.adapter_metadata = payload
        self.adapter_manifest = adapter_checkpoint_manifest(self.adapter_checkpoint, payload)

    @property
    def checkpoints(self) -> CheckpointConfig:
        return self.base.checkpoints

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def satellite(self) -> nn.Module | None:
        return self.base.satellite

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_image(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        lvd_features, anchor = self.base.encode_image(rgb)
        satellite = self.base.encode_satellite_structure(rgb) if self.adapter_config.use_satellite_features else None
        with torch.inference_mode(), self.base._autocast():
            features = self.adapter(lvd_features, satellite)
        return features, anchor

    def encode_satellite_structure(self, rgb: Tensor) -> Tensor:
        return self.base.encode_satellite_structure(rgb)

    @staticmethod
    def similarity_logits(patch_features: Tensor, text_features: Tensor) -> Tensor:
        return DINOTextSegmenter.similarity_logits(patch_features, text_features)


def load_adapter_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter checkpoint at {path} is not a mapping.")
    if payload.get("format_version") != ADAPTER_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported adapter checkpoint format at {path}: {payload.get('format_version')!r}."
        )
    if not isinstance(payload.get("adapter_config"), dict) or not isinstance(payload.get("adapter_state_dict"), dict):
        raise ValueError(f"Adapter checkpoint at {path} is missing its architecture or weights.")
    return payload


def adapter_checkpoint_manifest(path: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if payload is None:
        payload = load_adapter_checkpoint(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "format_version": payload["format_version"],
        "adapter_config": payload["adapter_config"],
        "source_dataset": payload.get("source_dataset"),
        "source_classes": payload.get("source_classes"),
        "training_epoch": payload.get("epoch"),
        "validation": payload.get("validation"),
    }


def model_checkpoint_manifest(model: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = checkpoint_manifest(model.checkpoints)
    adapter_manifest = getattr(model, "adapter_manifest", None)
    if adapter_manifest is not None:
        manifest["dense_open_vocabulary_adapter"] = adapter_manifest
    cost_aggregation_manifest = getattr(model, "cost_aggregation_manifest", None)
    if cost_aggregation_manifest is not None:
        manifest["cost_aggregation_decoder"] = cost_aggregation_manifest
    mask_manifest = getattr(model, "mask_manifest", None)
    if mask_manifest is not None:
        manifest["mask_query_decoder"] = mask_manifest
    return manifest


def make_adapter_checkpoint(
    *,
    adapter: DenseTextAdapter,
    source_dataset: dict[str, Any],
    source_classes: Sequence[ClassSpec],
    training_config: dict[str, Any],
    dino_checkpoints: CheckpointConfig,
    epoch: int,
    validation: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": ADAPTER_FORMAT_VERSION,
        "method": AdaptedDINOTextSegmenter.method_name,
        "adapter_config": asdict(adapter.config),
        "adapter_state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in source_classes],
        "training_config": training_config,
        "dino_checkpoints": checkpoint_manifest(dino_checkpoints),
        "epoch": epoch,
        "validation": validation,
        "optimizer_state_dict": optimizer_state_dict,
    }


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
