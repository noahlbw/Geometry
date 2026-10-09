from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CheckpointConfig
from .model import DINOTextSegmenter, checkpoint_manifest
from .ov_adapter import DenseAdapterConfig, DenseTextAdapter, load_adapter_checkpoint
from .prompts import ClassSpec


MASK_OV_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MaskOVConfig:
    """Architecture of the multi-scale, class-agnostic DINO mask decoder."""

    feature_dim: int
    adapter_hidden_dim: int = 512
    adapter_context_blocks: int = 4
    fusion_dim: int = 256
    num_queries: int = 64
    attention_heads: int = 8
    query_layers: int = 2
    dropout: float = 0.1
    feature_layers: tuple[int, ...] = (5, 11, 17, 23)
    residual_scale: float = 0.05

    def validate(self) -> None:
        if self.feature_dim < 1 or self.adapter_hidden_dim < 16 or self.fusion_dim < 32:
            raise ValueError("Mask OV feature dimensions are invalid.")
        if self.adapter_context_blocks < 0 or self.num_queries < 1 or self.query_layers < 1:
            raise ValueError("Mask OV depth and query counts are invalid.")
        if self.attention_heads < 1 or self.fusion_dim % self.attention_heads:
            raise ValueError("fusion_dim must be divisible by a positive attention_heads value.")
        if not self.feature_layers or len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError("feature_layers must contain unique DINO block indices.")
        if not 0.0 <= self.dropout < 1.0 or self.residual_scale < 0:
            raise ValueError("Mask OV dropout or residual_scale is invalid.")


@dataclass(frozen=True)
class MaskOVImageFeatures:
    adapted_features: Tensor
    frozen_features: Tensor
    multiscale_features: tuple[Tensor, ...]
    satellite_features: Tensor
    rgb_grid: Tensor


@dataclass(frozen=True)
class MaskDecoderOutput:
    residual_logits: Tensor
    mask_logits: Tensor
    class_scores: Tensor
    semantic_mask: Tensor


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        update = self.depthwise(self.activation(self.norm(features)))
        update = self.dropout(self.activation(self.pointwise(update)))
        return features + update


class MultiScaleMaskQueryDecoder(nn.Module):
    """Mask2Former-style query decoder with a variable text vocabulary.

    Queries are class agnostic. Text embeddings score each query, while the
    query-to-pixel masks provide the spatial prior missing from a single DINO
    patch-cost map. All projections are shared over the prompted class axis.
    """

    def __init__(self, config: MaskOVConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.fusion_dim
        self.feature_projections = nn.ModuleList(
            [nn.Conv2d(config.feature_dim, hidden, 1, bias=False) for _ in config.feature_layers]
        )
        self.satellite_projection = nn.Conv2d(config.feature_dim, hidden, 1, bias=False)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            _SpatialResidualBlock(hidden, config.dropout),
            _SpatialResidualBlock(hidden, config.dropout),
        )
        self.query_tokens = nn.Parameter(torch.randn(config.num_queries, hidden) * 0.02)
        self.query_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(config.query_layers))
        self.cross_attention = nn.ModuleList(
            nn.MultiheadAttention(hidden, config.attention_heads, dropout=config.dropout, batch_first=True)
            for _ in range(config.query_layers)
        )
        self.query_ffns = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden * 4, hidden),
            )
            for _ in range(config.query_layers)
        )
        self.mask_features = nn.Sequential(nn.Conv2d(hidden, hidden, 1, bias=False), nn.GELU())
        self.mask_embeddings = nn.Linear(hidden, hidden)
        self.text_embeddings = nn.Linear(hidden, config.feature_dim)
        self.query_bias = nn.Linear(hidden, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale)))

    def forward(
        self,
        multiscale_features: Sequence[Tensor],
        satellite_features: Tensor,
        rgb_grid: Tensor,
        text_features: Tensor,
    ) -> MaskDecoderOutput:
        if len(multiscale_features) != len(self.config.feature_layers):
            raise ValueError("The number of DINO feature maps does not match feature_layers.")
        reference_shape = multiscale_features[-1].shape
        if len(reference_shape) != 4:
            raise ValueError("DINO feature maps must have shape [B, D, H, W].")
        batch, _, height, width = reference_shape
        if satellite_features.shape != reference_shape or rgb_grid.shape != (batch, 3, height, width):
            raise ValueError("SAT or RGB guidance has an incompatible shape.")
        if text_features.ndim != 2 or text_features.shape[1] != self.config.feature_dim:
            raise ValueError("text_features must have shape [classes, feature_dim].")

        fused = self.satellite_projection(satellite_features)
        fused = fused + self.rgb_projection(rgb_grid)
        for projection, features in zip(self.feature_projections, multiscale_features):
            if features.shape != reference_shape:
                raise ValueError("All DINO feature maps must share their spatial shape and channel count.")
            fused = fused + projection(features)
        fused = self.fusion(fused / float(len(multiscale_features) + 2))

        spatial_tokens = fused.flatten(2).transpose(1, 2)
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        for norm, attention, ffn in zip(self.query_norms, self.cross_attention, self.query_ffns):
            attended, _ = attention(norm(queries), spatial_tokens, spatial_tokens, need_weights=False)
            queries = queries + attended
            queries = queries + ffn(queries)

        mask_features = F.normalize(self.mask_features(fused), dim=1)
        mask_embeddings = F.normalize(self.mask_embeddings(queries), dim=-1)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embeddings, mask_features)
        query_text = F.normalize(self.text_embeddings(queries), dim=-1)
        class_scores = torch.einsum("bqd,cd->bqc", query_text, F.normalize(text_features, dim=-1))
        class_scores = class_scores + self.query_bias(queries).squeeze(-1).unsqueeze(-1)
        combined = mask_logits.unsqueeze(2) + class_scores.unsqueeze(-1).unsqueeze(-1)
        residual = torch.logsumexp(combined, dim=1) - math.log(float(self.config.num_queries))
        weights = torch.softmax(class_scores.transpose(1, 2), dim=-1)
        semantic_mask = torch.einsum("bcq,bqhw->bchw", weights, torch.sigmoid(mask_logits))
        return MaskDecoderOutput(
            residual_logits=self.residual_scale * residual,
            mask_logits=mask_logits,
            class_scores=class_scores,
            semantic_mask=semantic_mask,
        )


class MaskOVDINOTextSegmenter:
    """DINO.text + SAT multi-scale, class-agnostic mask-query OVSS model."""

    patch_size = DINOTextSegmenter.patch_size
    method_name = "DINOv3 dino.txt multi-scale mask-query open-vocabulary decoder"
    implementation_note = (
        "DINOv3 dino.txt text alignment is retained while DINO LVD/SAT multi-scale features, "
        "RGB guidance, and class-agnostic mask queries provide dense spatial masks."
    )

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cuda",
        amp: bool = True,
        mask_config: MaskOVConfig | None = None,
        adapter_config: DenseAdapterConfig | None = None,
    ) -> None:
        payload = load_mask_ov_checkpoint(checkpoint_path) if checkpoint_path is not None else None
        if payload is not None:
            mask_config = MaskOVConfig(
                **{**payload["mask_config"], "feature_layers": tuple(payload["mask_config"]["feature_layers"])}
            )
            adapter_config = DenseAdapterConfig(**payload["adapter_config"])
        if mask_config is None or adapter_config is None:
            raise ValueError("mask_config and adapter_config are required when no checkpoint is provided.")
        mask_config.validate()
        adapter_config.validate()
        self.base = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=True)
        if self.base.text_feature_dim != mask_config.feature_dim or adapter_config.feature_dim != mask_config.feature_dim:
            raise ValueError("Mask OV feature dimensions do not match DINOv3 dino.txt.")
        self.adapter = DenseTextAdapter(adapter_config).to(self.base.device)
        self.decoder = MultiScaleMaskQueryDecoder(mask_config).to(self.base.device)
        if payload is not None:
            self.adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
            self.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
            self.base.load_visual_tuning_state_dict(payload.get("visual_tuning_state_dict"))
        self.adapter.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)
        self.mask_config = mask_config
        self.adapter_config = adapter_config
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None
        self.mask_metadata = payload
        self.mask_manifest = mask_ov_checkpoint_manifest(self.checkpoint_path, payload) if payload is not None else None

    @property
    def checkpoints(self) -> CheckpointConfig:
        return self.base.checkpoints

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def satellite(self) -> nn.Module | None:
        return self.base.satellite

    @property
    def visual_tuning_manifest(self) -> dict[str, object]:
        return self.base.visual_tuning_manifest

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_image(self, rgb: Tensor) -> tuple[MaskOVImageFeatures, Tensor]:
        image_features, anchor = self._encode_features(rgb, train_visual_backbone=False)
        return image_features, anchor

    def encode_image_for_training(self, rgb: Tensor, *, train_visual_backbone: bool) -> MaskOVImageFeatures:
        features, _ = self._encode_features(rgb, train_visual_backbone=train_visual_backbone)
        return features

    def similarity_logits(self, image_features: MaskOVImageFeatures | Tensor, text_features: Tensor) -> Tensor:
        if isinstance(image_features, Tensor):
            return DINOTextSegmenter.similarity_logits(image_features, text_features)
        base_logits = torch.einsum(
            "bdhw,cd->bchw",
            image_features.adapted_features,
            F.normalize(text_features, dim=-1),
        )
        decoded = self.decoder(
            image_features.multiscale_features,
            image_features.satellite_features,
            image_features.rgb_grid,
            text_features,
        )
        return base_logits + decoded.residual_logits

    def mask_outputs(self, image_features: MaskOVImageFeatures, text_features: Tensor) -> MaskDecoderOutput:
        return self.decoder(
            image_features.multiscale_features,
            image_features.satellite_features,
            image_features.rgb_grid,
            text_features,
        )

    def encode_satellite_structure(self, rgb: Tensor) -> Tensor:
        return self.base.encode_satellite_structure(rgb)

    def _encode_features(
        self,
        rgb: Tensor,
        *,
        train_visual_backbone: bool,
    ) -> tuple[MaskOVImageFeatures, Tensor]:
        multiscale, anchor = self.base.encode_image_multiscale_for_mask(
            rgb,
            layers=self.mask_config.feature_layers,
            train_visual_backbone=train_visual_backbone,
        )
        frozen_features = multiscale[-1]
        satellite_features = self.base.encode_satellite_structure_for_adapter(rgb)
        with torch.autocast(
            device_type=self.base.device.type,
            dtype=torch.bfloat16,
            enabled=self.base.use_amp and self.base.device.type == "cuda",
        ):
            adapted_features = self.adapter(frozen_features, satellite_features)
        rgb_grid = F.interpolate(rgb.to(self.base.device), size=frozen_features.shape[-2:], mode="area")
        return (
            MaskOVImageFeatures(
                adapted_features=adapted_features,
                frozen_features=frozen_features,
                multiscale_features=multiscale,
                satellite_features=satellite_features,
                rgb_grid=rgb_grid,
            ),
            anchor,
        )


def make_mask_ov_checkpoint(
    *,
    adapter: DenseTextAdapter,
    decoder: MultiScaleMaskQueryDecoder,
    source_dataset: dict[str, Any],
    source_classes: Sequence[ClassSpec],
    training_config: dict[str, Any],
    dino_checkpoints: CheckpointConfig,
    epoch: int,
    validation: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    visual_tuning_state_dict: dict[str, dict[str, Tensor]] | None = None,
    visual_tuning_manifest: dict[str, object] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": MASK_OV_FORMAT_VERSION,
        "method": MaskOVDINOTextSegmenter.method_name,
        "mask_config": asdict(decoder.config),
        "adapter_config": asdict(adapter.config),
        "adapter_state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "decoder_state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in source_classes],
        "training_config": training_config,
        "dino_checkpoints": checkpoint_manifest(dino_checkpoints),
        "epoch": epoch,
        "validation": validation,
        "optimizer_state_dict": optimizer_state_dict,
        "visual_tuning_state_dict": visual_tuning_state_dict,
        "visual_tuning_manifest": visual_tuning_manifest,
    }


def load_mask_ov_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != MASK_OV_FORMAT_VERSION:
        raise ValueError(f"Unsupported DINO mask OV checkpoint format at {path}.")
    required = ("mask_config", "adapter_config", "adapter_state_dict", "decoder_state_dict")
    if any(not isinstance(payload.get(key), dict) for key in required):
        raise ValueError(f"DINO mask OV checkpoint at {path} is missing architecture or weights.")
    return payload


def mask_ov_checkpoint_manifest(path: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if payload is None:
        payload = load_mask_ov_checkpoint(path)
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
        "mask_config": payload["mask_config"],
        "adapter_config": payload["adapter_config"],
        "source_dataset": payload.get("source_dataset"),
        "source_classes": payload.get("source_classes"),
        "training_epoch": payload.get("epoch"),
        "validation": payload.get("validation"),
        "visual_tuning_manifest": payload.get("visual_tuning_manifest"),
    }


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
