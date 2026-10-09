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


COST_AGGREGATION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CostAggregationConfig:
    """Small class-agnostic decoder for DINOv3 text-image cost maps.

    The decoder operates on a variable number of text classes. Spatial attention
    refines each class map independently, then class attention resolves local
    competition using the corresponding text embeddings. Its projections and
    reduction layers are shared across classes, preserving open-vocabulary use at
    inference time.
    """

    feature_dim: int
    hidden_dim: int = 128
    context_blocks: int = 6
    attention_heads: int = 8
    window_size: int = 7
    dropout: float = 0.0

    def validate(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if self.hidden_dim < 16:
            raise ValueError("hidden_dim must be at least 16.")
        if self.context_blocks < 0:
            raise ValueError("context_blocks must be non-negative.")
        if self.attention_heads < 1 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by a positive attention_heads value.")
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class _WindowSpatialAggregation(nn.Module):
    """Class-wise local attention with DINO patch features as visual guidance."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        window_size: int,
        dropout: float,
        shifted: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.window_size = window_size
        self.shifted = shifted
        self.norm = nn.LayerNorm(hidden_dim)
        self.guidance_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, features: Tensor, visual_guidance: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, D, C, H, W].")
        batch, channels, classes, height, width = features.shape
        if channels != self.hidden_dim or visual_guidance.shape != (batch, channels, height, width):
            raise ValueError("Spatial aggregation feature or guidance shape is invalid.")

        tokens = features.permute(0, 2, 3, 4, 1).reshape(batch * classes, height, width, channels)
        guidance = visual_guidance.permute(0, 2, 3, 1)
        guidance = guidance[:, None].expand(batch, classes, height, width, channels)
        guidance = guidance.reshape(batch * classes, height, width, channels)

        shift = self.window_size // 2 if self.shifted and min(height, width) > self.window_size else 0
        if shift:
            tokens = torch.roll(tokens, shifts=(-shift, -shift), dims=(1, 2))
            guidance = torch.roll(guidance, shifts=(-shift, -shift), dims=(1, 2))
        tokens, padded_height, padded_width = _pad_channel_last(tokens, self.window_size)
        guidance, _, _ = _pad_channel_last(guidance, self.window_size)
        token_windows = _partition_windows(tokens, self.window_size)
        guidance_windows = _partition_windows(guidance, self.window_size)

        normalized = self.norm(token_windows)
        normalized_guidance = self.guidance_norm(guidance_windows)
        qk_input = torch.cat((normalized, normalized_guidance), dim=-1)
        query = _split_heads(self.query(qk_input), self.heads)
        key = _split_heads(self.key(qk_input), self.heads)
        value = _split_heads(self.value(normalized), self.heads)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.output_dropout.p if self.training else 0.0,
        )
        attention = _merge_heads(attention)
        token_windows = token_windows + self.output_dropout(self.output(attention))
        token_windows = token_windows + self.ffn(self.ffn_norm(token_windows))

        tokens = _reverse_windows(
            token_windows,
            batch * classes,
            padded_height,
            padded_width,
            channels,
            self.window_size,
        )
        tokens = tokens[:, :height, :width]
        if shift:
            tokens = torch.roll(tokens, shifts=(shift, shift), dims=(1, 2))
        return tokens.reshape(batch, classes, height, width, channels).permute(0, 4, 1, 2, 3)


class _ClassAggregation(nn.Module):
    """Per-pixel class attention conditioned on the prompted text embeddings."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.guidance_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, features: Tensor, text_guidance: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, D, C, H, W].")
        batch, channels, classes, height, width = features.shape
        if channels != self.hidden_dim or text_guidance.shape != (classes, channels):
            raise ValueError("Class aggregation feature or text guidance shape is invalid.")

        tokens = features.permute(0, 3, 4, 2, 1).reshape(batch * height * width, classes, channels)
        guidance = text_guidance.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        normalized = self.norm(tokens)
        normalized_guidance = self.guidance_norm(guidance)
        qk_input = torch.cat((normalized, normalized_guidance), dim=-1)
        query = _split_heads(self.query(qk_input), self.heads)
        key = _split_heads(self.key(qk_input), self.heads)
        value = _split_heads(self.value(normalized), self.heads)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.output_dropout.p if self.training else 0.0,
        )
        tokens = tokens + self.output_dropout(self.output(_merge_heads(attention)))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.reshape(batch, height, width, classes, channels).permute(0, 4, 3, 1, 2)


class _AggregationBlock(nn.Module):
    def __init__(self, config: CostAggregationConfig) -> None:
        super().__init__()
        # A regular plus shifted window pair gives every class-cost map local
        # context across adjacent windows before class competition is resolved.
        self.spatial_regular = _WindowSpatialAggregation(
            config.hidden_dim,
            config.attention_heads,
            config.window_size,
            config.dropout,
            shifted=False,
        )
        self.spatial_shifted = _WindowSpatialAggregation(
            config.hidden_dim,
            config.attention_heads,
            config.window_size,
            config.dropout,
            shifted=True,
        )
        self.classes = _ClassAggregation(config.hidden_dim, config.attention_heads, config.dropout)

    def forward(self, features: Tensor, text_guidance: Tensor, visual_guidance: Tensor) -> Tensor:
        features = self.spatial_regular(features, visual_guidance)
        features = self.spatial_shifted(features, visual_guidance)
        return self.classes(features, text_guidance)


class CostAggregationDecoder(nn.Module):
    """CAFe-style open-vocabulary cost aggregation over frozen DINOv3 features.

    Unlike a closed-set classifier, all convolutions are shared over the prompted
    class dimension. Therefore a checkpoint trained on external source classes
    can still consume an arbitrary set of text prototypes during inference.
    """

    def __init__(self, config: CostAggregationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.cost_projection = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.visual_projection = nn.Sequential(
            nn.Conv2d(config.feature_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(nn.Linear(config.feature_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(_AggregationBlock(config) for _ in range(config.context_blocks))
        self.reducer = nn.Sequential(
            nn.Conv2d(hidden, max(hidden // 2, 8), kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(max(hidden // 2, 8)), max(hidden // 2, 8)),
            nn.GELU(),
            nn.Conv2d(max(hidden // 2, 8), 1, kernel_size=1),
        )
        # Start exactly at the frozen DINOv3 cosine-similarity cost volume.
        nn.init.zeros_(self.reducer[-1].weight)
        nn.init.zeros_(self.reducer[-1].bias)

    def forward(self, patch_features: Tensor, text_features: Tensor) -> Tensor:
        if patch_features.ndim != 4 or patch_features.shape[1] != self.config.feature_dim:
            raise ValueError("patch_features must have shape [B, feature_dim, H, W].")
        if text_features.ndim != 2 or text_features.shape[1] != self.config.feature_dim:
            raise ValueError("text_features must have shape [classes, feature_dim].")
        if text_features.shape[0] < 1:
            raise ValueError("At least one text feature is required.")

        base_cost = torch.einsum(
            "bdhw,cd->bchw",
            F.normalize(patch_features, dim=1),
            F.normalize(text_features, dim=1),
        )
        batch, classes, height, width = base_cost.shape
        projected = self.cost_projection(base_cost.reshape(batch * classes, 1, height, width))
        projected = projected.reshape(batch, classes, self.config.hidden_dim, height, width).permute(0, 2, 1, 3, 4)
        text_guidance = self.text_projection(F.normalize(text_features, dim=1))
        visual_guidance = self.visual_projection(F.normalize(patch_features, dim=1))
        for block in self.blocks:
            projected = block(projected, text_guidance, visual_guidance)
        residual = self.reducer(projected.permute(0, 2, 1, 3, 4).reshape(batch * classes, self.config.hidden_dim, height, width))
        residual = residual.reshape(batch, classes, height, width)
        return base_cost + residual


class CostAggregatedDINOTextSegmenter:
    """DINOv3 dino.txt with a learned, text-conditioned cost aggregation decoder."""

    patch_size = DINOTextSegmenter.patch_size
    method_name = "DINOv3 dino.txt cost-aggregation open-vocabulary decoder"
    implementation_note = (
        "The DINOv3 dino.txt backbone remains frozen. Only a class-agnostic "
        "cost aggregation decoder trained on external data changes text-image "
        "similarity maps, so inference remains DINOv3-based and open vocabulary."
    )

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        amp: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = load_cost_aggregation_checkpoint(self.checkpoint_path)
        config = CostAggregationConfig(**payload["decoder_config"])
        config.validate()
        self.base = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=False)
        visual_tuning = payload.get("visual_tuning_state_dict")
        if visual_tuning is not None:
            self.base.load_visual_tuning_state_dict(visual_tuning)
        if self.base.text_feature_dim != config.feature_dim:
            raise ValueError(
                "Cost aggregation checkpoint feature dimension does not match dino.txt: "
                f"{config.feature_dim} != {self.base.text_feature_dim}."
            )
        self.decoder = CostAggregationDecoder(config)
        self.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        self.decoder.to(self.base.device).eval().requires_grad_(False)
        self.decoder_config = config
        self.decoder_metadata = payload
        self.cost_aggregation_manifest = cost_aggregation_checkpoint_manifest(self.checkpoint_path, payload)

    @property
    def checkpoints(self) -> CheckpointConfig:
        return self.base.checkpoints

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def satellite(self) -> None:
        return None

    @property
    def visual_tuning_manifest(self) -> dict[str, object]:
        return self.base.visual_tuning_manifest

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_image(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        return self.base.encode_image(rgb)

    def encode_satellite_structure(self, rgb: Tensor) -> Tensor:
        return self.base.encode_satellite_structure(rgb)

    def similarity_logits(self, patch_features: Tensor, text_features: Tensor) -> Tensor:
        with torch.inference_mode(), self.base._autocast():
            return self.decoder(patch_features, text_features).float()


def make_cost_aggregation_checkpoint(
    *,
    decoder: CostAggregationDecoder,
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
        "format_version": COST_AGGREGATION_FORMAT_VERSION,
        "method": CostAggregatedDINOTextSegmenter.method_name,
        "decoder_config": asdict(decoder.config),
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


def load_cost_aggregation_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Cost aggregation checkpoint at {path} is not a mapping.")
    if payload.get("format_version") != COST_AGGREGATION_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cost aggregation checkpoint format at {path}: {payload.get('format_version')!r}."
        )
    if not isinstance(payload.get("decoder_config"), dict) or not isinstance(payload.get("decoder_state_dict"), dict):
        raise ValueError(f"Cost aggregation checkpoint at {path} is missing its architecture or weights.")
    if payload.get("visual_tuning_state_dict") is not None and not isinstance(payload["visual_tuning_state_dict"], dict):
        raise ValueError(f"Cost aggregation checkpoint at {path} has an invalid visual tuning state.")
    return payload


def cost_aggregation_checkpoint_manifest(
    path: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if payload is None:
        payload = load_cost_aggregation_checkpoint(path)
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
        "decoder_config": payload["decoder_config"],
        "source_dataset": payload.get("source_dataset"),
        "source_classes": payload.get("source_classes"),
        "training_epoch": payload.get("epoch"),
        "validation": payload.get("validation"),
        "visual_tuning_manifest": payload.get("visual_tuning_manifest"),
    }


def _pad_channel_last(features: Tensor, window_size: int) -> tuple[Tensor, int, int]:
    batch, height, width, channels = features.shape
    padded_height = ((height + window_size - 1) // window_size) * window_size
    padded_width = ((width + window_size - 1) // window_size) * window_size
    if (padded_height, padded_width) == (height, width):
        return features, padded_height, padded_width
    padded = F.pad(
        features.permute(0, 3, 1, 2),
        (0, padded_width - width, 0, padded_height - height),
        mode="replicate",
    )
    return padded.permute(0, 2, 3, 1), padded_height, padded_width


def _partition_windows(features: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = features.shape
    return (
        features.reshape(batch, height // window_size, window_size, width // window_size, window_size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )


def _reverse_windows(
    windows: Tensor,
    batch: int,
    height: int,
    width: int,
    channels: int,
    window_size: int,
) -> Tensor:
    return (
        windows.reshape(batch, height // window_size, width // window_size, window_size, window_size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


def _split_heads(features: Tensor, heads: int) -> Tensor:
    batch, tokens, channels = features.shape
    return features.reshape(batch, tokens, heads, channels // heads).transpose(1, 2)


def _merge_heads(features: Tensor) -> Tensor:
    return features.transpose(1, 2).reshape(features.shape[0], features.shape[2], -1)


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
