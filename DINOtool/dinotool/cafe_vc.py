"""CAFe-VC: visual-value cost reconstruction with the original CAFe output path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CafeVCConfig:
    arm: str = "vc"
    content_dim: int = 128
    heads: int = 4
    kernel_size: int = 7
    blocks: int = 2

    def validate(self) -> None:
        if self.arm not in {"plain", "cost_only", "concat", "vc"}:
            raise ValueError("arm must be plain, cost_only, concat, or vc")
        if self.content_dim < 8 or self.content_dim % self.heads:
            raise ValueError("content_dim must be positive and divisible by heads")
        if self.kernel_size < 3 or not self.kernel_size % 2:
            raise ValueError("kernel_size must be odd and at least 3")
        if self.blocks < 0 or (self.arm == "plain" and self.blocks):
            raise ValueError("plain CAFe uses zero reconstruction blocks")
        if self.arm != "plain" and self.blocks < 1:
            raise ValueError("reconstruction arms need at least one block")


def _group_count(channels: int) -> int:
    return math.gcd(channels, 8)


class VisualContentField(nn.Module):
    """Image-shared content tokens, independently projected from frozen DINO X/Y."""

    def __init__(self, content_dim: int) -> None:
        super().__init__()
        self.field = nn.Sequential(
            nn.Conv2d(2048, content_dim, 1),
            nn.GroupNorm(_group_count(content_dim), content_dim),
            nn.GELU(),
            nn.Conv2d(content_dim, content_dim, 3, padding=1, groups=content_dim),
            nn.GELU(),
            nn.Conv2d(content_dim, content_dim, 1),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.field(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), dim=1))


class CostOnlyField(nn.Module):
    """Capacity-matched non-visual control built only from the aggregated cost state."""

    def __init__(self, hidden_dim: int, content_dim: int) -> None:
        super().__init__()
        bridge_dim = 1280
        self.field = nn.Sequential(
            nn.Conv2d(hidden_dim, bridge_dim, 1),
            nn.GroupNorm(_group_count(bridge_dim), bridge_dim),
            nn.GELU(),
            nn.Conv2d(bridge_dim, content_dim, 1),
            nn.GroupNorm(_group_count(content_dim), content_dim),
            nn.GELU(),
            nn.Conv2d(content_dim, content_dim, 3, padding=1, groups=content_dim),
            nn.GELU(),
            nn.Conv2d(content_dim, content_dim, 1),
        )

    def forward(self, cost: Tensor) -> Tensor:
        return self.field(cost.mean(dim=2))


class VisualCostReconstruction(nn.Module):
    """A query-conditioned local visual-value read that produces a cost correction."""

    def __init__(self, hidden_dim: int, text_dim: int, config: CafeVCConfig, *, spatial_read: bool) -> None:
        super().__init__()
        self.content_dim = config.content_dim
        self.heads = config.heads
        self.kernel_size = config.kernel_size
        self.spatial_read = spatial_read
        dim = config.content_dim
        self.text_query = nn.Linear(text_dim, dim, bias=False)
        if spatial_read:
            self.state_query = nn.Linear(hidden_dim, dim)
            self.pixel_query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
        else:
            # Match active projection capacity without unused query parameters.
            query_parameters = (hidden_dim + 1) * dim + (dim + 1) * dim
            inner_dim = dim + round(query_parameters / (2 * dim + 1))
            self.local_transform = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(), nn.Linear(inner_dim, dim))
        self.evidence = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3 * dim),
            nn.Linear(hidden_dim + 3 * dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.zeros_(self.evidence[-1].weight)
        nn.init.zeros_(self.evidence[-1].bias)

    def _local_values(self, content: Tensor) -> tuple[Tensor, Tensor]:
        batch, dim, height, width = content.shape
        neighborhoods = F.unfold(content, self.kernel_size, padding=self.kernel_size // 2)
        # Unfold flattens channel before neighbor, not neighbor before channel.
        neighborhoods = neighborhoods.reshape(batch, dim, self.kernel_size**2, height * width).permute(0, 3, 2, 1)
        local = content.flatten(2).transpose(1, 2)
        return local, neighborhoods

    def forward(self, cost: Tensor, content: Tensor, text: Tensor) -> Tensor:
        batch, hidden, classes, height, width = cost.shape
        count = height * width
        local, neighborhoods = self._local_values(content)
        state = cost.permute(0, 2, 3, 4, 1).reshape(batch, classes, count, hidden)
        text_state = self.text_query(text)[None, :, None, :]
        if self.spatial_read:
            query = self.state_query(state) + self.pixel_query(local)[:, None] + text_state
            dim_per_head = self.content_dim // self.heads
            q = query.reshape(batch, classes, count, self.heads, dim_per_head)
            keys = self.key(neighborhoods).reshape(batch, count, self.kernel_size**2, self.heads, dim_per_head)
            values = self.value(neighborhoods).reshape(batch, count, self.kernel_size**2, self.heads, dim_per_head)
            weights = torch.einsum("bcnhd,bnkhd->bcnhk", q, keys) * (dim_per_head ** -0.5)
            weights = weights.softmax(dim=-1)
            retrieved = torch.einsum("bcnhk,bnkhd->bcnhd", weights, values).flatten(3)
        else:
            retrieved = self.local_transform(neighborhoods.mean(dim=2))[:, None].expand(-1, classes, -1, -1)
        local_by_class = local[:, None].expand(-1, classes, -1, -1)
        text_by_class = text_state.expand(batch, -1, count, -1)
        correction = self.evidence(torch.cat((state, local_by_class, retrieved, text_by_class), dim=-1))
        correction = correction.reshape(batch, classes, height, width, hidden).permute(0, 4, 1, 2, 3)
        return cost + correction


class CafeVC(nn.Module):
    """CAFe with frozen DINO/text/AnyUp/reduction and a trainable cost decoder.

    The visual-value blocks are intentionally placed after the original CAFe
    aggregation and before AnyUp. The final correction projections are zero
    initialized, so every arm begins as the same published CAFe function.
    """

    def __init__(self, cafe: nn.Module, config: CafeVCConfig) -> None:
        super().__init__()
        config.validate()
        self.cafe = cafe
        self.config = config
        for parameter in self.cafe.parameters():
            parameter.requires_grad_(False)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "aggregator"):
            getattr(self.cafe, name).requires_grad_(True)
        hidden_dim = cafe.aggregator_dim
        self.visual_field = VisualContentField(config.content_dim) if config.arm in {"concat", "vc"} else None
        self.cost_field = CostOnlyField(hidden_dim, config.content_dim) if config.arm == "cost_only" else None
        self.reconstruction = nn.ModuleList(
            [
                VisualCostReconstruction(
                    hidden_dim,
                    1024,
                    config,
                    spatial_read=config.arm in {"vc", "cost_only"},
                )
                for _ in range(config.blocks)
            ]
        )
        self.train(False)

    def train(self, mode: bool = True) -> "CafeVC":
        super().train(mode)
        self.cafe.backbone.eval()
        self.cafe.upsampler.eval()
        self.cafe.reduce_d.eval()
        return self

    def _set_aggregator_resolution(self, height: int, width: int, device: torch.device) -> None:
        for layer in self.cafe.aggregator:
            for block in (layer.spatial_agg.swin1, layer.spatial_agg.swin2):
                if tuple(block.input_resolution) != (height, width):
                    block.set_input_size((height, width), (7, 7))
                    if block.attn_mask is not None:
                        block.attn_mask = block.attn_mask.to(device)

    def _initial_state(self, x: Tensor, text: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, height, width = x.shape
        classes = len(text)
        raw = torch.einsum("bdhw,cd->bchw", F.normalize(x, dim=1), F.normalize(text, dim=-1))
        cost = self.cafe.corr_embed(raw.reshape(batch * classes, 1, height, width))
        cost = cost.reshape(batch, classes, -1, height, width).transpose(1, 2)
        text_guidance = self.cafe.text_guidance_projection(text)
        text_guidance = text_guidance[None, None, None].expand(batch, height, width, -1, -1).reshape(batch * height * width, classes, -1)
        visual_guidance = self.cafe.vis_guidance_projection(x).permute(0, 2, 3, 1)
        visual_guidance = visual_guidance[:, None].expand(-1, classes, -1, -1, -1).reshape(batch * classes, height, width, -1)
        return cost, text_guidance, visual_guidance

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or min(images.shape[-2:]) < 16 or any(size % 16 for size in images.shape[-2:]):
            raise ValueError("Images must be BCHW with dimensions divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty [queries, 1024] text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        batch, _, image_height, image_width = images.shape
        height, width = image_height // 16, image_width // 16
        self._set_aggregator_resolution(height, width, images.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "joint_finetuning", False)):
            _, y_tokens, x_tokens = self.cafe.backbone.encode_image_with_patch_tokens(images, normalize=False)
        if x_tokens.shape[1:] != (height * width, 1024) or y_tokens.shape != x_tokens.shape:
            raise RuntimeError("Unexpected frozen DINO X/Y token contract")
        x = x_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        y = y_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        cost, text_guidance, visual_guidance = self._initial_state(x, text)
        for aggregator in self.cafe.aggregator:
            cost = aggregator(cost, text_guidance, visual_guidance)
        if self.visual_field is not None:
            content = self.visual_field(x, y)
            for block in self.reconstruction:
                cost = block(cost, content, text)
        elif self.cost_field is not None:
            for block in self.reconstruction:
                cost = block(cost, self.cost_field(cost), text)
        selected = cost[:, :, :count].contiguous().reshape(batch, -1, height, width)
        # AnyUp must receive the original multi-channel cost representation.
        upsampled = self.cafe.upsampler(images, selected)
        class_features = upsampled.reshape(batch, self.cafe.aggregator_dim, count, image_height, image_width)
        class_features = class_features.transpose(1, 2).reshape(batch * count, self.cafe.aggregator_dim, image_height, image_width)
        logits = self.cafe.reduce_d(class_features).reshape(batch, count, image_height, image_width)
        return {"logits": logits}

    def adapted_state_dict(self) -> dict[str, Tensor]:
        excluded = ("cafe.backbone.", "cafe.upsampler.", "cafe.reduce_d.")
        return {name: tensor.detach().cpu() for name, tensor in self.state_dict().items() if not name.startswith(excluded)}

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = set(self.adapted_state_dict())
        if expected != set(state):
            raise ValueError(f"Adapted state mismatch: missing={expected - set(state)}, unexpected={set(state) - expected}")
        result = self.load_state_dict(state, strict=False)
        allowed = ("cafe.backbone.", "cafe.upsampler.", "cafe.reduce_d.")
        if result.unexpected_keys or any(not name.startswith(allowed) for name in result.missing_keys):
            raise RuntimeError("Incomplete CAFe-VC adapted checkpoint")

    def architecture(self) -> dict[str, Any]:
        return {
            "name": "CAFe-VC-v2",
            "neighbor_layout": "channel_then_neighbor_unfold_to_BNKD",
            "concat_read": "fixed_neighborhood_mean_then_mlp",
            **asdict(self.config),
            "upsampling": "original AnyUp cost features then frozen reduction",
            "aggregator_blocks": len(self.cafe.aggregator),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen_parameters": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        }
