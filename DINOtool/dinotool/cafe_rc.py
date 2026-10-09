"""CAFe-RC: regional visual-content reconstruction on the official CAFe model.

No class-specific parameters or ground-truth regions enter the forward pass.
The published encoder/AnyUp weights are supplied by the existing strict loader.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CafeRCConfig:
    content_dim: int = 128
    heads: int = 4
    region_stride: int = 4
    region_radius: int = 7
    variant: str = "regional"
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.content_dim < 8 or self.heads < 1 or self.content_dim % self.heads or self.content_dim % 8:
            raise ValueError("content_dim must be positive and divisible by heads")
        if self.region_stride < 1 or self.region_radius < self.region_stride:
            raise ValueError("Region radius must cover the region lattice")
        if self.variant not in {"regional", "content", "baseline", "mask_plain", "mask_dual", "mask_fixed"}:
            raise ValueError("Unknown CAFe experiment variant")


def _heads(values: Tensor, count: int) -> Tensor:
    return values.reshape(values.shape[0], -1, count, values.shape[-1] // count).transpose(1, 2)


def _unheads(values: Tensor) -> Tensor:
    return values.transpose(1, 2).flatten(2)


class RegionalMemory(nn.Module):
    """Locally overlapping, learned soft regions initialized on a regular grid."""

    def __init__(self, config: CafeRCConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.content_dim
        self.norm = nn.LayerNorm(dim)
        self.region_q = nn.Linear(dim, dim)
        self.pixel_k = nn.Linear(dim, dim)
        self.pixel_v = nn.Linear(dim, dim)
        self.pixel_q = nn.Linear(dim, dim)
        self.region_k = nn.Linear(dim, dim)
        self.region_v = nn.Linear(dim, dim)
        self.update = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def support(self, height: int, width: int, device: torch.device) -> Tensor:
        stride, radius = self.config.region_stride, self.config.region_radius
        py, px = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        ry, rx = torch.meshgrid(torch.arange(0, height, stride, device=device), torch.arange(0, width, stride, device=device), indexing="ij")
        return ((py.flatten()[None] - ry.flatten()[:, None]).abs() <= radius) & ((px.flatten()[None] - rx.flatten()[:, None]).abs() <= radius)

    def forward(self, content: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, dim, height, width = content.shape
        pixels = self.norm(content.flatten(2).transpose(1, 2))
        stride = self.config.region_stride
        anchors = F.avg_pool2d(content, 2 * stride - 1, stride=stride, padding=stride - 1, count_include_pad=False)
        anchors = self.norm(anchors.flatten(2).transpose(1, 2))
        support = self.support(height, width, content.device)
        heads = self.config.heads
        memory = _unheads(F.scaled_dot_product_attention(
            _heads(self.region_q(anchors), heads), _heads(self.pixel_k(pixels), heads),
            _heads(self.pixel_v(pixels), heads), attn_mask=support[None, None],
        ))
        memory = anchors + memory
        recovered = _unheads(F.scaled_dot_product_attention(
            _heads(self.pixel_q(pixels), heads), _heads(self.region_k(memory), heads),
            _heads(self.region_v(memory), heads), attn_mask=support.T[None, None],
        ))
        updated = pixels + self.update(recovered)
        return updated.transpose(1, 2).reshape(batch, dim, height, width), memory, support


class ContentReconstruction(nn.Module):
    """Read full visual values with spatial/query conditions before CAFe refinement."""

    def __init__(self, hidden_dim: int, text_dim: int, config: CafeRCConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.content_dim
        self.local_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(), nn.Conv2d(dim, dim, 1),
        )
        self.regions = RegionalMemory(config) if config.variant == "regional" else None
        self.text_content = nn.Linear(text_dim, dim)
        if self.regions is not None:
            self.cost_q = nn.Linear(hidden_dim, dim)
            self.pixel_q = nn.Linear(dim, dim)
            self.memory_k = nn.Linear(dim, dim)
            self.memory_v = nn.Linear(dim, dim)
        self.evidence = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3 * dim), nn.Linear(hidden_dim + 3 * dim, hidden_dim * 2),
            nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Keep the pretrained cost path dominant initially without a learned gate.
        nn.init.normal_(self.evidence[-1].weight, std=1e-3)
        nn.init.zeros_(self.evidence[-1].bias)

    def forward(self, content: Tensor, cost: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        batch, hidden, classes, height, width = cost.shape
        count = height * width
        content = content + self.local_mix(content)
        if self.regions is not None:
            content, memory, support = self.regions(content)
        pixels = content.flatten(2).transpose(1, 2)
        state = cost.permute(0, 2, 3, 4, 1).reshape(batch, classes, count, hidden)
        text_content = self.text_content(text)[None, :, None, :]
        local = pixels[:, None].expand(-1, classes, -1, -1)
        if self.regions is not None:
            query = self.cost_q(state) + self.pixel_q(local) + text_content
            mask = support.T[None].expand(classes, -1, -1).reshape(classes * count, -1)
            retrieved = _unheads(F.scaled_dot_product_attention(
                _heads(query.reshape(batch, classes * count, -1), self.config.heads),
                _heads(self.memory_k(memory), self.config.heads),
                _heads(self.memory_v(memory), self.config.heads), attn_mask=mask[None, None],
            )).reshape(batch, classes, count, -1)
        else:
            retrieved = local
        update = self.evidence(torch.cat((state, local * text_content, retrieved * text_content, retrieved), dim=-1))
        update = update.reshape(batch, classes, height, width, hidden).permute(0, 4, 1, 2, 3)
        return content, cost + update


class CafeRC(nn.Module):
    """Frozen DINO/text/AnyUp plus trainable CAFe and content reconstruction.

    Images are ImageNet-normalized. Text rows are normalized official patch-text
    embeddings; the number and order of query classes are runtime inputs.
    """

    def __init__(self, cafe: nn.Module, config: CafeRCConfig, *, teacher: bool = True) -> None:
        super().__init__()
        config.validate()
        self.cafe, self.config = cafe, config
        self.teacher = nn.ModuleDict()
        for parameter in self.cafe.parameters():
            parameter.requires_grad_(False)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "aggregator"):
            module = getattr(cafe, name)
            if teacher:
                self.teacher[name] = copy.deepcopy(module).requires_grad_(False).eval()
            module.requires_grad_(True)
        if config.variant in {"regional", "content"}:
            hidden = cafe.aggregator_dim
            self.content_input = nn.Sequential(nn.Conv2d(2048, config.content_dim, 1), nn.GroupNorm(8, config.content_dim), nn.GELU())
            self.reconstruction = nn.ModuleList([ContentReconstruction(hidden, 1024, config) for _ in cafe.aggregator])
        else:
            self.content_input = None
            self.reconstruction = nn.ModuleList()
        self.mask_decoder = None
        if config.variant.startswith("mask_"):
            from .cafe_membership import FlatMembershipDecoder
            self.mask_decoder = FlatMembershipDecoder(config.content_dim, config.heads, config.region_stride, config.region_radius, variant=config.variant)
        self.train(False)

    def train(self, mode: bool = True) -> CafeRC:
        super().train(mode)
        self.cafe.backbone.eval()
        self.cafe.upsampler.eval()
        self.cafe.reduce_d.eval()
        self.teacher.eval()
        return self

    def _initial_state(self, x: Tensor, text: Tensor, modules: Any) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, height, width = x.shape
        classes = len(text)
        raw = torch.einsum("bdhw,cd->bchw", F.normalize(x, dim=1), F.normalize(text, dim=-1))
        cost = modules["corr_embed"](raw.reshape(batch * classes, 1, height, width))
        cost = cost.reshape(batch, classes, -1, height, width).transpose(1, 2)
        t = modules["text_guidance_projection"](text)
        t = t[None, None, None].expand(batch, height, width, -1, -1).reshape(batch * height * width, classes, -1)
        visual = modules["vis_guidance_projection"](x).permute(0, 2, 3, 1)
        visual = visual[:, None].expand(-1, classes, -1, -1, -1).reshape(batch * classes, height, width, -1)
        return cost, t, visual

    def _coarse_logits(self, cost: Tensor) -> Tensor:
        batch, hidden, classes, height, width = cost.shape
        return self.cafe.reduce_d(cost.transpose(1, 2).reshape(batch * classes, hidden, height, width)).reshape(batch, classes, height, width)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None, preserve_from: int | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or min(images.shape[-2:]) < 16 or any(size % 16 for size in images.shape[-2:]):
            raise ValueError("Images must be BCHW with spatial dimensions divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or len(text) < 1:
            raise ValueError("Expected a nonempty [queries,1024] patch-text tensor")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        batch, _, ih, iw = images.shape
        height, width = ih // 16, iw // 16
        if min(height, width) < 7:
            raise ValueError("CAFe-RC needs at least a 7x7 patch grid; pad inputs to >=112 pixels")
        stacks = [self.cafe.aggregator]
        if self.teacher:
            stacks.append(self.teacher["aggregator"])
        for stack in stacks:
            for layer in stack:
                for block in (layer.spatial_agg.swin1, layer.spatial_agg.swin2):
                    if tuple(block.input_resolution) != (height, width):
                        block.set_input_size((height, width), (7, 7))
                        if block.attn_mask is not None:
                            block.attn_mask = block.attn_mask.to(images.device)
        with torch.no_grad():
            _, y_tokens, x_tokens = self.cafe.backbone.encode_image_with_patch_tokens(images, normalize=False)
        if x_tokens.shape[1:] != (height * width, 1024) or y_tokens.shape != x_tokens.shape:
            raise RuntimeError("Unexpected DINO X/Y contract")
        x = x_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        y = y_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        modules = {name: getattr(self.cafe, name) for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection")}
        cost, text_guidance, visual_guidance = self._initial_state(x, text, modules)
        content = self.content_input(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), dim=1)) if self.content_input is not None else None
        for index, aggregator in enumerate(self.cafe.aggregator):
            if content is not None:
                block = self.reconstruction[index]
                if self.training and self.config.gradient_checkpointing:
                    content, cost = checkpoint(block, content, cost, text, use_reentrant=False)
                else:
                    content, cost = block(content, cost, text)
            if self.training and self.config.gradient_checkpointing:
                cost = checkpoint(aggregator, cost, text_guidance, visual_guidance, use_reentrant=False)
            else:
                cost = aggregator(cost, text_guidance, visual_guidance)
        mask_result = None
        if self.mask_decoder is not None:
            coarse = self._coarse_logits(cost)
            if self.training and self.config.gradient_checkpointing:
                mask_result = checkpoint(self.mask_decoder, x, y, coarse, text, use_reentrant=False)
            else:
                mask_result = self.mask_decoder(x, y, coarse, text)
        preservation = cost.sum() * 0.0
        if preserve_from is not None:
            if not self.teacher or not 0 <= preserve_from < len(text) - 1:
                raise ValueError("Preservation needs a frozen teacher and at least two extra queries")
            with torch.no_grad():
                prior, tg, vg = self._initial_state(x, text, self.teacher)
                for block in self.teacher["aggregator"]:
                    prior = block(prior, tg, vg)
                teacher_logits = self._coarse_logits(prior)[:, preserve_from:].float()
            student_coarse = self._coarse_logits(cost) if mask_result is None else mask_result["coarse_logits"]
            student_logits = student_coarse[:, preserve_from:].float()
            temperature = 2.0
            preservation = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1), reduction="none").sum(1).mean() * temperature**2
        if mask_result is not None:
            selected = mask_result["coarse_logits"][:, :count].contiguous()
            if self.training and self.config.gradient_checkpointing:
                logits = checkpoint(self.cafe.upsampler, images, selected, use_reentrant=False)
            else:
                logits = self.cafe.upsampler(images, selected)
            return {"logits": logits, "preservation_loss": preservation, "mean_member_change": mask_result["mean_member_change"]}
        selected = cost[:, :, :count].contiguous().reshape(batch, -1, height, width)
        # AnyUp weights are frozen, but gradients must pass through its feature input.
        if self.training and self.config.gradient_checkpointing:
            upsampled = checkpoint(self.cafe.upsampler, images, selected, use_reentrant=False)
        else:
            upsampled = self.cafe.upsampler(images, selected)
        class_features = upsampled.reshape(batch, self.cafe.aggregator_dim, count, ih, iw).transpose(1, 2).reshape(batch * count, self.cafe.aggregator_dim, ih, iw)
        logits = self.cafe.reduce_d(class_features).reshape(batch, count, ih, iw)
        return {"logits": logits, "preservation_loss": preservation}

    def adapted_state_dict(self) -> dict[str, Tensor]:
        excluded = ("cafe.backbone.", "cafe.upsampler.", "teacher.")
        return {name: tensor.detach().cpu() for name, tensor in self.state_dict().items() if not name.startswith(excluded)}

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = set(self.adapted_state_dict())
        if expected != set(state):
            raise ValueError(f"Adapted state mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not name.startswith(("cafe.backbone.", "cafe.upsampler.", "teacher.")) for name in result.missing_keys):
            raise RuntimeError("Incomplete CAFe-RC adapted checkpoint")

    def architecture(self) -> dict[str, Any]:
        return {"name": "CAFe-flat-mask-v1" if self.mask_decoder is not None else "CAFe-RC-v1", **asdict(self.config),
                "upsampling": "AnyUp scalar logits" if self.mask_decoder is not None else "AnyUp cost features then frozen reduction",
                "aggregator_blocks": len(self.cafe.aggregator),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad)}
