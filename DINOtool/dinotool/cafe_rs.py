"""CAFe-RS: multiscale visual workspace with recurrent pixel-region reasoning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cafe_vc import CafeVC, CafeVCConfig
from .cafe_rs_regions import RegionReasoningStage, make_layout, region_auxiliary_losses


@dataclass(frozen=True)
class CafeRSConfig:
    arm: str = "rs"
    content_dim: int = 256
    heads: int = 4
    stages: int = 3
    modes: int = 4

    def validate(self):
        if self.arm != "rs" or min(self.heads, self.stages, self.modes) < 1:
            raise ValueError("Invalid CAFe-RS configuration")
        if self.content_dim < 16 or self.content_dim % self.heads or self.content_dim % 8:
            raise ValueError("RS width must be divisible by heads and eight")


@lru_cache(maxsize=4)
def load_msda(official_root: str):
    # Reuse the pinned Meta implementation under its existing DINOv3 license.
    path = Path(official_root) / "dinov3/eval/segmentation/models/utils/ms_deform_attn.py"
    spec = importlib.util.spec_from_file_location("cafe_rs_official_msda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutogradDeformableBlock(nn.Module):
    """Official sampling/projection parameters, with its pure-PyTorch core for autograd.

    The official custom Function needs a compiled extension for backward. Calling
    the reference core directly retains ordinary grid_sample gradients instead.
    """
    def __init__(self, dim: int, heads: int, official_root: str):
        super().__init__()
        module = load_msda(official_root)
        self.attention = module.MSDeformAttn(dim, n_levels=5, n_heads=heads, n_points=4)
        self.core = module.ms_deform_attn_core_pytorch
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query: Tensor, levels: list[Tensor], reference: Tensor) -> Tensor:
        a = self.attention
        b, n, d = query.shape
        shapes = [tuple(level.shape[-2:]) for level in levels]
        values = torch.cat([level.flatten(2).transpose(1, 2) for level in levels], 1)
        value = a.value_proj(values).reshape(b, -1, a.n_heads, d // a.n_heads)
        normalized = self.norm1(query)
        offsets = a.sampling_offsets(normalized).reshape(b, n, a.n_heads, a.n_levels, a.n_points, 2)
        weights = a.attention_weights(normalized).reshape(b, n, a.n_heads, -1).softmax(-1)
        weights = weights.reshape(b, n, a.n_heads, a.n_levels, a.n_points)
        normalizer = query.new_tensor([(w, h) for h, w in shapes])
        locations = reference[None, :, None, None, None] + offsets / normalizer[None, None, None, :, None]
        with torch.autocast(query.device.type, enabled=False):
            sampled = self.core(value.float(), shapes, locations.float(), weights.float())
        query = query + a.output_proj(sampled.to(query.dtype))
        return query + self.ffn(self.norm2(query))


class VisualWorkspace(nn.Module):
    def __init__(self, dim: int, heads: int, official_root: str):
        super().__init__()
        def conv(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.GELU())
        self.detail4 = nn.Sequential(conv(3, dim // 2, 2), conv(dim // 2, dim, 2))
        self.detail8 = conv(dim, dim, 2)
        self.patch = nn.Sequential(nn.Conv2d(2048, dim, 1), nn.GroupNorm(8, dim), nn.GELU())
        self.position = nn.Linear(2, dim)
        self.level_embeddings = nn.Parameter(torch.empty(5, dim))
        nn.init.normal_(self.level_embeddings, std=0.02)
        self.blocks = nn.ModuleList([AutogradDeformableBlock(dim, heads, official_root) for _ in range(2)])

    def forward(self, images: Tensor, x: Tensor, y: Tensor, positions: Tensor):
        detail4 = self.detail4(images)
        detail8 = self.detail8(detail4)
        patch = self.patch(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), 1))
        h, w = patch.shape[-2:]
        levels = [detail4, detail8, patch,
                  F.adaptive_avg_pool2d(patch, (max(1, (h + 1) // 2), max(1, (w + 1) // 2))),
                  F.adaptive_avg_pool2d(patch, (max(1, (h + 3) // 4), max(1, (w + 3) // 4)))]
        levels = [level + self.level_embeddings[i][None, :, None, None] for i, level in enumerate(levels)]
        query = patch.flatten(2).transpose(1, 2) + self.position(positions.to(patch.dtype))[None]
        for block in self.blocks:
            query = block(query, levels, positions)
            levels[2] = query.transpose(1, 2).reshape(len(x), -1, h, w)
        return query


class CafeRS(CafeVC):
    """Stage-A model: all published parameters frozen, new decoder trainable."""
    def __init__(self, cafe: nn.Module, config: CafeRSConfig, *, official_root: str):
        config.validate()
        super().__init__(cafe, CafeVCConfig(arm="plain", blocks=0))
        self.config = config
        self.cafe.requires_grad_(False)
        if len(cafe.aggregator) % config.stages or len(cafe.aggregator) < config.stages:
            raise ValueError("All original CAFe blocks must be evenly assigned to reasoning stages")
        self.workspace = VisualWorkspace(config.content_dim, config.heads, official_root)
        self.reasoning = nn.ModuleList([
            RegionReasoningStage(config.content_dim, cafe.aggregator_dim, config.heads, config.modes, update_state=i < config.stages - 1)
            for i in range(config.stages)
        ])
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cafe.eval()
        return self

    def _output(self, images: Tensor, cost: Tensor, count: int) -> Tensor:
        b, d, _, h, w = cost.shape
        ih, iw = images.shape[-2:]
        up = self.cafe.upsampler(images, cost[:, :, :count].contiguous().reshape(b, d * count, h, w))
        features = up.reshape(b, d, count, ih, iw).transpose(1, 2).reshape(b * count, d, ih, iw)
        return self.cafe.reduce_d(features).reshape(b, count, ih, iw)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None, return_aux: bool = False, teacher: bool = False):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 112 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("RS expects RGB BCHW inputs divisible by 16 and at least 112")
        if text.ndim != 2 or text.shape[1] != 1024 or len(text) < 1:
            raise ValueError("RS expects nonempty Q x 1024 text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text) or ((return_aux or teacher) and count != len(text)):
            raise ValueError("Training auxiliaries require the entire query set")
        b, _, ih, iw = images.shape
        h, w = ih // 16, iw // 16
        self._set_aggregator_resolution(h, w, images.device)
        if teacher and getattr(self, "joint_finetuning", False):
            raise ValueError("Joint fine-tuning requires an independent frozen teacher")
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "joint_finetuning", False)):
            _, yt, xt = self.cafe.backbone.encode_image_with_patch_tokens(images, normalize=False)
            if xt.shape[1:] != (h * w, 1024) or yt.shape != xt.shape:
                raise RuntimeError("Unexpected frozen X/Y feature contract")
            x, y = (v.transpose(1, 2).reshape(b, 1024, h, w) for v in (xt, yt))
            cost, tg, vg = self._initial_state(x, text)
            teacher_logits = None
            if teacher:
                teacher_cost = cost
                for layer in self.cafe.aggregator:
                    teacher_cost = layer(teacher_cost, tg, vg)
                teacher_logits = self._output(images, teacher_cost, count).detach()
        layout = make_layout(h, w, images.device)
        visual = self.workspace(images, x, y, layout.pixels)
        visual_map = visual.transpose(1, 2).reshape(b, -1, h, w)
        nodes = torch.cat([F.adaptive_avg_pool2d(visual_map, shape).flatten(2).transpose(1, 2) for shape in layout.shapes], 1)
        centers = layout.centers[None].expand(b, -1, -1)
        auxiliary = []
        group = len(self.cafe.aggregator) // self.config.stages
        for i, stage in enumerate(self.reasoning):
            for layer in self.cafe.aggregator[i * group:(i + 1) * group]:
                cost = layer(cost, tg, vg)
            state = cost.permute(0, 3, 4, 2, 1).reshape(b, h * w, len(text), -1)
            state, visual, nodes, centers, extra = stage(state, visual, nodes, centers, text, layout)
            cost = state.reshape(b, h, w, len(text), -1).permute(0, 4, 3, 1, 2).contiguous()
            if return_aux:
                auxiliary.append(extra)
        result = {"logits": self._output(images, cost, count)}
        if return_aux:
            result.update(stages=auxiliary, layout=layout, patch_shape=(h, w))
        if teacher:
            result["teacher_logits"] = teacher_logits
        return result

    def architecture(self):
        return {"name": "CAFe-RS-v1", **asdict(self.config), "training_stage": "A_frozen_published",
                "region_strides": [2, 4], "deformable_backend": "official_pytorch_autograd_fp32",
                "upsampling": "original AnyUp cost features then frozen reduction",
                "aggregator_blocks": len(self.cafe.aggregator),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad)}


def rs_losses(output: dict, target: Tensor, smoothing: float, region_weight: float, affinity_weight: float, kd_weight: float):
    valid = target != 255
    logits = output["logits"].float()
    ce = F.cross_entropy(logits, target, ignore_index=255, label_smoothing=smoothing) if valid.any() else logits.sum() * 0
    auxiliary = region_auxiliary_losses(output["stages"], target, logits.shape[1], output["patch_shape"], output["layout"])
    kd = logits.sum() * 0
    if kd_weight > 0:
        reference = output["teacher_logits"].float().detach()
        correct = valid & (reference.argmax(1) == target)
        divergence = F.kl_div((logits / 2).log_softmax(1), (reference / 2).softmax(1), reduction="none").sum(1) * 4
        kd = (divergence * correct).sum() / correct.sum().clamp_min(1)
    loss = ce + region_weight * auxiliary["region_loss"] + affinity_weight * auxiliary["affinity_loss"] + kd_weight * kd
    return loss, {"ce_loss": ce, **auxiliary, "kd_loss": kd}
