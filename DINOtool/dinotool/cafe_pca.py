"""PCA-DINO: PCA-Seg parallel cost aggregation on the CAFe-DINO decoder."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cafe_ped import ped_loss


@dataclass(frozen=True)
class PCADINOConfig:
    """Topology settings for matched CAFe and PCA-DINO experiments."""

    arm: str = "pca_epl_fod"
    experts: int = 4
    reduction_ratio: int = 4
    stages: int = 6
    corrected_class_attention: bool = False
    channel_init: str = "pretrained"
    visual_tune_blocks: int = 2

    def validate(self) -> None:
        if self.arm not in {"serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"}:
            raise ValueError("Unknown PCA-DINO arm")
        if self.experts < 2 or self.reduction_ratio < 1 or self.stages < 1:
            raise ValueError("PCA-DINO needs at least two experts and one stage")
        if self.channel_init not in {"pretrained", "fresh"}:
            raise ValueError("Unknown channel initialization")
        if self.arm == "serial" and self.channel_init != "pretrained":
            raise ValueError("Serial CAFe control must preserve the published initialization")
        if self.visual_tune_blocks not in {0, 2}:
            raise ValueError("Only zero or two visual blocks may be tuned")


class QueryAxisChannelAggregator(nn.Module):
    """CAFe channel aggregation with SDPA explicitly attending over queries.

    The published full-attention module passes ``[BHW, Q, heads, d]`` to
    PyTorch SDPA, so it attends over heads. This opt-in wrapper copies every
    learned projection and changes only the tensor layout. The correction is
    recorded as a separate control rather than attributed to PCA-DINO.
    """

    def __init__(self, published: nn.Module) -> None:
        super().__init__()
        self.heads = int(published.heads)
        self.use_linear_transformer = bool(published.use_linear_transformer)
        names = ("q", "k", "v", "norm", "ffn", "guidance_norm", "drop_path1", "drop_path2")
        for name in names:
            setattr(self, name, copy.deepcopy(getattr(published, name)))

    @staticmethod
    def _linear_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query, key = F.elu(query) + 1.0, F.elu(key) + 1.0
        key_value = torch.einsum("bshd,bshe->bhde", key, value)
        normalizer = 1.0 / (torch.einsum("blhd,bhd->blh", query, key.sum(1)) + 1e-6)
        return torch.einsum("blhd,bhde->blhe", query, key_value) * normalizer.unsqueeze(-1)

    def forward(self, cost: Tensor, guidance: Tensor) -> Tensor:
        batch, dim, queries, height, width = cost.shape
        state = cost.permute(0, 3, 4, 2, 1).reshape(batch * height * width, queries, dim)
        guide = self.guidance_norm(guidance) if self.guidance_norm is not None else guidance
        query = self.q(torch.cat((state, guide), dim=-1))
        key = self.k(torch.cat((state, guide), dim=-1))
        value = self.v(state)
        head_dim = dim // self.heads
        query = query.reshape(-1, queries, self.heads, head_dim)
        key = key.reshape(-1, queries, self.heads, head_dim)
        value = value.reshape(-1, queries, self.heads, head_dim)
        if self.use_linear_transformer:
            attended = self._linear_attention(query, key, value)
        else:
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
        attended = attended.reshape(-1, queries, dim)
        state = self.norm(state + self.drop_path1(attended))
        state = state + self.drop_path2(self.ffn(state))
        return state.reshape(batch, height, width, queries, dim).permute(0, 4, 3, 1, 2).contiguous()


class ExpertDrivenPerceptualLearning(nn.Module):
    """Official PCA-Seg EPL topology adapted to ``[B, D, Q, H, W]`` costs."""

    def __init__(self, dim: int, experts: int = 4, reduction_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(1, dim // reduction_ratio)
        self.experts = experts
        self.spatial_projection = nn.Conv2d(dim, dim, 1)
        self.class_projection = nn.Conv2d(dim, dim, 1)
        self.parsers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, 1),
                nn.SyncBatchNorm(2 * dim),
                nn.GELU(),
                nn.Conv2d(2 * dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
            )
            for _ in range(experts)
        ])
        self.coefficient_mapper = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, experts, 1),
        )
        self.residual = nn.Conv2d(2 * dim, dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, spatial: Tensor, semantic: Tensor, *, return_coefficients: bool = False):
        if spatial.shape != semantic.shape or spatial.ndim != 5:
            raise ValueError("EPL expects matching [B, D, Q, H, W] branch tensors")
        batch, dim, queries, height, width = spatial.shape
        spatial_2d = spatial.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, height, width)
        semantic_2d = semantic.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, height, width)
        joined = torch.cat((self.spatial_projection(spatial_2d), self.class_projection(semantic_2d)), dim=1)
        coefficients = self.coefficient_mapper(joined).softmax(dim=1)
        fused = sum(parser(joined) * coefficients[:, index:index + 1]
                    for index, parser in enumerate(self.parsers))
        fused = fused + self.residual_scale * self.residual(joined)
        fused = fused.reshape(batch, queries, dim, height, width).permute(0, 2, 1, 3, 4).contiguous()
        if return_coefficients:
            return fused, coefficients.reshape(batch, queries, self.experts, height, width)
        return fused


class TaskGatedParallelFusion(nn.Module):
    """Choose the original, spatial, or class cost at each query and position.

    The anchor gives the model a way to reject two harmful branch changes. The
    same segmentation loss that trains both aggregators also trains this gate.
    Its weights are diagnostic, not a claim that either branch has a fixed role.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(4 * dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor((0.69314718056, 0.0, 0.0)))

    def forward(self, original: Tensor, spatial: Tensor, semantic: Tensor,
                *, return_weights: bool = False):
        if original.shape != spatial.shape or original.shape != semantic.shape or original.ndim != 5:
            raise ValueError("Task gate expects matching [B, D, Q, H, W] tensors")
        batch, dim, queries, height, width = original.shape
        def flatten(value: Tensor) -> Tensor:
            return value.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, height, width)
        base = flatten(original)
        spatial_delta = flatten(spatial - original)
        semantic_delta = flatten(semantic - original)
        features = torch.cat((base, spatial_delta, semantic_delta,
                              (spatial_delta - semantic_delta).abs()), dim=1)
        weights = self.gate(features).softmax(dim=1)
        fused = (weights[:, 0:1] * base + weights[:, 1:2] * flatten(spatial)
                 + weights[:, 2:3] * flatten(semantic))
        fused = fused.reshape(batch, queries, dim, height, width).permute(0, 2, 1, 3, 4).contiguous()
        if return_weights:
            return fused, weights.reshape(batch, queries, 3, height, width)
        return fused


def feature_orthogonalization_loss(spatial: Tensor, semantic: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Mean squared point-wise cosine used by PCA-Seg FOD."""
    if spatial.shape != semantic.shape or spatial.ndim != 5:
        raise ValueError("FOD expects matching [B, D, Q, H, W] branch tensors")
    spatial_vector = spatial.permute(0, 2, 3, 4, 1).reshape(-1, spatial.shape[1])
    semantic_vector = semantic.permute(0, 2, 3, 4, 1).reshape(-1, semantic.shape[1])
    cosine = (F.normalize(spatial_vector, dim=-1, eps=eps) *
              F.normalize(semantic_vector, dim=-1, eps=eps)).sum(-1)
    return cosine.square().mean(), cosine.mean()


class CafePCA(nn.Module):
    """Matched serial, plain-parallel, and PCA aggregation on CAFe-DINO."""

    def __init__(self, cafe: nn.Module, config: PCADINOConfig) -> None:
        super().__init__()
        config.validate()
        if len(cafe.aggregator) != config.stages:
            raise ValueError(f"Requested {config.stages} stages, official CAFe has {len(cafe.aggregator)}")
        self.cafe = cafe
        self.config = config
        self.cost_dim = int(cafe.aggregator_dim)
        self.warmup = False
        self.joint_finetuning = True
        self.spatial = nn.ModuleList()
        self.channel = nn.ModuleList()
        self.fusion = nn.ModuleList()
        if config.arm == "serial" and config.corrected_class_attention:
            for layer in cafe.aggregator:
                layer.class_agg = QueryAxisChannelAggregator(layer.class_agg)
        if config.arm != "serial":
            self.spatial = nn.ModuleList(copy.deepcopy(layer.spatial_agg) for layer in cafe.aggregator)
            published_channels = [copy.deepcopy(layer.class_agg) for layer in cafe.aggregator]
            self.channel = nn.ModuleList(
                QueryAxisChannelAggregator(module) if config.corrected_class_attention else module
                for module in published_channels
            )
            if config.channel_init == "fresh":
                for channel in self.channel:
                    for module in channel.modules():
                        if module is not channel and hasattr(module, "reset_parameters"):
                            module.reset_parameters()
            if config.arm.startswith("pca_epl"):
                self.fusion = nn.ModuleList(
                    ExpertDrivenPerceptualLearning(self.cost_dim, config.experts, config.reduction_ratio)
                    for _ in range(config.stages)
                )
            elif config.arm == "task_parallel":
                self.fusion = nn.ModuleList(
                    TaskGatedParallelFusion(self.cost_dim) for _ in range(config.stages)
                )
            cafe.aggregator = nn.ModuleList()
        self._configure_trainable_parameters()
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_task_parallel_v1" if self.config.arm == "task_parallel" else "cafe_pca_v1"

    def _configure_trainable_parameters(self) -> None:
        self.cafe.requires_grad_(False)
        if self.config.arm == "serial":
            self.cafe.aggregator.requires_grad_(True)
        else:
            self.spatial.requires_grad_(True)
            self.channel.requires_grad_(True)
            self.fusion.requires_grad_(True)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "reduce_d"):
            getattr(self.cafe, name).requires_grad_(True)
        blocks = self.cafe.backbone.visual_model.backbone.blocks
        if len(blocks) < 2:
            raise ValueError("CAFe-DINO backbone must expose at least two visual transformer blocks")
        self.tuned_indices = tuple(range(len(blocks) - self.config.visual_tune_blocks, len(blocks)))
        for index in self.tuned_indices:
            blocks[index].requires_grad_(True)
        self.joint_finetuning = bool(self.tuned_indices)

    def train(self, mode: bool = True) -> "CafePCA":
        nn.Module.train(self, mode)
        self.cafe.backbone.eval()
        self.cafe.upsampler.eval()
        self.cafe.reduce_d.eval()
        active = mode and not self.warmup
        if self.config.arm == "serial":
            self.cafe.aggregator.train(active)
        else:
            self.spatial.train(active)
            self.channel.train(active)
            # EPL is the new module optimized during decoder warmup; migrated
            # CAFe branches remain inactive until the joint phase begins.
            self.fusion.train(mode)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].train(active)
        for module in (self.cafe.corr_embed, self.cafe.text_guidance_projection, self.cafe.vis_guidance_projection):
            module.train(active)
        for module in self.cafe.reduce_d.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_warmup(self, warmup: bool) -> None:
        self.warmup = warmup
        self.train(self.training)

    def _set_aggregator_resolution(self, height: int, width: int, device: torch.device) -> None:
        modules = (layer.spatial_agg for layer in self.cafe.aggregator) if self.config.arm == "serial" else self.spatial
        for module in modules:
            for block in (module.swin1, module.swin2):
                if tuple(block.input_resolution) != (height, width):
                    block.set_input_size((height, width), (7, 7))
                    if block.attn_mask is not None:
                        block.attn_mask = block.attn_mask.to(device)

    def _encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, image_height, image_width = images.shape
        height, width = image_height // 16, image_width // 16
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.joint_finetuning):
            _, y_tokens, x_tokens = self.cafe.backbone.encode_image_with_patch_tokens(images, normalize=False)
        if x_tokens.shape[1:] != (height * width, 1024) or y_tokens.shape != x_tokens.shape:
            raise RuntimeError("Unexpected DINO X/Y token contract")
        x = x_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        y = y_tokens.transpose(1, 2).reshape(batch, 1024, height, width)
        return x, y

    def _initial_state(self, visual: Tensor, text: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, height, width = visual.shape
        queries = len(text)
        raw = torch.einsum("bdhw,qd->bqhw", F.normalize(visual, dim=1), F.normalize(text, dim=-1))
        cost = self.cafe.corr_embed(raw.reshape(batch * queries, 1, height, width))
        cost = cost.reshape(batch, queries, self.cost_dim, height, width).transpose(1, 2)
        text_guidance = self.cafe.text_guidance_projection(text)
        text_guidance = text_guidance[None, None, None].expand(batch, height, width, -1, -1)
        text_guidance = text_guidance.reshape(batch * height * width, queries, self.cost_dim)
        visual_guidance = self.cafe.vis_guidance_projection(visual).permute(0, 2, 3, 1)
        visual_guidance = visual_guidance[:, None].expand(-1, queries, -1, -1, -1)
        visual_guidance = visual_guidance.reshape(batch * queries, height, width, self.cost_dim)
        return cost, text_guidance, visual_guidance

    def _aggregate(self, cost: Tensor, text_guidance: Tensor, visual_guidance: Tensor,
                   *, return_aux: bool) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.arm == "serial":
            for layer in self.cafe.aggregator:
                cost = layer(cost, text_guidance, visual_guidance)
            zero = cost.sum() * 0
            return cost, {"fod_loss": zero, "branch_cosine": zero}
        fod_terms, cosine_terms, gate_terms = [], [], []
        for index, (spatial, channel) in enumerate(zip(self.spatial, self.channel)):
            spatial_cost = spatial(cost, visual_guidance)
            semantic_cost = channel(cost, text_guidance)
            if self.config.arm == "parallel":
                cost = cost + (spatial_cost - cost) + (semantic_cost - cost)
            elif self.config.arm == "task_parallel":
                if return_aux:
                    cost, weights = self.fusion[index](cost, spatial_cost, semantic_cost,
                                                       return_weights=True)
                    gate_terms.append(weights.mean(dim=(0, 1, 3, 4)))
                else:
                    cost = self.fusion[index](cost, spatial_cost, semantic_cost)
            else:
                cost = self.fusion[index](spatial_cost, semantic_cost)
            if return_aux:
                fod, cosine = feature_orthogonalization_loss(spatial_cost, semantic_cost)
                fod_terms.append(fod)
                cosine_terms.append(cosine)
        zero = cost.sum() * 0
        auxiliary = {
            "fod_loss": torch.stack(fod_terms).mean() if fod_terms else zero,
            "branch_cosine": torch.stack(cosine_terms).mean() if cosine_terms else zero,
        }
        if gate_terms:
            gate_mean = torch.stack(gate_terms).mean(0)
            auxiliary.update({"gate_anchor": gate_mean[0], "gate_spatial": gate_mean[1],
                              "gate_class": gate_mean[2]})
        return cost, auxiliary

    def _decode(self, images: Tensor, cost: Tensor, count: int) -> Tensor:
        batch, _, _, height, width = cost.shape
        image_height, image_width = images.shape[-2:]
        selected = cost[:, :, :count].contiguous().reshape(batch, self.cost_dim * count, height, width)
        upsampled = self.cafe.upsampler(images, selected)
        features = upsampled.reshape(batch, self.cost_dim, count, image_height, image_width)
        features = features.transpose(1, 2).reshape(batch * count, self.cost_dim, image_height, image_width)
        return self.cafe.reduce_d(features).reshape(batch, count, image_height, image_width)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                return_aux: bool = False) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty Q x 1024 DINO.text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        height, width = images.shape[-2] // 16, images.shape[-1] // 16
        self._set_aggregator_resolution(height, width, images.device)
        visual, _ = self._encode(images)
        initial, text_guidance, visual_guidance = self._initial_state(visual, text)
        cost, auxiliary = self._aggregate(initial, text_guidance, visual_guidance, return_aux=return_aux)
        result = {"logits": self._decode(images, cost, count)}
        if return_aux:
            result.update(auxiliary)
        return result

    def optimizer_groups(self, new_lr: float, head_lr: float, visual_lr: float) -> list[dict[str, Any]]:
        groups: dict[str, list[nn.Parameter]] = {"visual": [], "head": [], "new": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("cafe.backbone."):
                groups["visual"].append(parameter)
            elif name.startswith("channel.") and self.config.channel_init == "fresh":
                groups["new"].append(parameter)
            elif name.startswith(("cafe.", "spatial.", "channel.")):
                groups["head"].append(parameter)
            else:
                groups["new"].append(parameter)
        rates = {"visual": visual_lr, "head": head_lr, "new": new_lr}
        return [dict(params=params, role=role, lr=rates[role], initial_lr=rates[role])
                for role, params in groups.items() if params]

    def adapted_state_dict(self) -> dict[str, Tensor]:
        excluded = ("cafe.backbone.", "cafe.upsampler.")
        tuned = tuple(f"cafe.backbone.visual_model.backbone.blocks.{index}." for index in self.tuned_indices)
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if not name.startswith(excluded) or name.startswith(tuned)}

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected, actual = set(self.adapted_state_dict()), set(state)
        if actual != expected:
            raise ValueError(f"PCA-DINO checkpoint keys differ: missing={expected - actual}, unexpected={actual - expected}")
        result = self.load_state_dict(state, strict=False)
        allowed = ("cafe.backbone.", "cafe.upsampler.")
        if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
            raise RuntimeError("Incomplete PCA-DINO inference state")

    def architecture(self) -> dict[str, Any]:
        topology = {
            "serial": "published CAFe Spatial->Class aggregation",
            "parallel": "matched residual Spatial(C)||Class(C) control without EPL",
            "pca_epl": "parallel Spatial(C)||Class(C) with per-query dense EPL",
            "pca_epl_fod": "parallel Spatial(C)||Class(C) with per-query dense EPL and training-only FOD",
            "task_parallel": "parallel Spatial(C)||Class(C) with an anchored, query-local three-way gate",
        }[self.config.arm]
        if self.config.arm == "serial" and self.config.corrected_class_attention:
            topology = "CAFe Spatial->Class aggregation with corrected query-axis attention"
        return dict(
            name="DINO-TaskParallel-v1" if self.config.arm == "task_parallel" else "PCA-DINO-v1",
            **asdict(self.config),
            topology=topology,
            epl=("official four-parser topology with pixel-wise softmax coefficient mapper"
                 if self.config.arm.startswith("pca_epl") else "none"),
            task_gate=("anchored per-query and per-position three-way convex fusion"
                       if self.config.arm == "task_parallel" else "none"),
            fod="mean squared point-wise branch cosine; training only" if self.config.arm == "pca_epl_fod" else "none",
            class_attention=("corrected query-axis SDPA" if self.config.corrected_class_attention else "published CAFe implementation"),
            source_training="selected by the launcher; external datasets excluded from checkpoint selection",
            text_frozen=True,
            anyup_frozen=True,
            reduce_d_trainable=True,
            tuned_visual_indices=list(self.tuned_indices),
            trainable_parameters=sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            frozen_parameters=sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        )


def config_from_checkpoint(payload: dict) -> PCADINOConfig:
    architecture = payload.get("architecture", {})
    valid = {("cafe_pca_v1", "PCA-DINO-v1"),
             ("cafe_task_parallel_v1", "DINO-TaskParallel-v1")}
    if (payload.get("format"), architecture.get("name")) not in valid:
        raise ValueError("Not a supported PCA-DINO or TaskParallel checkpoint")
    config = PCADINOConfig(**{item.name: architecture.get(item.name, item.default) for item in fields(PCADINOConfig)})
    config.validate()
    return config


def pca_loss(output: dict[str, Tensor], target: Tensor, teacher_logits: Tensor, *, smoothing: float = 0.1,
             kd_weight: float = 0.02, fod_weight: float = 0.0) -> tuple[Tensor, dict[str, Tensor]]:
    loss, terms = ped_loss(output, target, teacher_logits, smoothing=smoothing, kd_weight=kd_weight)
    fod = output.get("fod_loss", loss * 0)
    if fod_weight and "fod_loss" not in output:
        raise ValueError("Positive FOD weight requires return_aux=True")
    terms["affinity_loss"] = fod
    terms["fod_loss"] = fod
    terms["branch_cosine"] = output.get("branch_cosine", loss * 0)
    return loss + fod_weight * fod, terms
