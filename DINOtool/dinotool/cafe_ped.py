"""CAFe-DINO decoder variants for the parallel-evidence training study."""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .parallel_evidence import (
    CompressedVisualMemory,
    EvidenceReadout,
    ParallelEvidenceConfig,
    QueryConditionedEvidenceRead,
    QueryVisualRead,
    SpatiallyAnchoredPyramid,
    SynchronousEvidenceExchange,
    VisualEvidenceField,
    ZeroCostFeedback,
)


def ped_loss(output: dict[str, Tensor], target: Tensor, teacher_logits: Tensor, *, smoothing: float = 0.1,
             kd_weight: float = 0.02) -> tuple[Tensor, dict[str, Tensor]]:
    """The locked CE plus teacher-KL loss, kept local to the PED package."""
    logits = output["logits"].float()
    valid = target != 255
    ce = F.cross_entropy(logits, target, ignore_index=255, label_smoothing=smoothing) if valid.any() else logits.sum() * 0
    correct = valid & (teacher_logits.argmax(1) == target)
    divergence = F.kl_div((logits / 2).log_softmax(1), (teacher_logits.detach().float() / 2).softmax(1), reduction="none").sum(1) * 4
    kd = (divergence * correct).sum() / correct.sum().clamp_min(1)
    zero = logits.sum() * 0
    return ce + kd_weight * kd, {"ce_loss": ce, "kd_loss": kd, "region_loss": zero, "affinity_loss": zero}


class CafePED(nn.Module):
    """CAFe cost-decoder controls and the spatial support/surround revision.

    The constructor copies CAFe's published spatial/channel aggregators into
    the selected topology and removes the unused original aggregator list. It
    therefore cannot silently execute both the old and new decoders.
    """

    def __init__(self, cafe: nn.Module, config: ParallelEvidenceConfig) -> None:
        super().__init__()
        config.validate()
        if len(cafe.aggregator) != config.stages:
            raise ValueError(f"Requested {config.stages} stages, official CAFe has {len(cafe.aggregator)}")
        self.cafe = cafe
        self.config = config
        self.warmup = False
        # This wrapper is always optimized through the locked joint protocol.
        self.joint_finetuning = True
        self.cost_dim = cafe.aggregator_dim
        self.spatial = nn.ModuleList()
        self.channel = nn.ModuleList()
        if config.arm == "plain":
            # Keep the published serial aggregation path intact for the
            # same-seed control. Other arms own copied operators instead.
            pass
        else:
            self.spatial = nn.ModuleList(copy.deepcopy(layer.spatial_agg) for layer in cafe.aggregator)
            self.channel = nn.ModuleList(copy.deepcopy(layer.class_agg) for layer in cafe.aggregator)
            cafe.aggregator = nn.ModuleList()
        self.visual_field = None
        self.memory = None
        self.reader = None
        self.exchange = None
        self.readout = None
        self.pyramid = None
        self.relation_reader = None
        self.semantic_feedback = None
        self.spatial_feedback = None
        self.relation_feedback = None
        if config.arm in {"serial_content", "ped"}:
            self.visual_field = VisualEvidenceField(config.evidence_dim, config.visual_blocks)
            self.memory = CompressedVisualMemory(config.evidence_dim, config.memory_tokens, config.memory_heads)
            self.reader = QueryVisualRead(self.cost_dim, 1024, config)
            self.exchange = SynchronousEvidenceExchange(self.cost_dim, config)
            self.readout = EvidenceReadout(self.cost_dim, config.message_hidden)
        elif config.arm in {"anchored_multiscale", "paired_support"}:
            self.visual_field = VisualEvidenceField(config.evidence_dim, config.visual_blocks)
            self.pyramid = SpatiallyAnchoredPyramid(config.pyramid_grids)
            self.relation_reader = QueryConditionedEvidenceRead(self.cost_dim, 1024, config)
            self.semantic_feedback = ZeroCostFeedback(self.cost_dim)
            self.spatial_feedback = ZeroCostFeedback(self.cost_dim)
            self.relation_feedback = ZeroCostFeedback(self.cost_dim)
        self._configure_trainable_parameters()
        self.train(False)

    def _configure_trainable_parameters(self) -> None:
        self.cafe.requires_grad_(False)
        if self.config.arm == "plain":
            self.cafe.aggregator.requires_grad_(True)
        else:
            self.spatial.requires_grad_(True)
            self.channel.requires_grad_(True)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "reduce_d"):
            getattr(self.cafe, name).requires_grad_(True)
        for module in (self.visual_field, self.memory, self.reader, self.exchange, self.readout, self.pyramid,
                       self.relation_reader, self.semantic_feedback, self.spatial_feedback, self.relation_feedback):
            if module is not None:
                module.requires_grad_(True)
        blocks = self.cafe.backbone.visual_model.backbone.blocks
        if len(blocks) < 2:
            raise ValueError("CAFe-DINO backbone must expose at least two visual transformer blocks")
        self.tuned_indices = tuple(range(len(blocks) - 2, len(blocks)))
        for index in self.tuned_indices:
            blocks[index].requires_grad_(True)

    @property
    def uses_content(self) -> bool:
        return self.config.arm in {"serial_content", "ped", "anchored_multiscale", "paired_support"}

    @property
    def uses_targeted_readout(self) -> bool:
        return self.config.arm in {"anchored_multiscale", "paired_support"}

    @property
    def checkpoint_format(self) -> str:
        return "cafe_ped_v2" if self.config.arm in {"plain", "naive_parallel", "anchored_multiscale", "paired_support"} else "cafe_ped_v1"

    def train(self, mode: bool = True) -> "CafePED":
        super().train(mode)
        self.cafe.backbone.eval()
        self.cafe.upsampler.eval()
        self.cafe.reduce_d.eval()
        old_active = mode and not self.warmup
        if self.config.arm == "plain":
            self.cafe.aggregator.train(old_active)
        else:
            self.spatial.train(old_active)
            self.channel.train(old_active)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].train(old_active)
        for module in (self.cafe.corr_embed, self.cafe.text_guidance_projection, self.cafe.vis_guidance_projection):
            module.train(old_active)
        for module in (self.visual_field, self.memory, self.reader, self.exchange, self.readout, self.pyramid,
                       self.relation_reader, self.semantic_feedback, self.spatial_feedback, self.relation_feedback):
            if module is not None:
                module.train(mode)
        # Fixed BN statistics are part of the matched source protocol.
        for module in self.cafe.reduce_d.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_warmup(self, warmup: bool) -> None:
        self.warmup = warmup
        self.train(self.training)

    def _set_aggregator_resolution(self, height: int, width: int, device: torch.device) -> None:
        modules = (layer.spatial_agg for layer in self.cafe.aggregator) if self.config.arm == "plain" else self.spatial
        for module in modules:
            for block in (module.swin1, module.swin2):
                if tuple(block.input_resolution) != (height, width):
                    block.set_input_size((height, width), (7, 7))
                    if block.attn_mask is not None:
                        block.attn_mask = block.attn_mask.to(device)

    def _initial_state(self, x: Tensor, text: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, height, width = x.shape
        classes = len(text)
        raw = torch.einsum("bdhw,cd->bchw", F.normalize(x, dim=1), F.normalize(text, dim=-1))
        cost = self.cafe.corr_embed(raw.reshape(batch * classes, 1, height, width))
        cost = cost.reshape(batch, classes, self.cost_dim, height, width).transpose(1, 2)
        text_guidance = self.cafe.text_guidance_projection(text)
        text_guidance = text_guidance[None, None, None].expand(batch, height, width, -1, -1)
        text_guidance = text_guidance.reshape(batch * height * width, classes, self.cost_dim)
        visual_guidance = self.cafe.vis_guidance_projection(x).permute(0, 2, 3, 1)
        visual_guidance = visual_guidance[:, None].expand(-1, classes, -1, -1, -1)
        visual_guidance = visual_guidance.reshape(batch * classes, height, width, self.cost_dim)
        return cost, text_guidance, visual_guidance

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

    def _naive_parallel(self, initial: Tensor, text_guidance: Tensor, visual_guidance: Tensor) -> Tensor:
        cost = initial
        for spatial, channel in zip(self.spatial, self.channel):
            spatial_cost = spatial(cost, visual_guidance)
            class_cost = channel(cost, text_guidance)
            # Both operators contain their own identity paths; subtract the
            # duplicated state to make this a true residual parallel control.
            cost = cost + (spatial_cost - cost) + (class_cost - cost)
        return cost

    def _plain(self, initial: Tensor, text_guidance: Tensor, visual_guidance: Tensor) -> Tensor:
        cost = initial
        for aggregator in self.cafe.aggregator:
            cost = aggregator(cost, text_guidance, visual_guidance)
        return cost

    def _targeted_parallel(self, initial: Tensor, text_guidance: Tensor, visual_guidance: Tensor,
                           pyramid: tuple[Tensor, ...], text: Tensor, *, paired: bool) -> Tensor:
        """Parallel CAFe updates with zero-initialized persistent evidence feedback.

        At initialization every feedback projection is exactly zero. The full
        six-stage cost path is therefore the same function as naive-parallel;
        only training can make anchored evidence alter a prediction.
        """
        assert self.relation_reader is not None
        assert self.semantic_feedback is not None and self.spatial_feedback is not None
        assert self.relation_feedback is not None
        cost = initial
        semantic_state = torch.zeros_like(initial)
        spatial_state = torch.zeros_like(initial)
        relation_state = torch.zeros_like(initial)
        for spatial, channel in zip(self.spatial, self.channel):
            class_cost = channel(cost + self.semantic_feedback(semantic_state), text_guidance)
            visual_cost = spatial(cost + self.spatial_feedback(spatial_state), visual_guidance)
            relation_state = relation_state + self.relation_reader(cost, relation_state, pyramid, text, paired=paired)
            semantic_state = class_cost - cost
            spatial_state = visual_cost - cost
            cost = cost + (visual_cost - cost) + (class_cost - cost) + self.relation_feedback(relation_state)
        return cost

    def _serial_content(self, initial: Tensor, text_guidance: Tensor, visual_guidance: Tensor,
                        field: Tensor, memory: Tensor, text: Tensor) -> Tensor:
        assert self.reader is not None and self.exchange is not None and self.readout is not None
        cost = initial
        for spatial, channel in zip(self.spatial, self.channel):
            spatial_cost = spatial(cost, visual_guidance)
            class_cost = channel(spatial_cost, text_guidance)
            visual_cost = self.reader(class_cost, field, memory, text, initial)
            a, s, r = self.exchange(class_cost, spatial_cost, visual_cost, initial)
            cost = (a + s + r) / 3
        return self.readout(initial, cost, cost, cost)

    def _parallel_evidence(self, initial: Tensor, text_guidance: Tensor, visual_guidance: Tensor,
                           field: Tensor, memory: Tensor, text: Tensor) -> Tensor:
        assert self.reader is not None and self.exchange is not None and self.readout is not None
        a = s = r = initial
        for spatial, channel in zip(self.spatial, self.channel):
            # These three calls all consume old-round states, then exchange.
            a_hat = channel(a, text_guidance)
            s_hat = spatial(s, visual_guidance)
            r_hat = self.reader(r, field, memory, text, initial)
            a, s, r = self.exchange(a_hat, s_hat, r_hat, initial)
        return self.readout(initial, a, s, r)

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
        x, y = self._encode(images)
        initial, text_guidance, visual_guidance = self._initial_state(x, text)
        if self.config.arm == "plain":
            cost = self._plain(initial, text_guidance, visual_guidance)
        elif self.config.arm == "naive_parallel":
            cost = self._naive_parallel(initial, text_guidance, visual_guidance)
        else:
            assert self.visual_field is not None
            field = self.visual_field(x, y)
            if self.uses_targeted_readout:
                assert self.pyramid is not None
                cost = self._targeted_parallel(initial, text_guidance, visual_guidance, self.pyramid(field), text,
                                               paired=self.config.arm == "paired_support")
            else:
                assert self.memory is not None
                memory = self.memory(field)
                cost = (self._serial_content(initial, text_guidance, visual_guidance, field, memory, text)
                        if self.config.arm == "serial_content"
                        else self._parallel_evidence(initial, text_guidance, visual_guidance, field, memory, text))
        selected = cost[:, :, :count].contiguous().reshape(batch, -1, height, width)
        upsampled = self.cafe.upsampler(images, selected)
        features = upsampled.reshape(batch, self.cost_dim, count, image_height, image_width)
        features = features.transpose(1, 2).reshape(batch * count, self.cost_dim, image_height, image_width)
        logits = self.cafe.reduce_d(features).reshape(batch, count, image_height, image_width)
        return {"logits": logits}

    def optimizer_groups(self, new_lr: float, head_lr: float, visual_lr: float) -> list[dict[str, Any]]:
        groups: dict[str, list[nn.Parameter]] = {"visual": [], "head": [], "new": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("cafe.backbone."):
                groups["visual"].append(parameter)
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
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith(excluded) or name.startswith(tuned)
        }

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = set(self.adapted_state_dict())
        actual = set(state)
        if actual != expected:
            raise ValueError(f"PED checkpoint keys differ: missing={expected - actual}, unexpected={actual - expected}")
        result = self.load_state_dict(state, strict=False)
        allowed = ("cafe.backbone.", "cafe.upsampler.")
        if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
            raise RuntimeError("Incomplete PED inference state")

    def architecture(self) -> dict[str, Any]:
        topology = {
            "plain": "published CAFe serial Spatial->Class aggregation",
            "naive_parallel": "C+(Spatial(C)-C)+(Class(C)-C)",
            "serial_content": "Spatial->Class->VisualRead then collapse states each round",
            "ped": "persistent class/spatial/visual states, synchronous exchange",
            "anchored_multiscale": "naive-parallel main cost plus merged anchored deformable evidence feedback",
            "paired_support": "naive-parallel main cost plus separately normalized support/surround relation feedback",
        }[self.config.arm]
        return {
            "name": "CAFe-PED-v2" if self.checkpoint_format == "cafe_ped_v2" else "CAFe-PED-v1",
            **asdict(self.config),
            "topology": topology,
            "original_aggregator_weights": "published modules retained" if self.config.arm == "plain" else "copied into distinct spatial/channel branches",
            "visual_read": ("separately normalized query-conditioned support/surround reads"
                            if self.config.arm == "paired_support" else
                            "merged query-conditioned anchored multi-scale reads" if self.config.arm == "anchored_multiscale" else
                            "3x3 local plus compressed image memory" if self.uses_content else "none"),
            "initial_function": "naive-parallel exact match via zero feedback" if self.uses_targeted_readout else "not applicable",
            "text_frozen": True,
            "anyup_frozen": True,
            "reduce_d_trainable": True,
            "bn_running_stats_frozen": True,
            "tuned_visual_indices": list(self.tuned_indices),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen_parameters": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        }
