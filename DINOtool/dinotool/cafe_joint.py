"""Matched last-two-block fine-tuning of CAFe, concat, and RS."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .cafe_vc import CafeVC, CafeVCConfig
from .cafe_rs import CafeRS, CafeRSConfig, rs_losses


@dataclass(frozen=True)
class JointConfig:
    arm: str = "rs"
    visual_blocks: int = 2
    rs_dim: int = 256
    concat_dim: int = 128
    heads: int = 4
    stages: int = 3
    modes: int = 4

    def validate(self):
        if self.arm not in {"plain", "concat", "rs", "rs_aux"} or self.visual_blocks != 2:
            raise ValueError("Joint v1 supports four arms and exactly two tuned visual blocks")
        CafeRSConfig(content_dim=self.rs_dim, heads=self.heads, stages=self.stages, modes=self.modes).validate()
        CafeVCConfig(arm="concat", content_dim=self.concat_dim, heads=self.heads).validate()


class JointCafe(nn.Module):
    def __init__(self, cafe: nn.Module, config: JointConfig, *, official_root: str):
        super().__init__()
        config.validate()
        self.config = config
        self.warmup = False
        if config.arm.startswith("rs"):
            self.student = CafeRS(cafe, CafeRSConfig(content_dim=config.rs_dim, heads=config.heads,
                                                   stages=config.stages, modes=config.modes), official_root=official_root)
            if config.arm == "rs":
                for stage in self.student.reasoning:
                    stage.region_classifier.requires_grad_(False)
        else:
            self.student = CafeVC(cafe, CafeVCConfig(arm=config.arm, content_dim=config.concat_dim,
                                                    heads=config.heads, blocks=0 if config.arm == "plain" else 2))
        cafe.requires_grad_(False)
        blocks = cafe.backbone.visual_model.backbone.blocks
        if len(blocks) < config.visual_blocks:
            raise ValueError("Backbone has fewer blocks than requested")
        self.tuned_indices = tuple(range(len(blocks) - config.visual_blocks, len(blocks)))
        for index in self.tuned_indices:
            blocks[index].requires_grad_(True)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "aggregator", "reduce_d"):
            getattr(cafe, name).requires_grad_(True)
        self.student.joint_finetuning = True
        self.train(False)

    @property
    def cafe(self):
        return self.student.cafe

    def train(self, mode: bool = True):
        super().train(mode)
        self.cafe.eval()
        active = mode and not self.warmup
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].train(active)
        for name in ("corr_embed", "text_guidance_projection", "vis_guidance_projection", "aggregator", "reduce_d"):
            getattr(self.cafe, name).train(active)
        # Equal, fixed pretrained BN statistics across arms and source domains.
        # Affine BN parameters remain trainable; no rank-local running-stat drift.
        for module in self.cafe.reduce_d.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_warmup(self, warmup: bool):
        self.warmup = warmup
        self.train(self.training)

    def forward(self, images, text):
        if self.config.arm.startswith("rs"):
            return self.student(images, text, return_aux=self.training and self.config.arm == "rs_aux")
        return self.student(images, text)

    def optimizer_groups(self, new_lr: float, head_lr: float, visual_lr: float):
        groups = {"visual": [], "head": [], "new": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            kind = "visual" if name.startswith("student.cafe.backbone.") else "head" if name.startswith("student.cafe.") else "new"
            groups[kind].append(parameter)
        rates = dict(visual=visual_lr, head=head_lr, new=new_lr)
        return [dict(params=params, role=kind, lr=rates[kind], initial_lr=rates[kind]) for kind, params in groups.items() if params]

    def adapted_state_dict(self):
        decoder = {name: tensor.detach().cpu() for name, tensor in self.student.state_dict().items()
                   if not name.startswith(("cafe.backbone.", "cafe.upsampler."))}
        blocks = self.cafe.backbone.visual_model.backbone.blocks
        visual = {str(index): {name: tensor.detach().cpu() for name, tensor in blocks[index].state_dict().items()}
                  for index in self.tuned_indices}
        return dict(decoder=decoder, visual_blocks=visual)

    def load_adapted_state_dict(self, state):
        if set(state) != {"decoder", "visual_blocks"} or set(state["visual_blocks"]) != {str(i) for i in self.tuned_indices}:
            raise ValueError("Joint checkpoint must contain exactly the decoder and both tuned visual blocks")
        expected = {name for name in self.student.state_dict() if not name.startswith(("cafe.backbone.", "cafe.upsampler."))}
        if set(state["decoder"]) != expected:
            raise ValueError("Joint decoder checkpoint keys differ from the requested architecture")
        result = self.student.load_state_dict(state["decoder"], strict=False)
        if result.unexpected_keys or any(not k.startswith(("cafe.backbone.", "cafe.upsampler.")) for k in result.missing_keys):
            raise ValueError("Incomplete joint decoder checkpoint")
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].load_state_dict(state["visual_blocks"][str(index)], strict=True)

    def architecture(self):
        return {"name": "CAFe-Joint-v1", **asdict(self.config), "tuned_visual_indices": list(self.tuned_indices),
                "text_frozen": True, "anyup_frozen": True, "reduce_d_trainable": True, "bn_running_stats_frozen": True,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad)}


def joint_loss(output, target, teacher_logits, *, auxiliary: bool, smoothing=0.1, kd_weight=0.02):
    if auxiliary:
        return rs_losses({**output, "teacher_logits": teacher_logits}, target, smoothing, 0.05, 0.02, kd_weight)
    logits = output["logits"].float()
    valid = target != 255
    ce = F.cross_entropy(logits, target, ignore_index=255, label_smoothing=smoothing) if valid.any() else logits.sum() * 0
    correct = valid & (teacher_logits.argmax(1) == target)
    reference = teacher_logits.detach().float()
    divergence = F.kl_div((logits / 2).log_softmax(1), (reference / 2).softmax(1), reduction="none").sum(1) * 4
    kd = (divergence * correct).sum() / correct.sum().clamp_min(1)
    zero = logits.sum() * 0
    return ce + kd_weight * kd, dict(ce_loss=ce, kd_loss=kd, region_loss=zero, affinity_loss=zero)
