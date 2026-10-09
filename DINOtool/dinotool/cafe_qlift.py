"""CAFe-DINO adapter for the controlled Q-Lift decoder study."""
from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .qlift import QLiftConfig, QLiftDecoder


class CafeQLift(nn.Module):
    """Replace CAFe aggregation with a reconstructible query-cost decoder.

    The published CAFe aggregator remains loaded but frozen and unused. This
    keeps the base checkpoint verifiable while making the replacement explicit.
    """

    def __init__(self, cafe: nn.Module, config: QLiftConfig) -> None:
        super().__init__()
        config.validate()
        self.cafe = cafe
        self.config = config
        self.cost_dim = int(cafe.aggregator_dim)
        self.qlift = QLiftDecoder(self.cost_dim, config)
        self.warmup = False
        blocks = self.cafe.backbone.visual_model.backbone.blocks
        if len(blocks) < 2:
            raise ValueError("CAFe-DINO backbone must expose at least two visual blocks")
        self.tuned_indices = (tuple(range(len(blocks) - 2, len(blocks)))
                              if config.tune_visual_blocks else ())
        self.joint_finetuning = bool(self.tuned_indices)
        self._configure_trainable_parameters()
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_qlift_v1"

    def _configure_trainable_parameters(self) -> None:
        self.cafe.requires_grad_(False)
        self.qlift.requires_grad_(True)
        # These two CAFe interfaces remain part of the learned dense readout.
        self.cafe.corr_embed.requires_grad_(True)
        self.cafe.reduce_d.requires_grad_(True)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].requires_grad_(True)

    def train(self, mode: bool = True) -> "CafeQLift":
        nn.Module.train(self, mode)
        self.cafe.eval()
        self.qlift.train(mode)
        active = mode and not self.warmup
        self.cafe.corr_embed.train(active)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].train(active)
        # AnyUp and BN running statistics remain fixed under the locked source
        # protocol, while reduce_d convolution weights can still receive grads.
        self.cafe.reduce_d.eval()
        for module in self.cafe.reduce_d.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_warmup(self, warmup: bool) -> None:
        self.warmup = warmup
        active = not warmup
        self.cafe.corr_embed.requires_grad_(active)
        self.cafe.reduce_d.requires_grad_(active)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].requires_grad_(active)
        self.joint_finetuning = bool(self.tuned_indices) and active
        self.train(self.training)

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

    def _initial_state(self, x: Tensor, text: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        classes = len(text)
        raw = torch.einsum("bdhw,cd->bchw", F.normalize(x, dim=1), F.normalize(text, dim=-1))
        cost = self.cafe.corr_embed(raw.reshape(batch * classes, 1, height, width))
        return cost.reshape(batch, classes, self.cost_dim, height, width).transpose(1, 2)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                return_aux: bool = False, lifting_text: Tensor | None = None,
                lifting_query_conditioned: bool | None = None,
                gain_text: Tensor | None = None, gain_query_conditioned: bool | None = None,
                detail_mode: str = "native") -> dict[str, Tensor | dict[str, Tensor]]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(size % 16 for size in images.shape[-2:]):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty Q x 1024 DINO.text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        batch, _, image_height, image_width = images.shape
        height, width = image_height // 16, image_width // 16
        x, y = self._encode(images)
        initial = self._initial_state(x, text)
        cost, trace = self.qlift(
            initial, x, y, text, return_trace=return_aux, lifting_text=lifting_text,
            lifting_query_conditioned=lifting_query_conditioned, gain_text=gain_text,
            gain_query_conditioned=gain_query_conditioned, detail_mode=detail_mode,
        )
        selected = cost[:, :, :count].contiguous().reshape(batch, self.cost_dim * count, height, width)
        upsampled = self.cafe.upsampler(images, selected)
        features = upsampled.reshape(batch, self.cost_dim, count, image_height, image_width)
        features = features.transpose(1, 2).reshape(batch * count, self.cost_dim, image_height, image_width)
        result: dict[str, Tensor | dict[str, Tensor]] = {
            "logits": self.cafe.reduce_d(features).reshape(batch, count, image_height, image_width)
        }
        if return_aux:
            result["qlift"] = trace
        return result

    def optimizer_groups(self, new_lr: float, head_lr: float, visual_lr: float) -> list[dict[str, Any]]:
        groups: dict[str, list[nn.Parameter]] = {"visual": [], "head": [], "new": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("cafe.backbone."):
                groups["visual"].append(parameter)
            elif name.startswith("cafe."):
                groups["head"].append(parameter)
            else:
                groups["new"].append(parameter)
        rates = {"visual": visual_lr, "head": head_lr, "new": new_lr}
        return [dict(params=params, role=role, lr=rates[role], initial_lr=rates[role])
                for role, params in groups.items() if params]

    def adapted_state_dict(self) -> dict[str, Tensor]:
        tuned = tuple(f"cafe.backbone.visual_model.backbone.blocks.{index}." for index in self.tuned_indices)
        included = ("qlift.", "cafe.corr_embed.", "cafe.reduce_d.", *tuned)
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name.startswith(included)}

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = set(self.adapted_state_dict())
        actual = set(state)
        if expected != actual:
            raise ValueError(f"QLift checkpoint keys differ: missing={expected - actual}, unexpected={actual - expected}")
        result = self.load_state_dict(state, strict=False)
        allowed = ("cafe.backbone.", "cafe.upsampler.", "cafe.aggregator.",
                   "cafe.text_guidance_projection.", "cafe.vis_guidance_projection.")
        if result.unexpected_keys or any(not name.startswith(allowed) for name in result.missing_keys):
            raise RuntimeError("Incomplete QLift inference state")

    def architecture(self) -> dict[str, Any]:
        topology = {
            "skip": "full-resolution cost skips with capacity-matched image-only state-update modulation",
            "haar": "fixed 2x2 Haar analysis/synthesis with capacity-matched image-only state-update modulation",
            "image_lift": "image-conditioned local P/U lifting with cached inverse synthesis",
            "query_lift": "query-conditioned local P/U lifting with cached inverse synthesis",
        }[self.config.arm]
        return {
            "name": "Q-Lift-DINO-v1",
            **asdict(self.config),
            "topology": topology,
            "old_cafe_aggregator": "loaded from official checkpoint but frozen and bypassed",
            "details": "all three 2x2 polyphase detail streams retained at every level",
            "capacity_control": ("all arms own identical local P/U generators; skip/Haar use them only to modulate "
                                 "state updates, not to change their skip/Haar analysis or synthesis"),
            "padding": "right/bottom replicate pad before analysis and crop after synthesis",
            "query_conditioning": ("shared text-conditioned state updates plus query-conditioned P/U"
                                    if self.config.arm == "query_lift" else
                                    "shared text-conditioned state updates; P/U does not receive query text"),
            "inverse_contract": "same cached P/U used for analysis and synthesis; identity when updates are inactive",
            "text_frozen": True,
            "anyup_frozen": True,
            "reduce_d_trainable": True,
            "bn_running_stats_frozen": True,
            "tuned_visual_indices": list(self.tuned_indices),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen_parameters": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        }


def config_from_checkpoint(payload: dict) -> QLiftConfig:
    architecture = payload.get("architecture", {})
    if payload.get("format") != "cafe_qlift_v1" or architecture.get("name") != "Q-Lift-DINO-v1":
        raise ValueError("Not a Q-Lift-DINO v1 checkpoint")
    config = QLiftConfig(**{item.name: architecture[item.name] for item in fields(QLiftConfig)})
    config.validate()
    return config
