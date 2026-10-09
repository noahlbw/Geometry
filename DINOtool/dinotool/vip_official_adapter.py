"""Frozen VIP inference using the pinned upstream proxy-attention implementation."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .prompts import load_vip_official_aliases


@dataclass(frozen=True)
class VIPSettings:
    tau: float = 4.0
    tem: float = 1.0
    prob_thd: float = 0.0
    background: bool = False
    bg_idx: int = 0
    resize_long_edge: int = 448
    slide_crop: int = 336
    slide_stride: int = 112
    logit_scale: float = 40.0


@dataclass(frozen=True)
class VIPQueries:
    features: Tensor  # [alias, ImageNet template, 1024]
    parents: Tensor
    class_names: tuple[str, ...]
    aliases: tuple[str, ...]


def upstream_settings(dataset: str) -> VIPSettings:
    if dataset == "potsdam":
        return VIPSettings(tau=1.0, tem=2.0, prob_thd=0.25, background=True, bg_idx=5)
    if dataset == "vdd":
        return VIPSettings(tau=1.0, tem=1.0, prob_thd=0.35, background=True, bg_idx=0)
    if dataset == "vaihingen":
        return VIPSettings(tau=1.0, tem=10.0, prob_thd=0.1, background=True, bg_idx=5)
    return VIPSettings()


def upstream_aliases(path: Path) -> tuple[tuple[str, ...], ...]:
    return load_vip_official_aliases(path)


def _load_upstream(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned VIP source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VIPOfficialAdapter:
    def __init__(self, backbone, upstream_root: Path):
        self.backbone = backbone
        self.upstream_root = upstream_root.resolve()
        vision_source = self.upstream_root / "dinov3/eval/text/vision_tower.py"
        prompt_source = self.upstream_root / "prompts/imagenet_template.py"
        self.vip_head = _load_upstream(vision_source, "pinned_vip_vision_tower").VisionHead
        prompt_module = _load_upstream(prompt_source, "pinned_vip_imagenet_templates")
        self.templates = prompt_module.get_text_template("openai_imagenet_template")
        self.backbone.model.half().eval().requires_grad_(False)
        self.backbone.model.visual_model.head.patch_size = 21
        if len(self.backbone.model.visual_model.head.blocks) != 2:
            raise ValueError("VIP requires both frozen DINO.text vision-head blocks.")

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @torch.inference_mode()
    def encode_queries(self, class_names: tuple[str, ...], aliases_by_class: tuple[tuple[str, ...], ...]) -> VIPQueries:
        if len(class_names) != len(aliases_by_class) or any(not aliases for aliases in aliases_by_class):
            raise ValueError("Each scored class must have at least one VIP query.")
        aliases = tuple(alias for group in aliases_by_class for alias in group)
        parents = torch.tensor([index for index, group in enumerate(aliases_by_class)
                                for _ in group], device=self.device, dtype=torch.long)
        features = []
        for alias in aliases:
            prompts = [template(alias) for template in self.templates]
            encoded = []
            for start in range(0, len(prompts), 40):
                tokens = self.backbone.tokenize(prompts[start:start + 40]).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feature = self.backbone.model.encode_text(tokens)
                encoded.append(F.normalize(feature[:, 1024:].float(), dim=-1).half())
            features.append(torch.cat(encoded, dim=0))
        return VIPQueries(torch.stack(features), parents, class_names, aliases)

    @torch.inference_mode()
    def crop_patch_features(self, rgb: Tensor) -> Tensor:
        if rgb.shape != (3, 336, 336):
            raise ValueError("Pinned VIP head requires an exact 336x336 crop (21x21 patches).")
        image = rgb.unsqueeze(0).to(self.device)
        image = ((image - self.backbone._imagenet_mean) / self.backbone._imagenet_std).half()
        visual = self.backbone.model.visual_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls, raw, registers = visual.get_backbone_features(image)
            tokens = torch.cat((cls.unsqueeze(1), registers, raw), dim=1)
            head = visual.head
            for block in head.blocks:
                attention = self.vip_head.proxy_attn(head, block.attn, block.norm1(tokens), raw)
                tokens = tokens + block.ls1(attention)
                tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
            projected = head.linear_projection(head.ln_final(tokens))
            return F.normalize(projected[:, 1 + registers.shape[1]:].float(), dim=-1)

    @torch.inference_mode()
    def crop_logits(self, rgb: Tensor, queries: VIPQueries, settings: VIPSettings) -> Tensor:
        patch = self.crop_patch_features(rgb)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            similarities = torch.einsum("bnd,mtd->bnmt", patch, queries.features.float()).mean(-1)[0]
            patch_mean = F.normalize(patch.mean(1), dim=-1)
            text_mean = F.normalize(queries.features.float().mean(1), dim=-1)
            salience = ((patch_mean @ text_mean.T)[0] / settings.tem).float()
            alias_logits = (similarities * settings.logit_scale).T.reshape(-1, 21, 21)
            class_logits = []
            for index in range(len(queries.class_names)):
                members = queries.parents == index
                weights = salience[members].softmax(0)
                scaled = alias_logits[members] * (weights / weights.mean())[:, None, None]
                class_logits.append(torch.logsumexp(settings.tau * scaled, dim=0) / settings.tau)
            logits = torch.stack(class_logits)[None]
            return F.interpolate(logits, size=(336, 336), mode="bilinear", align_corners=False)[0].float()

    @torch.inference_mode()
    def predict_probabilities(self, rgb: Tensor, queries: VIPQueries,
                              settings: VIPSettings) -> tuple[Tensor, int]:
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError("RGB image must have shape [3,H,W].")
        original_height, original_width = rgb.shape[-2:]
        ratio = settings.resize_long_edge / max(original_height, original_width)
        height = int(original_height * ratio + 0.5)
        width = int(original_width * ratio + 0.5)
        array = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255)
        count = torch.zeros((1, height, width), device=self.device)
        logits_sum = torch.zeros((len(queries.class_names), height, width), device=self.device)
        short_edge_pads = 0
        y_count = max(height - settings.slide_crop + settings.slide_stride - 1, 0) // settings.slide_stride + 1
        x_count = max(width - settings.slide_crop + settings.slide_stride - 1, 0) // settings.slide_stride + 1
        for iy in range(y_count):
            for ix in range(x_count):
                bottom = min(iy * settings.slide_stride + settings.slide_crop, height)
                right = min(ix * settings.slide_stride + settings.slide_crop, width)
                top = max(bottom - settings.slide_crop, 0)
                left = max(right - settings.slide_crop, 0)
                crop = image[:, top:bottom, left:right]
                crop_height, crop_width = crop.shape[-2:]
                if crop_height < settings.slide_crop or crop_width < settings.slide_crop:
                    # Upstream fixes the head at 21x21; keep-ratio panoramas need this fallback.
                    crop = F.pad(crop, (0, settings.slide_crop - crop_width,
                                        0, settings.slide_crop - crop_height))
                    short_edge_pads += 1
                window = self.crop_logits(crop, queries, settings)[:, :crop_height, :crop_width]
                logits_sum[:, top:bottom, left:right] += window
                count[:, top:bottom, left:right] += 1
        logits = logits_sum / count
        logits = F.interpolate(logits[None], size=(original_height, original_width),
                               mode="bilinear", align_corners=False)[0]
        return logits.softmax(0), short_edge_pads

    @torch.inference_mode()
    def predict(self, rgb: Tensor, queries: VIPQueries, settings: VIPSettings) -> tuple[np.ndarray, int]:
        probabilities, short_edge_pads = self.predict_probabilities(rgb, queries, settings)
        prediction = probabilities.argmax(0)
        if settings.background:
            prediction = prediction.masked_fill(probabilities.amax(0) < settings.prob_thd,
                                                settings.bg_idx)
        return prediction.to(torch.uint8).cpu().numpy(), short_edge_pads
