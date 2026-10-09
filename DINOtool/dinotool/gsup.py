from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import GSUPConfig


@dataclass(frozen=True)
class GSUPDiagnostics:
    steps: int
    initial_reconstruction_l1: float
    final_reconstruction_l1: float


@dataclass
class GaussianParameters:
    log_scale: Tensor
    angle: Tensor
    log_color_sigma: Tensor

    def trainable(self) -> list[Tensor]:
        return [self.log_scale, self.angle, self.log_color_sigma]

    def detach(self) -> "GaussianParameters":
        return GaussianParameters(
            log_scale=self.log_scale.detach(),
            angle=self.angle.detach(),
            log_color_sigma=self.log_color_sigma.detach(),
        )


class GaussianSplatUpsampler:
    """RGB-guided 2D Gaussian splatting with per-image test-time optimization."""

    def __init__(self, config: GSUPConfig) -> None:
        config.validate()
        self.config = config

    def fit(self, guidance_rgb: Tensor, low_size: tuple[int, int]) -> tuple[GaussianParameters, GSUPDiagnostics]:
        if guidance_rgb.ndim != 4 or guidance_rgb.shape[0] != 1 or guidance_rgb.shape[1] != 3:
            raise ValueError("GSUP currently expects guidance_rgb with shape [1, 3, H, W].")
        target = _resize_for_optimization(guidance_rgb.float(), self.config.optimization_size)
        low_rgb = F.interpolate(target, size=low_size, mode="area")
        parameters = self._initial_parameters(low_size, guidance_rgb.device)

        with torch.enable_grad():
            initial = self._reconstruction_loss(low_rgb, target, parameters)
            if self.config.steps:
                optimizer = torch.optim.Adam(parameters.trainable(), lr=self.config.learning_rate)
                for _ in range(self.config.steps):
                    optimizer.zero_grad(set_to_none=True)
                    reconstruction = self.render(low_rgb, target, target.shape[-2:], parameters)
                    loss = F.l1_loss(reconstruction, target)
                    loss.backward()
                    optimizer.step()
                    self._stabilize(parameters)
            final = self._reconstruction_loss(low_rgb, target, parameters)

        diagnostics = GSUPDiagnostics(
            steps=self.config.steps,
            initial_reconstruction_l1=float(initial.item()),
            final_reconstruction_l1=float(final.item()),
        )
        return parameters.detach(), diagnostics

    def upsample(self, values: Tensor, guidance_rgb: Tensor, parameters: GaussianParameters) -> Tensor:
        if values.ndim != 4 or values.shape[0] != 1:
            raise ValueError("GSUP currently expects values with shape [1, C, H, W].")
        return self.render(values.float(), guidance_rgb.float(), guidance_rgb.shape[-2:], parameters).to(values.dtype)

    def render(
        self,
        values: Tensor,
        guidance_rgb: Tensor,
        output_size: tuple[int, int],
        parameters: GaussianParameters,
    ) -> Tensor:
        low_h, low_w = values.shape[-2:]
        out_h, out_w = output_size
        if guidance_rgb.shape[-2:] != output_size:
            guidance_rgb = F.interpolate(guidance_rgb, size=output_size, mode="bilinear", align_corners=False)
        low_guidance = F.interpolate(guidance_rgb, size=(low_h, low_w), mode="area")
        outputs: list[Tensor] = []
        total = out_h * out_w
        for start in range(0, total, self.config.chunk_pixels):
            stop = min(start + self.config.chunk_pixels, total)
            outputs.append(
                self._render_chunk(values, low_guidance, guidance_rgb, output_size, parameters, start, stop)
            )
        flattened = torch.cat(outputs, dim=0)
        return flattened.transpose(0, 1).reshape(1, values.shape[1], out_h, out_w)

    def _render_chunk(
        self,
        values: Tensor,
        low_guidance: Tensor,
        high_guidance: Tensor,
        output_size: tuple[int, int],
        parameters: GaussianParameters,
        start: int,
        stop: int,
    ) -> Tensor:
        low_h, low_w = values.shape[-2:]
        out_h, out_w = output_size
        flat = torch.arange(start, stop, device=values.device)
        pixel_y = torch.div(flat, out_w, rounding_mode="floor")
        pixel_x = flat.remainder(out_w)
        low_y = (pixel_y.float() + 0.5) * low_h / out_h - 0.5
        low_x = (pixel_x.float() + 0.5) * low_w / out_w - 0.5

        side = int(math.sqrt(self.config.neighbors))
        offsets = torch.arange(1 - side // 2, 1 + side // 2, device=values.device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        base_y = torch.floor(low_y).long().unsqueeze(1)
        base_x = torch.floor(low_x).long().unsqueeze(1)
        query_y = base_y + offset_y.flatten().unsqueeze(0)
        query_x = base_x + offset_x.flatten().unsqueeze(0)
        valid = (query_y >= 0) & (query_y < low_h) & (query_x >= 0) & (query_x < low_w)
        clipped_y = query_y.clamp(0, low_h - 1)
        clipped_x = query_x.clamp(0, low_w - 1)
        indices = clipped_y * low_w + clipped_x

        gathered_values = _gather_map(values, indices)
        gathered_low_rgb = _gather_map(low_guidance, indices)
        scale = _gather_map(parameters.log_scale, indices).exp()
        angle = _gather_map(parameters.angle, indices)[..., 0]
        color_sigma = _gather_map(parameters.log_color_sigma, indices)[..., 0].exp().clamp_min(1e-4)

        delta_x = low_x.unsqueeze(1) - query_x.float()
        delta_y = low_y.unsqueeze(1) - query_y.float()
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        rotated_x = cosine * delta_x + sine * delta_y
        rotated_y = -sine * delta_x + cosine * delta_y
        spatial = -0.5 * ((rotated_x / scale[..., 0]).square() + (rotated_y / scale[..., 1]).square())

        high_rgb = high_guidance[0, :, pixel_y, pixel_x].transpose(0, 1)
        color_distance = (gathered_low_rgb - high_rgb.unsqueeze(1)).square().sum(dim=-1)
        color = -color_distance / (2.0 * color_sigma.square())
        weight_logits = (spatial + color).masked_fill(~valid, torch.finfo(spatial.dtype).min)
        weights = torch.softmax(weight_logits, dim=1)
        return (gathered_values * weights.unsqueeze(-1)).sum(dim=1)

    def _initial_parameters(self, low_size: tuple[int, int], device: torch.device) -> GaussianParameters:
        low_h, low_w = low_size
        spatial = math.log(self.config.initial_spatial_scale)
        color = math.log(self.config.initial_color_sigma)
        return GaussianParameters(
            log_scale=torch.full((1, 2, low_h, low_w), spatial, device=device, requires_grad=True),
            angle=torch.zeros((1, 1, low_h, low_w), device=device, requires_grad=True),
            log_color_sigma=torch.full((1, 1, low_h, low_w), color, device=device, requires_grad=True),
        )

    def _reconstruction_loss(self, low_rgb: Tensor, target: Tensor, parameters: GaussianParameters) -> Tensor:
        with torch.no_grad():
            reconstruction = self.render(low_rgb, target, target.shape[-2:], parameters)
            return F.l1_loss(reconstruction, target)

    @staticmethod
    def _stabilize(parameters: GaussianParameters) -> None:
        with torch.no_grad():
            parameters.log_scale.clamp_(math.log(0.15), math.log(4.0))
            parameters.log_color_sigma.clamp_(math.log(0.01), math.log(1.0))
            parameters.angle.remainder_(2 * math.pi)


def _gather_map(value: Tensor, indices: Tensor) -> Tensor:
    flattened = value[0].flatten(1).transpose(0, 1)
    return F.embedding(indices, flattened)


def _resize_for_optimization(rgb: Tensor, maximum: int) -> Tensor:
    height, width = rgb.shape[-2:]
    scale = min(1.0, maximum / max(height, width))
    target = (max(1, round(height * scale)), max(1, round(width * scale)))
    if target == (height, width):
        return rgb
    return F.interpolate(rgb, size=target, mode="bilinear", align_corners=False)
