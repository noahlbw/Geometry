"""Frozen Geometry observations with alias-preserving evidence reconstruction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .prompts import REMOTE_SENSING_TEMPLATES
from .tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank


@dataclass(frozen=True)
class GearConfig:
    tile_size: int = 512
    alias_temperature: float = 0.07
    observation_temperature: float = 1.0
    geometry_temperature: float = 0.1
    text_ridge: float = 0.1
    support_size: int = 64
    support_radius: float = 2.5
    iterations: int = 6
    backtracks: int = 8
    stopping_tolerance: float = 1e-3
    huber_delta: float = 1.0

    def validate(self, patch_size: int) -> None:
        if self.tile_size != 512 or patch_size != 16:
            raise ValueError("GEAR-OV currently requires the locked 512 tile and 16 patch size.")
        if min(self.alias_temperature, self.observation_temperature,
               self.geometry_temperature, self.text_ridge, self.support_radius,
               self.huber_delta) <= 0:
            raise ValueError("GEAR-OV temperatures, ridge and radius must be positive.")
        if self.support_size < 16 or self.iterations < 1 or self.backtracks < 1:
            raise ValueError("GEAR-OV support size and optimization budgets are invalid.")
        if not 0 < self.stopping_tolerance < 1:
            raise ValueError("GEAR-OV stopping tolerance must be in (0,1).")


@dataclass(frozen=True)
class GearObservations:
    local_aligned: Tensor       # [32,32,D]
    detail_aligned: Tensor      # [64,64,D]
    context_aligned: Tensor     # [16,16,D], original-image center area
    detail_raw: Tensor          # [64,64,D]
    context_raw: Tensor         # [16,16,D]
    actual_height: int
    actual_width: int
    local_global: Tensor | None = None
    detail_global: Tensor | None = None
    context_global: Tensor | None = None


@dataclass(frozen=True)
class GearText:
    features: Tensor
    parents: Tensor
    basis: Tensor               # [M,r], alias changes induced by text span
    whitening: Tensor           # [M,M], fixed text-correlation metric
    class_count: int


@dataclass(frozen=True)
class GearOutput:
    local_logits: Tensor        # [1,C,512,512], original score units
    multiscale_logits: Tensor
    logits: Tensor
    diagnostics: dict[str, float | int]


def _crop_at(image: Tensor, top: int, left: int, extent: int) -> Tensor:
    """Replicate image boundaries, including a context crop outside the image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a [3,H,W] RGB image.")
    h, w = image.shape[-2:]
    ys = torch.arange(top, top + extent, device=image.device).clamp(0, h - 1)
    xs = torch.arange(left, left + extent, device=image.device).clamp(0, w - 1)
    return image.index_select(1, ys).index_select(2, xs).unsqueeze(0)


def _map(prepared, name: str) -> Tensor:
    value = getattr(prepared, name)[0]
    return value.reshape(prepared.grid_height, prepared.grid_width, -1)


def _four_to_grid(quadrants: list[Tensor]) -> Tensor:
    if len(quadrants) != 4:
        raise ValueError("Expected the four detail quadrants in row-major order.")
    return torch.cat((torch.cat(quadrants[:2], dim=1),
                      torch.cat(quadrants[2:], dim=1)), dim=0)


def _score_map(features: Tensor, text: Tensor, temperature: float) -> Tensor:
    return (features.float() @ text.T) / temperature


def center(scores: Tensor) -> Tensor:
    return scores - scores.mean(dim=-1, keepdim=True)


def class_scores(scores: Tensor, parents: Tensor, classes: int) -> Tensor:
    """Log-mean-exp class evidence in temperature-scaled alias units."""
    values = []
    for class_index in range(classes):
        subset = scores[..., parents == class_index]
        values.append(torch.logsumexp(subset, dim=-1) - math.log(subset.shape[-1]))
    return torch.stack(values, dim=-1)


def build_text_basis(bank: TCPRTextBank, ridge: float) -> GearText:
    bank.validate()
    text = F.normalize(bank.features.detach().float(), dim=-1)
    # P @ T removes the shared alias mode, not the feature-coordinate mean.
    centered = text - text.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = int((singular > singular.max() * 1e-5).sum().item())
    if rank == 0:
        raise ValueError("Encoded aliases contain no discriminative text direction.")
    basis = centered @ vh[:rank].T
    gram = centered @ centered.T
    values, vectors = torch.linalg.eigh(gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    ))
    whitening = (vectors * values.rsqrt()[None, :]) @ vectors.T
    return GearText(text, bank.parent_indices, basis, whitening, bank.class_count)


def build_support(detail_raw: Tensor, context_raw: Tensor,
                  valid_fine: Tensor, valid_context: Tensor,
                  config: GearConfig) -> tuple[Tensor, Tensor]:
    """Sparse cross-view support, retaining all 4x4 mapped fine positions."""
    fine_h, fine_w = detail_raw.shape[:2]
    context_h, context_w = context_raw.shape[:2]
    if (fine_h, fine_w) != (64, 64) or (context_h, context_w) != (16, 16):
        raise ValueError("GEAR-OV support expects 64x64 detail and 16x16 context grids.")
    device = detail_raw.device
    fy, fx = torch.meshgrid(torch.arange(64, device=device),
                            torch.arange(64, device=device), indexing="ij")
    cy, cx = torch.meshgrid(torch.arange(16, device=device),
                            torch.arange(16, device=device), indexing="ij")
    cy, cx = cy.flatten(), cx.flatten()
    fine_pos_y, fine_pos_x = (fy.flatten() + 0.5) * 8, (fx.flatten() + 0.5) * 8
    context_pos_y, context_pos_x = (cy + 0.5) * 32, (cx + 0.5) * 32
    dy = fine_pos_y[None, :] - context_pos_y[:, None]
    dx = fine_pos_x[None, :] - context_pos_x[:, None]
    radius = config.support_radius * 32
    allowed = ((dy.abs() <= radius) & (dx.abs() <= radius)
               & valid_fine.flatten()[None, :])
    similarity = (F.normalize(context_raw.reshape(256, -1).float(), dim=-1)
                  @ F.normalize(detail_raw.reshape(4096, -1).float(), dim=-1).T)
    score = similarity / config.geometry_temperature - (dx.square() + dy.square()) / (2 * 64**2)
    score = score.masked_fill(~allowed, -1e9)

    oy, ox = torch.meshgrid(torch.arange(4, device=device),
                            torch.arange(4, device=device), indexing="ij")
    mapped = ((cy[:, None] * 4 + oy.flatten()[None, :]) * 64
              + cx[:, None] * 4 + ox.flatten()[None, :])
    rest = score.clone()
    rest.scatter_(1, mapped, -1e9)
    additional = rest.topk(config.support_size - 16, dim=1).indices
    indices = torch.cat((mapped, additional), dim=1)
    selected = score.gather(1, indices)
    usable = allowed.gather(1, indices) & valid_context.flatten()[:, None]
    selected = selected.masked_fill(~usable, -1e9)
    weights = selected.softmax(dim=1) * usable.float()
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    return indices, weights


def observe_context(scores: Tensor, indices: Tensor, weights: Tensor,
                    temperature: float) -> Tensor:
    """Predict each context alias observation from its fine support."""
    selected = scores[indices]
    log_weights = weights.clamp_min(1e-12).log()
    return temperature * torch.logsumexp(
        log_weights[:, :, None] + selected / temperature, dim=1
    )


def _huber(residual: Tensor, threshold: float) -> Tensor:
    magnitude = residual.abs()
    return torch.where(magnitude <= threshold,
                       0.5 * residual.square(),
                       threshold * (magnitude - 0.5 * threshold)).mean()


def reconstruction_energy(update: Tensor, local: Tensor, detail: Tensor,
                          context: Tensor, basis: Tensor, whitening: Tensor,
                          indices: Tensor, weights: Tensor,
                          valid_fine: Tensor, valid_context: Tensor,
                          config: GearConfig) -> Tensor:
    reconstructed = local + update @ basis.T
    delta = (reconstructed - local) @ whitening
    anchor = 0.5 * delta[valid_fine].square().mean()
    detail_loss = _huber(((reconstructed - detail) @ whitening)[valid_fine],
                         config.huber_delta)
    prediction = center(observe_context(reconstructed, indices, weights,
                                        config.observation_temperature))
    context_loss = _huber(((prediction - context) @ whitening)[valid_context],
                          config.huber_delta)
    return anchor + detail_loss + context_loss


def solve_evidence(local: Tensor, detail: Tensor, context: Tensor,
                   text: GearText, indices: Tensor, weights: Tensor,
                   valid_fine: Tensor, valid_context: Tensor,
                   config: GearConfig) -> tuple[Tensor, dict[str, float | int]]:
    """Optimize image-local evidence only; all observations and weights are fixed."""
    if not bool(valid_fine.any()) or not bool(valid_context.any()):
        raise ValueError("At least one valid fine and context position is required.")
    basis = text.basis.detach().clone()
    whitening = text.whitening.detach().clone()
    local = local.detach().clone()
    detail = detail.detach().clone()
    context = context.detach().clone()
    indices = indices.detach().clone()
    weights = weights.detach().clone()
    update = torch.zeros(local.shape[0], basis.shape[1], device=local.device,
                         dtype=torch.float32)

    def energy(current: Tensor) -> Tensor:
        return reconstruction_energy(current, local, detail, context, basis,
                                     whitening, indices, weights, valid_fine,
                                     valid_context, config)

    with torch.no_grad():
        start = float(energy(update))
    previous = start
    accepted = 0
    for _ in range(config.iterations):
        with torch.enable_grad():
            variable = update.detach().requires_grad_(True)
            objective = energy(variable)
            gradient = torch.autograd.grad(objective, variable)[0]
        rms = gradient.square().mean().sqrt()
        if not torch.isfinite(rms) or float(rms) < 1e-12:
            break
        direction = -gradient / rms
        accepted_step = False
        with torch.no_grad():
            for trial in range(config.backtracks):
                proposal = update + direction * (0.5 ** trial)
                value = float(energy(proposal))
                if math.isfinite(value) and value < previous:
                    update = proposal
                    accepted += 1
                    accepted_step = True
                    break
        if not accepted_step:
            break
        relative = (previous - value) / max(abs(previous), 1e-12)
        previous = value
        if relative < config.stopping_tolerance:
            break
    reconstructed = local + update @ basis.T
    delta = reconstructed - local
    return reconstructed.detach(), {
        "iterations": accepted,
        "objective_initial": start,
        "objective_final": previous,
        "mean_abs_alias_delta": float(delta[valid_fine].abs().mean()),
        "negative_alias_delta_fraction": float((delta[valid_fine] < -1e-5).float().mean()),
    }


class GearOVSegmenter:
    def __init__(self, backbone, config: GearConfig = GearConfig()) -> None:
        config.validate(backbone.patch_size)
        self.config = config
        self.base = TCPRSegmenter(backbone, TCPRConfig(
            maximum_aliases_per_class=20, geometry_depth=2, prefix_policy="preserve",
            relation_policy="dense",
        ))

    @property
    def device(self) -> torch.device:
        return self.base.device

    def encode_text(self, specs) -> TCPRTextBank:
        return self.base.encode_text(specs)

    def encode_alias_groups(self, class_names, aliases_by_class) -> TCPRTextBank:
        """Encode exact query groups without inserting a class-name alias."""
        features, parents, canonical = self.base.backbone.encode_text_aliases(
            aliases_by_class, templates=REMOTE_SENSING_TEMPLATES
        )
        bank = TCPRTextBank(
            features, parents, canonical, tuple(class_names),
            tuple(alias for group in aliases_by_class for alias in group),
        )
        bank.validate()
        return bank

    def text_basis(self, bank: TCPRTextBank) -> GearText:
        return build_text_basis(bank, self.config.text_ridge)

    def prepare_views(self, image: Tensor, top: int, left: int) -> GearObservations:
        tile = self.config.tile_size
        h = min(tile, image.shape[-2] - top)
        w = min(tile, image.shape[-1] - left)
        with torch.inference_mode():
            local_rgb = _crop_at(image, top, left, tile).to(self.device)
            local = self.base.prepare_image(local_rgb)
            detail_rgb = F.interpolate(local_rgb, scale_factor=2, mode="bilinear",
                                       align_corners=False)
            detail_aligned = []
            detail_raw = []
            detail_global = []
            for dy in (0, tile):
                for dx in (0, tile):
                    prepared = self.base.prepare_image(detail_rgb[:, :, dy:dy + tile, dx:dx + tile])
                    detail_aligned.append(_map(prepared, "geometry_projected"))
                    detail_raw.append(_map(prepared, "raw_patch_tokens"))
                    detail_global.append(prepared.native_global[0])
                    del prepared
            context_rgb = _crop_at(image, top - tile // 2, left - tile // 2,
                                   2 * tile).to(self.device)
            context_rgb = F.interpolate(context_rgb, size=(tile, tile),
                                        mode="bilinear", align_corners=False)
            context = self.base.prepare_image(context_rgb)
            return GearObservations(
                _map(local, "geometry_projected"),
                _four_to_grid(detail_aligned),
                _map(context, "geometry_projected")[8:24, 8:24],
                _four_to_grid(detail_raw),
                _map(context, "raw_patch_tokens")[8:24, 8:24],
                h, w,
                local.native_global[0],
                torch.stack(detail_global),
                context.native_global[0],
            )

    def read_views(self, views: GearObservations, text: GearText,
                   *, reconstruct: bool = True) -> GearOutput:
        config = self.config
        local_cosine = (views.local_aligned.float() @ text.features.T)
        local_scaled = _score_map(views.local_aligned, text.features,
                                  config.alias_temperature)
        detail_scaled = _score_map(views.detail_aligned, text.features,
                                   config.alias_temperature)
        context_scaled = _score_map(views.context_aligned, text.features,
                                    config.alias_temperature)
        local_grid = F.interpolate(center(local_scaled).permute(2, 0, 1)[None],
                                   size=(64, 64), mode="bilinear",
                                   align_corners=False)[0].permute(1, 2, 0)
        local = local_grid.reshape(-1, text.features.shape[0])
        detail = center(detail_scaled).reshape(-1, text.features.shape[0])
        context = center(context_scaled).reshape(-1, text.features.shape[0])
        fine_y = torch.arange(64, device=self.device) * 8 < views.actual_height
        fine_x = torch.arange(64, device=self.device) * 8 < views.actual_width
        valid_fine = (fine_y[:, None] & fine_x[None, :]).flatten()
        context_y = torch.arange(16, device=self.device) * 32 < views.actual_height
        context_x = torch.arange(16, device=self.device) * 32 < views.actual_width
        valid_context = (context_y[:, None] & context_x[None, :]).flatten()

        indices, weights = build_support(views.detail_raw, views.context_raw,
                                         valid_fine.reshape(64, 64),
                                         valid_context.reshape(16, 16), config)
        local_class = class_scores(local_cosine / config.alias_temperature,
                                   text.parents, text.class_count)
        local_logits = F.interpolate(
            (config.alias_temperature * local_class).permute(2, 0, 1)[None],
            size=(512, 512), mode="bilinear", align_corners=False,
        )
        detail_class = class_scores(detail_scaled, text.parents, text.class_count)
        context_class = class_scores(context_scaled, text.parents, text.class_count)
        detail_logits = F.interpolate(
            (config.alias_temperature * detail_class).permute(2, 0, 1)[None],
            size=(512, 512), mode="bilinear", align_corners=False,
        )
        context_logits = F.interpolate(
            (config.alias_temperature * context_class).permute(2, 0, 1)[None],
            size=(512, 512), mode="bilinear", align_corners=False,
        )
        multiscale = (local_logits + detail_logits + context_logits) / 3
        if reconstruct:
            with torch.inference_mode(False):
                reconstructed, diagnostics = solve_evidence(
                    local, detail, context, text, indices, weights,
                    valid_fine, valid_context, config,
                )
        else:
            reconstructed = local
            diagnostics = {"iterations": 0, "objective_initial": 0.0,
                           "objective_final": 0.0, "mean_abs_alias_delta": 0.0,
                           "negative_alias_delta_fraction": 0.0}
        difference = (class_scores(reconstructed, text.parents, text.class_count)
                      - class_scores(local, text.parents, text.class_count))
        fine_delta = difference.reshape(64, 64, text.class_count).permute(2, 0, 1)[None]
        dense_delta = F.interpolate(fine_delta, size=(512, 512), mode="bilinear",
                                    align_corners=False)
        final = local_logits + config.alias_temperature * dense_delta
        diagnostics = {**diagnostics,
                       "support_edges": int((weights > 0).sum().item()),
                       "valid_fine": int(valid_fine.sum().item()),
                       "valid_context": int(valid_context.sum().item())}
        return GearOutput(local_logits, multiscale, final, diagnostics)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
