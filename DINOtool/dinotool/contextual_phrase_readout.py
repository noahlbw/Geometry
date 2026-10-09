"""Training-free contextual phrase readout for frozen DINO.text.

The local Geometry route remains the localization anchor. A single resized
whole-image observation supplies phrase-level context. Local phrase responses
select which contextual aliases are relevant; the final correction is the
residual over an unconditioned context readout, so broad scene priors do not get
added directly to every local score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .hypothesis_readout import uniform_subset_scores
from .tcpr import (
    TCPRConfig,
    TCPRPreparedImage,
    TCPRSegmenter,
    TCPRTextBank,
    _normalize_valid_mask,
    _tokens_to_map,
)


@dataclass(frozen=True)
class ContextualPhraseConfig:
    alias_temperature: float = 0.07
    overview_size: int = 512
    residual_strength: float = 1.0
    epsilon: float = 1e-6

    def validate(self, patch_size: int = 16) -> None:
        if self.alias_temperature <= 0 or self.epsilon <= 0:
            raise ValueError("Temperatures and epsilon must be positive.")
        if self.overview_size < patch_size or self.overview_size % patch_size:
            raise ValueError("overview_size must be a positive multiple of patch_size.")
        if self.residual_strength < 0:
            raise ValueError("residual_strength must be non-negative.")


@dataclass(frozen=True)
class ContextualPhraseDiagnostics:
    mean_direct_residual: float
    mean_structured_residual: float
    mean_geometry_overview_disagreement: float
    changed_phrase_from_geometry: float
    changed_overview_fusion_from_geometry: float
    changed_contextual_from_geometry: float
    changed_class_union_from_geometry: float
    changed_alias_union_from_geometry: float
    changed_structured_alias_union_from_geometry: float

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ContextualPhraseResult:
    geometry_logits: Tensor
    phrase_residual_logits: Tensor
    overview_fusion_logits: Tensor
    contextual_phrase_logits: Tensor
    class_scale_union_logits: Tensor
    alias_scale_union_logits: Tensor
    structured_alias_union_logits: Tensor
    diagnostics: ContextualPhraseDiagnostics


class ContextualPhraseSegmenter:
    """Geometry-localized OVSS with phrase-conditioned whole-image context."""

    method_name = "Contextual Phrase Readout"

    def __init__(
        self,
        backbone,
        config: ContextualPhraseConfig = ContextualPhraseConfig(),
        tcpr_config: TCPRConfig = TCPRConfig(maximum_aliases_per_class=20),
    ) -> None:
        config.validate(backbone.patch_size)
        self.backbone = backbone
        self.config = config
        self.base = TCPRSegmenter(backbone, tcpr_config)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def encode_text(self, classes, batch_size: int = 64) -> TCPRTextBank:
        return self.base.encode_text(classes, batch_size=batch_size)

    def prepare_image(self, rgb: Tensor) -> TCPRPreparedImage:
        return self.base.prepare_image(rgb)

    def prepare_overview(self, rgb: Tensor) -> TCPRPreparedImage:
        resized = F.interpolate(
            rgb,
            size=(self.config.overview_size, self.config.overview_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.prepare_image(resized)

    def overview_alias_map(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
    ) -> Tensor:
        text_bank.validate()
        text = F.normalize(text_bank.features.float(), dim=-1)
        aliases = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
        return aliases.transpose(1, 2).reshape(
            aliases.shape[0], aliases.shape[2], prepared.grid_height, prepared.grid_width
        )

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        overview_aliases: Tensor,
        text_bank: TCPRTextBank,
        *,
        image_height: int,
        image_width: int,
        tile_top: int,
        tile_left: int,
        valid_mask: Tensor | None = None,
    ) -> ContextualPhraseResult:
        text_bank.validate()
        batch, patches, _ = prepared.geometry_projected.shape
        if batch != 1 or overview_aliases.shape[0] != 1:
            raise ValueError("Contextual phrase evaluation currently expects one image per batch.")
        if overview_aliases.shape[1] != text_bank.features.shape[0]:
            raise ValueError("Overview alias channels do not match the text bank.")
        valid = _normalize_valid_mask(
            valid_mask,
            batch,
            prepared.grid_height,
            prepared.grid_width,
            prepared.geometry_projected.device,
        ).reshape(batch, patches)
        text = F.normalize(text_bank.features.float(), dim=-1)
        local_aliases = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
        direct_context = sample_overview_aliases(
            overview_aliases,
            image_height=image_height,
            image_width=image_width,
            tile_top=tile_top,
            tile_left=tile_left,
            grid_height=prepared.grid_height,
            grid_width=prepared.grid_width,
            patch_size=self.patch_size,
        )
        geometry = masked_geometry(
            prepared.geometry_patch_conditional.float(), valid, self.config.epsilon
        )
        structured_context = geometry @ direct_context

        all_aliases = torch.ones(
            text_bank.features.shape[0], device=text_bank.features.device, dtype=torch.bool
        )
        geometry_scores = uniform_subset_scores(
            local_aliases,
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        direct_scores = uniform_subset_scores(
            direct_context,
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        structured_scores = uniform_subset_scores(
            structured_context,
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        direct_conditioned = conditioned_alias_scores(
            local_aliases, direct_context, text_bank, self.config.alias_temperature
        )
        structured_conditioned = conditioned_alias_scores(
            local_aliases, structured_context, text_bank, self.config.alias_temperature
        )

        strength = self.config.residual_strength
        phrase_residual = geometry_scores + strength * (direct_conditioned - direct_scores)
        overview_fusion = 0.5 * (geometry_scores + direct_scores)
        contextual = geometry_scores + strength * (structured_conditioned - structured_scores)
        class_scale_union = normalized_logmeanexp(
            torch.stack((geometry_scores, direct_scores), dim=-1),
            self.config.alias_temperature,
        )
        alias_scale_union = uniform_subset_scores(
            normalized_logmeanexp(
                torch.stack((local_aliases, direct_context), dim=-1),
                self.config.alias_temperature,
            ),
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        structured_alias_union = uniform_subset_scores(
            normalized_logmeanexp(
                torch.stack((local_aliases, structured_context), dim=-1),
                self.config.alias_temperature,
            ),
            text_bank.parent_indices,
            text_bank.class_count,
            all_aliases,
            self.config.alias_temperature,
        )
        phrase_residual = torch.where(valid[..., None], phrase_residual, geometry_scores)
        overview_fusion = torch.where(valid[..., None], overview_fusion, geometry_scores)
        contextual = torch.where(valid[..., None], contextual, geometry_scores)
        class_scale_union = torch.where(valid[..., None], class_scale_union, geometry_scores)
        alias_scale_union = torch.where(valid[..., None], alias_scale_union, geometry_scores)
        structured_alias_union = torch.where(
            valid[..., None], structured_alias_union, geometry_scores
        )

        geometry_label = geometry_scores.argmax(-1)
        direct_label = direct_scores.argmax(-1)
        count = max(int(valid.sum().item()), 1)

        def changed(scores: Tensor) -> float:
            return float(((scores.argmax(-1) != geometry_label) & valid).sum().item() / count)

        diagnostics = ContextualPhraseDiagnostics(
            mean_direct_residual=_valid_mean((direct_conditioned - direct_scores).abs(), valid),
            mean_structured_residual=_valid_mean(
                (structured_conditioned - structured_scores).abs(), valid
            ),
            mean_geometry_overview_disagreement=float(
                ((geometry_label != direct_label) & valid).sum().item() / count
            ),
            changed_phrase_from_geometry=changed(phrase_residual),
            changed_overview_fusion_from_geometry=changed(overview_fusion),
            changed_contextual_from_geometry=changed(contextual),
            changed_class_union_from_geometry=changed(class_scale_union),
            changed_alias_union_from_geometry=changed(alias_scale_union),
            changed_structured_alias_union_from_geometry=changed(structured_alias_union),
        )
        height, width = prepared.grid_height, prepared.grid_width
        return ContextualPhraseResult(
            geometry_logits=_tokens_to_map(geometry_scores, height, width),
            phrase_residual_logits=_tokens_to_map(phrase_residual, height, width),
            overview_fusion_logits=_tokens_to_map(overview_fusion, height, width),
            contextual_phrase_logits=_tokens_to_map(contextual, height, width),
            class_scale_union_logits=_tokens_to_map(class_scale_union, height, width),
            alias_scale_union_logits=_tokens_to_map(alias_scale_union, height, width),
            structured_alias_union_logits=_tokens_to_map(structured_alias_union, height, width),
            diagnostics=diagnostics,
        )


def conditioned_alias_scores(
    local_aliases: Tensor,
    context_aliases: Tensor,
    text_bank: TCPRTextBank,
    temperature: float,
) -> Tensor:
    """Read context aliases using the local alias posterior of each class."""
    if local_aliases.shape != context_aliases.shape or local_aliases.ndim != 3:
        raise ValueError("Alias tensors must share shape [B,N,M].")
    output: list[Tensor] = []
    tiny = torch.finfo(context_aliases.dtype).tiny
    for class_index in range(text_bank.class_count):
        members = text_bank.parent_indices == class_index
        local = local_aliases[..., members]
        context = context_aliases[..., members]
        weights = torch.softmax(local / temperature, dim=-1)
        output.append(
            temperature * torch.logsumexp(
                context / temperature + weights.clamp_min(tiny).log(), dim=-1
            )
        )
    return torch.stack(output, dim=-1)


def normalized_logmeanexp(values: Tensor, temperature: float) -> Tensor:
    """A count-normalized soft evidence union over the final axis."""
    if values.shape[-1] < 1 or temperature <= 0:
        raise ValueError("Evidence union requires values and a positive temperature.")
    count = values.shape[-1]
    return temperature * (
        torch.logsumexp(values / temperature, dim=-1)
        - torch.log(values.new_tensor(float(count)))
    )


def sample_overview_aliases(
    overview_aliases: Tensor,
    *,
    image_height: int,
    image_width: int,
    tile_top: int,
    tile_left: int,
    grid_height: int,
    grid_width: int,
    patch_size: int,
) -> Tensor:
    """Sample overview phrase maps at local patch centers in original coordinates."""
    if overview_aliases.ndim != 4 or overview_aliases.shape[0] != 1:
        raise ValueError("overview_aliases must have shape [1,M,H,W].")
    if image_height < 1 or image_width < 1 or min(grid_height, grid_width, patch_size) < 1:
        raise ValueError("Image and grid dimensions must be positive.")
    device = overview_aliases.device
    dtype = overview_aliases.dtype
    y = tile_top + (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) * patch_size
    x = tile_left + (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) * patch_size
    y = 2.0 * y / float(image_height) - 1.0
    x = 2.0 * x / float(image_width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        overview_aliases,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.flatten(2).transpose(1, 2)


def masked_geometry(geometry: Tensor, valid: Tensor, epsilon: float) -> Tensor:
    """Remove padded keys and retain a row-stochastic local structure readout."""
    if geometry.ndim != 3 or valid.shape != geometry.shape[:2] or geometry.shape[1] != geometry.shape[2]:
        raise ValueError("Expected geometry [B,N,N] and valid [B,N].")
    masked = geometry.masked_fill(~valid[:, None, :], 0.0)
    denominator = masked.sum(-1, keepdim=True)
    identity = torch.eye(geometry.shape[-1], device=geometry.device, dtype=geometry.dtype)[None]
    masked = torch.where(denominator > epsilon, masked / denominator.clamp_min(epsilon), identity)
    return torch.where(valid[:, :, None], masked, identity)


def _valid_mean(values: Tensor, valid: Tensor) -> float:
    expanded = valid
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    return float(values[expanded].mean().item()) if bool(expanded.any()) else 0.0
