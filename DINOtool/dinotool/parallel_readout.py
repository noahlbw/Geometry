"""Training-free parallel relations inside a frozen DINO.text vision head.

The module deliberately changes only one attention block. It keeps the
checkpoint's Q/K/V projections, output projection, residual path, MLP, and
special-token attention intact. Only the conditional distribution from patch
queries to patch keys can be replaced or jointly read with a DINO structural
relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


READOUT_MODES = ("native", "geometry", "mixture", "temperature", "joint")


@dataclass(frozen=True)
class ParallelReadoutConfig:
    """A single training-free relation-reading arm."""

    mode: str
    beta: float = 0.5
    geometry_temperature: float = 0.1
    spatial_sigma: float = 0.25
    head_block: int = -1

    def validate(self) -> None:
        if self.mode not in READOUT_MODES:
            raise ValueError(f"mode must be one of {READOUT_MODES}, got {self.mode!r}.")
        if self.beta < 0:
            raise ValueError("beta must be non-negative.")
        if self.geometry_temperature <= 0:
            raise ValueError("geometry_temperature must be positive.")
        if self.spatial_sigma <= 0:
            raise ValueError("spatial_sigma must be positive.")


def structural_logits(
    backbone_patch_tokens: Tensor,
    grid_height: int,
    grid_width: int,
    *,
    temperature: float,
    spatial_sigma: float,
) -> Tensor:
    """Return DINO feature/position structural logits on the patch grid."""
    if backbone_patch_tokens.ndim != 3:
        raise ValueError("backbone_patch_tokens must have shape [B,N,D].")
    batch, patches, _ = backbone_patch_tokens.shape
    if patches != grid_height * grid_width:
        raise ValueError("grid dimensions must account for every patch token.")
    if temperature <= 0 or spatial_sigma <= 0:
        raise ValueError("temperature and spatial_sigma must be positive.")

    features = torch.nn.functional.normalize(backbone_patch_tokens.float(), dim=-1)
    similarity = features @ features.transpose(-1, -2)
    y = torch.linspace(0.0, 1.0, grid_height, device=features.device, dtype=features.dtype)
    x = torch.linspace(0.0, 1.0, grid_width, device=features.device, dtype=features.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((yy, xx), dim=-1).reshape(patches, 2)
    distance_squared = (coordinates[:, None] - coordinates[None, :]).square().sum(dim=-1)
    geometry = similarity / temperature - distance_squared / (2.0 * spatial_sigma * spatial_sigma)
    return geometry.expand(batch, -1, -1)


def patch_relation_weights(
    native_attention: Tensor,
    native_logits: Tensor,
    geometry: Tensor,
    prefix_tokens: int,
    config: ParallelReadoutConfig,
) -> Tensor:
    """Replace only patch-to-patch conditionals while preserving row mass."""
    config.validate()
    if native_attention.ndim != 4 or native_logits.shape != native_attention.shape:
        raise ValueError("native_attention and native_logits must both have shape [B,H,N,N].")
    _, _, tokens, _ = native_attention.shape
    patches = tokens - prefix_tokens
    if prefix_tokens < 0 or patches < 1 or geometry.shape != (native_attention.shape[0], patches, patches):
        raise ValueError("Invalid prefix_tokens or geometry shape.")
    if config.mode == "native" or (config.mode == "joint" and config.beta == 0.0):
        return native_attention

    patch_logits = native_logits[..., prefix_tokens:, prefix_tokens:]
    native_patch = native_attention[..., prefix_tokens:, prefix_tokens:]
    patch_mass = native_patch.sum(dim=-1, keepdim=True)
    geometry = geometry[:, None].to(dtype=patch_logits.dtype)

    if config.mode == "geometry":
        conditional = torch.softmax(geometry, dim=-1)
    elif config.mode == "mixture":
        native_conditional = torch.softmax(patch_logits, dim=-1)
        geometry_conditional = torch.softmax(geometry, dim=-1)
        geometry_weight = config.beta / (1.0 + config.beta)
        conditional = (1.0 - geometry_weight) * native_conditional + geometry_weight * geometry_conditional
    elif config.mode == "temperature":
        conditional = torch.softmax(patch_logits / (1.0 + config.beta), dim=-1)
    elif config.mode == "joint":
        conditional = torch.softmax((patch_logits + config.beta * geometry) / (1.0 + config.beta), dim=-1)
    else:
        raise AssertionError(f"Unhandled readout mode {config.mode!r}.")

    output = native_attention.clone()
    output[..., prefix_tokens:, prefix_tokens:] = patch_mass * conditional.to(dtype=native_attention.dtype)
    return output


def parallel_vision_head_readouts(
    head: nn.Module,
    image_tokens: Tensor,
    backbone_patch_tokens: Tensor,
    grid_height: int,
    grid_width: int,
    configs: Mapping[str, ParallelReadoutConfig],
) -> dict[str, Tensor]:
    """Run frozen head readouts that differ only at one attention block."""
    if not configs:
        raise ValueError("At least one readout config is required.")
    if image_tokens.ndim != 3:
        raise ValueError("image_tokens must have shape [B,N,D].")
    if image_tokens.shape[0] != backbone_patch_tokens.shape[0]:
        raise ValueError("image_tokens and backbone_patch_tokens must share a batch dimension.")
    if image_tokens.shape[1] - backbone_patch_tokens.shape[1] < 1:
        raise ValueError("image_tokens must include at least the class token before patch tokens.")
    resolved = []
    for config in configs.values():
        config.validate()
        block_index = config.head_block if config.head_block >= 0 else len(head.blocks) + config.head_block
        if block_index < 0 or block_index >= len(head.blocks):
            raise ValueError(f"head_block {config.head_block} is outside the vision head.")
        resolved.append(block_index)
    if len(set(resolved)) != 1:
        raise ValueError("All readout arms must use the same head_block for a fair shared-state comparison.")
    block_index = resolved[0]
    prefix_tokens = image_tokens.shape[1] - backbone_patch_tokens.shape[1]

    shared = image_tokens
    for block in head.blocks[:block_index]:
        shared = block(shared)

    results: dict[str, Tensor] = {}
    target_block = head.blocks[block_index]
    for name, config in configs.items():
        if config.mode == "native" or (config.mode == "joint" and config.beta == 0.0):
            current = target_block(shared)
        else:
            geometry = structural_logits(
                backbone_patch_tokens,
                grid_height,
                grid_width,
                temperature=config.geometry_temperature,
                spatial_sigma=config.spatial_sigma,
            )
            current = _parallel_self_attention_block(target_block, shared, geometry, prefix_tokens, config)
        for block in head.blocks[block_index + 1 :]:
            current = block(current)
        current = head.ln_final(current)
        results[name] = head.linear_projection(current)
    return results


def _parallel_self_attention_block(
    block: nn.Module,
    x: Tensor,
    geometry: Tensor,
    prefix_tokens: int,
    config: ParallelReadoutConfig,
) -> Tensor:
    """Equivalent to a DINO SelfAttentionBlock except patch relations."""
    normalized = block.norm1(x)
    attention = block.attn
    qkv = attention.qkv(normalized)
    batch, tokens, _ = qkv.shape
    channels = attention.qkv.in_features
    qkv = qkv.reshape(batch, tokens, 3, attention.num_heads, channels // attention.num_heads)
    query, key, value = torch.unbind(qkv, dim=2)
    query, key, value = (part.transpose(1, 2) for part in (query, key, value))
    logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
    native_attention = torch.softmax(logits, dim=-1).to(dtype=value.dtype)
    relation = patch_relation_weights(native_attention, logits, geometry, prefix_tokens, config)
    attended = relation @ value
    attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
    residual = attention.proj_drop(attention.proj(attended))
    x_attn = x + block.ls1(residual)
    return x_attn + block.ls2(block.mlp(block.norm2(x_attn)))
