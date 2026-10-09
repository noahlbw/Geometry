"""Locality controls and the published NACLIP reduced last-block operator."""
import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended, _intervened_head


IMPLEMENTATION = "geometry-matched-neighborhood-controls-v1-20261001"
METHODS = ("Geometry", "NACLIP_Last", "Native_Spatial", "SpatialOnly")
OPERATOR_NOTES = {
    "NACLIP_Last": "Author reduced KK attention plus additive Gaussian kernel, sigma5 patch units; last residual/MLP removed.",
    "Native_Spatial": "Native QK conditional patch relation plus Geometry's log-distance prior, native prefix/mass retained, both blocks.",
    "SpatialOnly": "Only Geometry's log-distance relation, without raw DINO feature similarity, both blocks with native prefix/mass.",
}


def spatial_logits(height, width, *, device):
    y, x = torch.meshgrid(torch.linspace(0., 1., height, device=device),
                          torch.linspace(0., 1., width, device=device), indexing="ij")
    coords = torch.stack((y, x), -1).reshape(-1, 2)
    return -((coords[:, None]-coords[None]).square().sum(-1))/(2*.25**2)


def run_neighborhood(head, prepared, method):
    if method not in METHODS:
        raise ValueError(method)
    prefix, target = prepared.prefix_tokens, prepared.block_index
    distance = spatial_logits(prepared.grid_height, prepared.grid_width,
                              device=prepared.backbone_tokens.device)
    if method in ("Geometry", "SpatialOnly"):
        relation = (prepared.geometry_patch_conditional if method == "Geometry" else
                    distance.softmax(-1)[None].to(prepared.backbone_tokens.dtype))
        projected = _intervened_head(head, prepared.backbone_tokens, relation, prefix, target, 2, "preserve")
        return F.normalize(projected[:, prefix:].float(), dim=-1)
    current = prepared.backbone_tokens
    for index, block in enumerate(head.blocks[:target+1]):
        if method == "NACLIP_Last" and index < target:
            current = block(current)
            continue
        query, key, value = qkv(block, current)
        if method == "NACLIP_Last":
            yy, xx = torch.meshgrid(torch.arange(prepared.grid_height, device=current.device),
                                   torch.arange(prepared.grid_width, device=current.device), indexing="ij")
            coords = torch.stack((yy, xx), -1).reshape(-1, 2).float()
            gaussian = torch.exp(-((coords[:, None]-coords[None]).square().sum(-1))/(2*5.**2))
            addition = F.pad(gaussian, (prefix, 0, prefix, 0))
            weights = ((key.float() @ key.float().transpose(-1, -2))*block.attn.scale+addition).softmax(-1)
            attended = weights.to(value.dtype) @ value
            merged = attended.transpose(1, 2).reshape(current.shape)
            current = block.attn.proj_drop(block.attn.proj(merged))
        else:
            logits = (query.float() @ key.float().transpose(-1, -2))*block.attn.scale
            native = logits.softmax(-1).to(value.dtype)
            conditional = (logits[..., prefix:, prefix:]+distance).softmax(-1).to(value.dtype)
            # Conditional relations here are head-specific; assemble without averaging heads.
            special = native[..., prefix:, :prefix] @ value[..., :prefix, :]
            mass = native[..., prefix:, prefix:].sum(-1, keepdim=True)
            patches = conditional @ value[..., prefix:, :]
            attended = torch.cat((native[..., :prefix, :] @ value, special+mass*patches), dim=2)
            current = _finish_attention_block(block, current, attended)
    projected = _finish_head(head, current, target)
    return F.normalize(projected[:, prefix:].float(), dim=-1)
