"""Published readout operators adapted to the same frozen DINO.text head.

These are mechanism controls, not full official CLIP method reproductions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head, _intervened_head


IMPLEMENTATION = "geometry-matched-readout-controls-v1-20261001"
METHODS = ("Geometry", "Native", "SCLIP_Last", "SCLIP_Two", "ProxyCLIP_Last",
           "VIPProxy_Two", "Geometry_NoSpatial", "Geometry_BlockPrefix")
OPERATOR_NOTES = {
    "SCLIP_Last": "Author CSA: softmax(QQ*scale)+softmax(KK*scale), last block, residual/MLP retained.",
    "SCLIP_Two": "Same CSA operator in both DINO.text head blocks; a depth-matched adaptation.",
    "ProxyCLIP_Last": "Author beta1.2/gamma3.0 masked cosine proxy; final attention-only output, no last residual/MLP.",
    "VIPProxy_Two": "Pinned VIP beta1.5/gamma0.5 proxy in both blocks, including author prefix increment; empty rows read self.",
    "Geometry_NoSpatial": "Original conditional replacement and native mass, raw cosine/.10 without distance prior.",
    "Geometry_BlockPrefix": "Original relation and native prefix queries, patch queries read unit-mass patches only.",
}


def qkv(block, current):
    encoded = block.attn.qkv(block.norm1(current))
    batch, count, channels3 = encoded.shape
    heads = block.attn.num_heads
    return tuple(part.transpose(1, 2) for part in
                 encoded.reshape(batch, count, 3, heads, channels3//(3*heads)).unbind(2))


def proxy_relation(raw, beta, gamma, *, strict_zero):
    normalized = F.normalize(raw.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    logits = (similarity-similarity.mean()*beta)*gamma
    active = logits > 0 if strict_zero else logits >= 0
    empty = ~active.any(-1)
    diagonal = torch.eye(raw.shape[1], device=raw.device, dtype=torch.bool)[None]
    masked = logits.masked_fill(~active, -torch.inf)
    masked = masked.masked_fill(empty[..., None] & diagonal, 0.)
    return masked, int(empty.sum()), empty.numel()


def run_head(head, tokens, raw, geometry, prefix, method, target_block=1):
    if method not in METHODS:
        raise ValueError(f"Unknown matched method: {method}")
    diagnostics = {"proxy_empty_rows": 0., "proxy_rows": 0.}
    if method in ("Geometry", "Geometry_NoSpatial", "Geometry_BlockPrefix"):
        relation = geometry
        if method == "Geometry_NoSpatial":
            normalized = F.normalize(raw.float(), dim=-1)
            relation = ((normalized @ normalized.transpose(-1, -2))/.10).softmax(-1).to(geometry.dtype)
        projected = _intervened_head(head, tokens, relation, prefix, target_block, 2,
                                    "block" if method == "Geometry_BlockPrefix" else "preserve")
        return F.normalize(projected[:, prefix:].float(), dim=-1), diagnostics
    current = tokens
    first = target_block if method in ("SCLIP_Last", "ProxyCLIP_Last") else 0
    proxy = None
    if method in ("ProxyCLIP_Last", "VIPProxy_Two"):
        beta, gamma = (1.2, 3.) if method == "ProxyCLIP_Last" else (1.5, .5)
        proxy, empty, rows = proxy_relation(raw, beta, gamma, strict_zero=method == "VIPProxy_Two")
        diagnostics.update(proxy_empty_rows=float(empty), proxy_rows=float(rows))
    for index, block in enumerate(head.blocks[:target_block+1]):
        if index < first or (method == "Native" and index < target_block):
            current = block(current)
            continue
        query, key, value = qkv(block, current)
        if method == "Native":
            weights = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1)
            current = _finish_attention_block(block, current, weights.to(value.dtype) @ value)
        elif method.startswith("SCLIP"):
            scale = block.attn.scale
            weights = ((query.float() @ query.float().transpose(-1, -2))*scale).softmax(-1)
            weights += ((key.float() @ key.float().transpose(-1, -2))*scale).softmax(-1)
            current = _finish_attention_block(block, current, weights.to(value.dtype) @ value)
        elif method == "ProxyCLIP_Last":
            weights = proxy.to(query.dtype).softmax(-1)[:, None]
            attended = weights @ value[..., prefix:, :]
            merged = attended.transpose(1, 2).reshape(tokens.shape[0], raw.shape[1], -1)
            current = block.attn.proj_drop(block.attn.proj(merged))
            projected = _finish_head(head, current, target_block)
            return F.normalize(projected.float(), dim=-1), diagnostics
        else:
            # VIP returns normalized prefix inputs as an attention increment.
            weights = proxy.to(query.dtype).softmax(-1)[:, None]
            attended = weights @ value[..., prefix:, :]
            merged = attended.transpose(1, 2).reshape(tokens.shape[0], raw.shape[1], -1)
            increment = torch.cat((block.norm1(current)[:, :prefix],
                                   block.attn.proj_drop(block.attn.proj(merged))), dim=1)
            after_attention = current+block.ls1(increment)
            current = after_attention+block.ls2(block.mlp(block.norm2(after_attention)))
    projected = _finish_head(head, current, target_block)
    return F.normalize(projected[:, prefix:].float(), dim=-1), diagnostics


@torch.inference_mode()
def encode_single(backbone, rgb, method):
    """One deployed arm: one backbone call and only the requested head path."""
    backbone._validate_rgb(rgb)
    normalized = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
    with backbone._autocast():
        visual = backbone.model.visual_model
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        relation = None
        if method.startswith("Geometry"):
            relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                         temperature=.10, spatial_sigma=.25).softmax(-1).to(raw.dtype)
        features, diagnostics = run_head(visual.head, tokens, raw, relation, registers.shape[1]+1,
                                         method, len(visual.head.blocks)-1)
    return features, diagnostics
