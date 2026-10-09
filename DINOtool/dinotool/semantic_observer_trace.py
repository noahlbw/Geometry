"""Matched frozen semantic observers; diagnostic outputs, not a correction rule."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def matched_vip_features(head, prepared, proxy_attention):
    if prepared.prefix_tokens != 5 or prepared.backbone_tokens.shape[0] != 1:
        raise ValueError("Pinned VIP proxy requires one image and five prefix tokens.")
    if prepared.grid_height != prepared.grid_width:
        raise ValueError("Pinned VIP proxy requires a square patch grid.")
    original = getattr(head, "patch_size", None)
    existed = hasattr(head, "patch_size")
    head.patch_size = prepared.grid_height
    try:
        tokens = prepared.backbone_tokens
        for block in head.blocks:
            attention = proxy_attention(head, block.attn, block.norm1(tokens),
                                        prepared.raw_patch_tokens)
            tokens = tokens + block.ls1(attention)
            tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
        projected = head.linear_projection(head.ln_final(tokens))[:, prepared.prefix_tokens:]
        return F.normalize(projected.float(), dim=-1)
    finally:
        if existed:
            head.patch_size = original
        else:
            del head.patch_size


def nonself_supported(scores, relation, valid):
    weights = relation.float().masked_fill(~valid[:, None], 0.)
    diagonal = torch.eye(weights.shape[-1], device=weights.device, dtype=torch.bool)[None]
    weights = weights.masked_fill(diagonal, 0.)
    mass = weights.sum(-1, keepdim=True)
    supported = (weights / mass.clamp_min(1e-12)) @ scores.float()
    return torch.where(mass > 0, supported, scores.float())
