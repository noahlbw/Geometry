"""Readout contrast on shared tokens; no absolute coarse posterior transfer."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .finite_vip_observer import finite_proxy_attention
from .geometry_semantic_innovation import anchored_innovation


IMPLEMENTATION = "geometry-matched-counterfactual-innovation-v4-20261001"


@torch.inference_mode()
def matched_proxy_features(head, prepared):
    if prepared.prefix_tokens != 5 or prepared.backbone_tokens.shape[0] != 1:
        raise ValueError("The matched proxy requires one image and five prefix tokens.")
    if prepared.grid_height != prepared.grid_width:
        raise ValueError("The matched proxy requires a square patch grid.")
    existed = hasattr(head, "patch_size")
    previous = getattr(head, "patch_size", None)
    head.patch_size = prepared.grid_height
    empty, rows = 0, 0
    try:
        tokens = prepared.backbone_tokens
        for block in head.blocks:
            attention, count, total = finite_proxy_attention(
                head, block.attn, block.norm1(tokens), prepared.raw_patch_tokens)
            empty += count
            rows += total
            tokens = tokens+block.ls1(attention)
            tokens = tokens+block.ls2(block.mlp(block.norm2(tokens)))
        projected = head.linear_projection(head.ln_final(tokens))[:, 5:]
        result = F.normalize(projected.float(), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite matched proxy features.")
        return result, {"empty_rows": empty, "observed_rows": rows}
    finally:
        if existed:
            head.patch_size = previous
        else:
            del head.patch_size


@torch.inference_mode()
def counterfactual_innovation(local, treated, control, relation, valid):
    """delta target = treated-control at identical view, text and precision.

    min_d .5||d||^2 + .5||A(d-(treated-control))||^2.
    Shared class-specific view/template offsets cancel before reconstruction.
    That algebraic cancellation does not establish semantic correctness.
    """
    if treated.shape != local.shape or control.shape != local.shape:
        raise ValueError("Matched treated/control scores must have the local score shape.")
    return anchored_innovation(local, local.float()+(treated.float()-control.float()), relation, valid)
