"""Batch-safe execution shim for the frozen DINOv3 attention diagnostic.

The pinned upstream VIP attention implementation computes the returned
patch-attention diagnostic with ``q.squeeze(0)``.  Its actual SDPA output is
already batch-safe, but the diagnostic path raises for a batch of local
windows.  This opt-in shim keeps the original function for B=1 and uses the
same operations with a flattened ``B*H`` batch for B>1.  It changes no model
weights or attention equation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import inspect


def enable_batched_attention() -> None:
    """Make the pinned DINOv3 SelfAttention diagnostic batch-safe once."""

    from dinov3.layers.attention import SelfAttention

    if getattr(SelfAttention, "_geov2_batch_safe", False):
        return
    original = SelfAttention.compute_attention
    returns_diagnostic = "attn_weights" in inspect.getsource(SelfAttention.forward)

    @torch.inference_mode()
    def compute_attention(self, qkv, attn_bias=None, rope=None):
        if qkv.shape[0] == 1:
            return original(self, qkv, attn_bias=attn_bias, rope=rope)
        if attn_bias is not None:
            raise AssertionError("Geov2 batch shim preserves the pinned no-bias path")
        batch, tokens, _ = qkv.shape
        channels = self.qkv.in_features
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [value.transpose(1, 2) for value in (q, k, v)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        head_dim = q.shape[-1]
        flat_q = q.reshape(batch * self.num_heads, tokens, head_dim)
        flat_k = k.reshape(batch * self.num_heads, tokens, head_dim)
        scores = torch.bmm(flat_q * (head_dim ** -0.5), flat_k.transpose(1, 2))
        scores = scores.reshape(batch, self.num_heads, tokens, tokens)
        diagnostic = F.softmax(scores, dim=-1)[..., 5:, 5:].mean(dim=1)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        # DINO_Soars returns only the value tensor; the pinned VIP fork also
        # returns the patch-attention diagnostic.  Preserve the hosting class
        # contract so the shim works with either import order.
        return (output, diagnostic) if returns_diagnostic else output

    SelfAttention.compute_attention = compute_attention
    SelfAttention._geov2_batch_safe = True
