"""Geometry relation transport without feedback into the frozen semantic path."""
import torch
import torch.nn.functional as F

from .tcpr import _geometry_attended


IMPLEMENTATION = "geometry-frozen-semantic-path-v2-exact-native-20261001"
METHODS = ("Geometry", "Native", "PostHeadGeometry", "FrozenPathGeometry")


@torch.inference_mode()
def read_frozen_path(head, prepared):
    """Accumulate relation interventions at native states before final alignment.

    n[l+1] = native_block(n[l]); d[l+1] = d[l] + G_att(n[l]) - N_att(n[l]).
    Only the readout n[L]+d[L] changes. Q/K/V and MLP always see native states.
    Prefix queries retain their native path. No network parameter is changed.
    """
    native = prepared.backbone_tokens
    relation = prepared.geometry_patch_conditional
    prefix = prepared.prefix_tokens
    if native is None or relation.shape != (native.shape[0], native.shape[1]-prefix,
                                            native.shape[1]-prefix):
        raise ValueError("Matching backbone tokens and Geometry relation are required.")
    if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("Geometry relations must be finite and nonnegative.")
    delta = torch.zeros_like(native, dtype=torch.float32)
    for index, block in enumerate(head.blocks):
        attention = block.attn
        qkv = attention.qkv(block.norm1(native))
        batch, count, triple = qkv.shape
        channels = triple//3
        query, key, value = qkv.reshape(batch, count, 3, attention.num_heads,
                                       channels//attention.num_heads).unbind(2)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        weights = ((query.float() @ key.float().transpose(-1, -2))*attention.scale).softmax(-1).to(value.dtype)
        native_read = weights @ value
        geometry_read = _geometry_attended(weights, relation.to(value.dtype), value, prefix)

        def project(read):
            merged = read.transpose(1, 2).reshape(batch, count, channels)
            return block.ls1(attention.proj_drop(attention.proj(merged)))

        native_increment, geometry_increment = project(native_read), project(geometry_read)
        delta = delta+geometry_increment.float()-native_increment.float()
        if index < prepared.block_index:
            # Match prepare_image's original fused native blocks exactly.
            native = block(native)
        else:
            attended = native+native_increment
            native = attended+block.ls2(block.mlp(block.norm2(attended)))
    transported = (native.float()+delta).to(native.dtype)
    projected = head.linear_projection(head.ln_final(transported))
    original = head.linear_projection(head.ln_final(native))
    native_features = F.normalize(original[:, prefix:].float(), dim=-1)
    features = F.normalize(projected[:, prefix:].float(), dim=-1)
    posthead = F.normalize(relation.float() @ native_features, dim=-1)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite frozen-path readout.")
    return {"Native": native_features, "PostHeadGeometry": posthead,
            "FrozenPathGeometry": features}, {
        "mean_abs_state_transport": float(delta[:, prefix:].abs().mean()),
        "prefix_transport_max": float(delta[:, :prefix].abs().max()) if prefix else 0.,
        "mean_cosine_vs_geometry": float((features*prepared.geometry_projected).sum(-1).mean()),
    }
