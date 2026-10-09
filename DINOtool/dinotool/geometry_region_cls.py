"""Read the frozen image-level semantic branch on Geometry-defined supports."""
import torch
import torch.nn.functional as F

from .tcpr import _finish_attention_block


IMPLEMENTATION = "geometry-region-cls-semantic-readout-v1-20261001"
METHODS = ("Geometry", "NativeGlobal", "RegionCLS", "RegionFull", "MeanLogitRegion", "AnchoredRegion")


def split_qkv(block, tokens):
    batch, count, _ = tokens.shape
    attention = block.attn
    qkv = attention.qkv(block.norm1(tokens))
    channels = qkv.shape[-1]//3
    parts = qkv.reshape(batch, count, 3, attention.num_heads, channels//attention.num_heads).unbind(2)
    return tuple(part.transpose(1, 2) for part in parts)


@torch.inference_mode()
def region_descriptors(head, prepared, valid):
    """Virtual CLS tokens do not enter the native patch or register stream.

    Each query retains its learned prefix/patch attention mass. Its conditional
    patch weights are replaced by the original Geometry support. Native donor
    states continue unchanged; the virtual CLS follows both frozen MLP blocks.
    """
    native = prepared.backbone_tokens
    prefix = prepared.prefix_tokens
    relation = prepared.geometry_patch_conditional.float()
    batch, patches, _ = relation.shape
    if (native is None or prefix < 1 or relation.shape != (batch, patches, patches)
            or native.shape[1] != patches+prefix or valid.shape != (batch, patches)):
        raise ValueError("Expected native tokens, prefix and matching region supports.")
    if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("Invalid Geometry relation.")
    relation = relation.masked_fill(~valid[:, None], 0.)
    mass = relation.sum(-1, keepdim=True)
    usable = valid & (mass[..., 0] > 0)
    relation = relation/mass.clamp_min(1e-12)
    virtual = native[:, :1].expand(-1, patches, -1).clone()
    mean_patch_mass = 0.
    for index, block in enumerate(head.blocks):
        _, native_key, native_value = split_qkv(block, native)
        query, self_key, self_value = split_qkv(block, virtual)
        logits_self = (query.float()*self_key.float()).sum(-1, keepdim=True)*block.attn.scale
        logits_reg = (query.float() @ native_key[:, :, 1:prefix].float().transpose(-1, -2))*block.attn.scale
        logits_patch = (query.float() @ native_key[:, :, prefix:].float().transpose(-1, -2))*block.attn.scale
        logits_patch = logits_patch.masked_fill(~valid[:, None, None], -torch.inf)
        weights = torch.cat((logits_self, logits_reg, logits_patch), -1).softmax(-1).to(native_value.dtype)
        patch_mass = weights[..., prefix:].sum(-1, keepdim=True)
        support_value = relation[:, None].to(native_value.dtype) @ native_value[:, :, prefix:]
        attended = (weights[..., :1]*self_value
                    + weights[..., 1:prefix] @ native_value[:, :, 1:prefix]
                    + patch_mass*support_value)
        virtual = _finish_attention_block(block, virtual, attended)
        mean_patch_mass += float(patch_mass.mean())
        if index < prepared.block_index:
            native = block(native)
        else:
            query, key, value = split_qkv(block, native)
            attention = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
            native = _finish_attention_block(block, native, attention @ value)
    native_projected = head.linear_projection(head.ln_final(native)).float()
    virtual_projected = head.linear_projection(head.ln_final(virtual)).float()
    pooled = relation @ native_projected[:, prefix:]
    full = F.normalize(torch.cat((virtual_projected, pooled), -1), dim=-1)
    global_full = F.normalize(torch.cat((native_projected[:, 0], native_projected[:, prefix:].mean(1)), -1), dim=-1)
    global_error = float((global_full-prepared.native_global).abs().max())
    if global_error != 0.:
        raise ValueError(f"Native global path is not exact: {global_error}")
    return {"RegionCLS": F.normalize(virtual_projected, dim=-1), "RegionFull": full,
            "NativeGlobal": global_full[:, None].expand(-1, patches, -1)}, usable, {
        "region_patch_attention_mass": mean_patch_mass/len(head.blocks),
        "region_usable_fraction": float(usable.float().mean()),
        "native_global_replay_max_error": global_error,
        "region_full_mean_cosine_vs_global": float((full*global_full[:, None]).sum(-1)[usable].mean()) if usable.any() else 0.,
    }
