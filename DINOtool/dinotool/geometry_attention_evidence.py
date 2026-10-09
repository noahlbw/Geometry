"""Read attention evidence separately from the unedited native semantic stream."""
import torch
import torch.nn.functional as F

from .tcpr import _geometry_attended


IMPLEMENTATION = "geometry-attention-evidence-v1-native-donors-20261001"
METHODS = ("Geometry", "Native", "EvidenceNative", "EvidenceGeometry",
           "MeanLogitEvidence", "AnchoredEvidence")


@torch.inference_mode()
def read_attention_evidence(head, prepared):
    """Sum projected attention increments; native residual/MLP only update donors.

    Unlike the frozen-path intervention, the evidence descriptor contains no
    explicit backbone residual or MLP increment. Learned normalization and
    projection remain unchanged; this is not a semantic reliability certificate.
    """
    native = prepared.backbone_tokens
    relation = prepared.geometry_patch_conditional
    prefix = prepared.prefix_tokens
    if (native is None or native.ndim != 3 or not 0 <= prefix < native.shape[1]
            or relation.shape != (native.shape[0], native.shape[1]-prefix,
                                  native.shape[1]-prefix)
            or prepared.block_index != len(head.blocks)-1):
        raise ValueError("Requires matching backbone tokens, relation and final head block.")
    if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("Geometry relation must be finite and nonnegative.")
    sums = {name: torch.zeros_like(native, dtype=torch.float32)
            for name in ("EvidenceNative", "EvidenceGeometry")}
    for index, block in enumerate(head.blocks):
        attention = block.attn
        qkv = attention.qkv(block.norm1(native))
        batch, count, triple = qkv.shape
        channels = triple//3
        query, key, value = qkv.reshape(batch, count, 3, attention.num_heads,
                                       channels//attention.num_heads).unbind(2)
        query, key, value = (part.transpose(1, 2) for part in (query, key, value))
        weights = ((query.float() @ key.float().transpose(-1, -2))*attention.scale).softmax(-1).to(value.dtype)
        readings = {"EvidenceNative": weights @ value,
                    "EvidenceGeometry": _geometry_attended(weights, relation.to(value.dtype), value, prefix)}
        increments = {}
        for name, reading in readings.items():
            merged = reading.transpose(1, 2).reshape(batch, count, channels)
            increments[name] = block.ls1(attention.proj_drop(attention.proj(merged)))
            sums[name] = sums[name]+increments[name].float()
        if index < prepared.block_index:
            native = block(native)
        else:
            attended = native+increments["EvidenceNative"]
            native = attended+block.ls2(block.mlp(block.norm2(attended)))

    def finish(state):
        projected = head.linear_projection(head.ln_final(state.to(native.dtype)))
        return F.normalize(projected[:, prefix:].float(), dim=-1)

    features = {name: finish(state) for name, state in sums.items()}
    features["Native"] = finish(native)
    if not all(bool(torch.isfinite(value).all()) for value in features.values()):
        raise ValueError("Nonfinite attention-evidence descriptor.")
    return features, {
        "mean_evidence_state_magnitude": float(sums["EvidenceGeometry"][:, prefix:].abs().mean()),
        "mean_geometry_native_evidence_cosine": float(
            (features["EvidenceGeometry"]*features["EvidenceNative"]).sum(-1).mean()),
        "mean_evidence_geometry_cosine": float(
            (features["EvidenceGeometry"]*prepared.geometry_projected).sum(-1).mean()),
    }
