"""Isolate relation, attention strength and key-group allocation in Geometry."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head


IMPLEMENTATION = "geometry-csa-allocation-factorial-v1-20261005"
PRIMARY = "Geometry_CSAAllocationSum"
BASELINES = ("Geometry", "Geometry_BlockPrefix", "SCLIP_Two", "VIPProxy_Two")
FACTORS = {
    "Geometry_DoubleRow": ("native", 2.),
    "CSA_NativePrefixSum": ("csa", 1.),
    "CSA_NativePrefixMean": ("csa", .5),
    "Geometry_CSAAllocationMean": ("allocation", .5),
    PRIMARY: ("allocation", 1.),
    "Geometry_CSAMassMean": ("mass", .5),
    "Geometry_CSAMassSum": ("mass", 1.),
    "Geometry_CSAConditional": ("conditional", 1.),
}
METHODS = (*BASELINES, *FACTORS)


def settings():
    return {"implementation": IMPLEMENTATION, "primary": PRIMARY,
        "primary_rule": "CSA_special @ V_special + sum(CSA_patch) * G @ V_patch; native prefix queries",
        "csa": "author sum softmax(QQ*scale)+softmax(KK*scale), all keys, separately at each evolving block",
        "source_attribution": "CSA is borrowed SCLIP; this is a mechanism study, not an original CSA claim",
        "geometry_relation": "unchanged raw cosine/.10 and spatial sigma .25 relation in both blocks",
        "factors": {name: list(value) for name, value in FACTORS.items()},
        "double_control": "double original patch-query attention read before projection; prefix-query reads stay native",
        "mean_controls": "half the summed CSA row, not a changed softmax temperature",
        "mass_controls": "CSA group allocations but native normalized special-key content and original G",
        "conditional_control": "CSA conditional patch relation with original native special read and native patch mass",
        "zero_group_fallback": "uniform special conditional when own native special mass=0; G when CSA patch mass=0",
        "prefix": "native full QK attention for special query rows in all new arms; original four references unedited",
        "shell": "unchanged Value/output projection/LayerScale/residual/MLP/final projection, two evolving blocks",
        "precision": "unchanged bf16 AMP path; CSA softmax and additive probabilities follow matched SCLIP implementation",
        "text": "fixed20 aliases/six RS templates/normalized LME .07",
        "view": "native512/overlap128(step384)/Hann",
        "cost": "one backbone/one selected head; no second encoder, view, extra MLP or optimizer",
        "limits": "CSA rows sum to2, not probabilities; allocation or norm is not certified class correctness"}


def csa_weights(query, key, scale):
    qq = ((query.float() @ query.float().transpose(-1, -2))*scale).softmax(-1)
    kk = ((key.float() @ key.float().transpose(-1, -2))*scale).softmax(-1)
    return qq+kk


def factor_attended(native, csa, relation, value, prefix, mode, multiplier):
    if mode not in ("native", "csa", "allocation", "mass", "conditional"):
        raise ValueError("Unknown allocation factor.")
    if not 0 < prefix < value.shape[-2] or multiplier not in (.5, 1., 2.):
        raise ValueError("Nonempty key groups and fixed declared multiplier required.")
    g = relation[:, None] if relation.ndim == 3 else relation
    patches = value.shape[-2]-prefix
    if g.shape[-2:] != (patches, patches):
        raise ValueError("Geometry does not match patch grid.")
    native_special = native[..., prefix:, :prefix] @ value[..., :prefix, :]
    native_m = native[..., prefix:, prefix:].sum(-1, keepdim=True)
    patch = g.to(value.dtype) @ value[..., prefix:, :]
    if mode == "native":
        read = native_special+native_m*patch
    elif mode == "csa":
        read = csa[..., prefix:, :].to(value.dtype) @ value
    elif mode == "allocation":
        special = csa[..., prefix:, :prefix].to(value.dtype) @ value[..., :prefix, :]
        mass = csa[..., prefix:, prefix:].sum(-1, keepdim=True).to(value.dtype)
        read = special+mass*patch
    elif mode == "mass":
        old_w = native[..., prefix:, :prefix].sum(-1, keepdim=True).float()
        own = torch.where(old_w > 0, native_special.float()/old_w.clamp_min(1e-30),
                          value[..., :prefix, :].float().mean(-2, keepdim=True))
        special_m = csa[..., prefix:, :prefix].sum(-1, keepdim=True)
        patch_m = csa[..., prefix:, prefix:].sum(-1, keepdim=True)
        read = (special_m*own+patch_m*patch.float()).to(value.dtype)
    else:
        weights = csa[..., prefix:, prefix:].float()
        mass = weights.sum(-1, keepdim=True)
        conditional = torch.where(mass > 0, weights/mass.clamp_min(1e-30), g.float())
        read = native_special+native_m*(conditional.to(value.dtype) @ value[..., prefix:, :])
    # Scale the Value read, not the projection bias or query residual.
    read = read*multiplier
    return torch.cat((native[..., :prefix, :] @ value, read), dim=2)


@torch.inference_mode()
def read_head(head, prepared, method, *, trace=False):
    if method not in METHODS or prepared.block_index != 1:
        raise ValueError("Declared method and original two-block head required.")
    if method in BASELINES:
        return run_head(head, prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, method, 1)
    current, diagnostics = prepared.backbone_tokens, {}
    mode, multiplier = FACTORS[method]
    for index, block in enumerate(head.blocks[:2]):
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        csa = None if mode == "native" and not trace else csa_weights(query, key, block.attn.scale)
        attended = factor_attended(native, csa, prepared.geometry_patch_conditional, value,
                                   prepared.prefix_tokens, mode, multiplier)
        after = _finish_attention_block(block, current, attended)
        if trace:
            p = prepared.prefix_tokens
            merged = attended.transpose(1, 2).reshape(current.shape)
            increment = block.ls1(block.attn.proj_drop(block.attn.proj(merged)))
            n_m = native[..., p:, p:].float().sum(-1)
            c_m = csa[..., p:, p:].sum(-1)*.5
            diagnostics.update({f"block{index}_native_patch_mass": float(n_m.mean()),
                f"block{index}_csa_mean_patch_mass": float(c_m.mean()),
                f"block{index}_group_mass_abs_difference": float((n_m-c_m).abs().mean()),
                f"block{index}_csa_sum_row_mass_error": float((csa.sum(-1)-2).abs().max()),
                f"block{index}_value_read_norm": float(merged[:, p:].float().norm(dim=-1).mean()),
                f"block{index}_projected_increment_norm": float(increment[:, p:].float().norm(dim=-1).mean()),
                f"block{index}_query_state_norm": float(current[:, p:].float().norm(dim=-1).mean()),
                f"block{index}_output_state_norm": float(after[:, p:].float().norm(dim=-1).mean())})
        current = after
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_factors(geometry, prepared):
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        for method in METHODS:
            features[method], row = read_head(geometry.backbone.model.visual_model.head, prepared, method, trace=True)
            diagnostics.update({method+"__"+key: value for key, value in row.items()})
    if not torch.equal(features["Geometry"], prepared.geometry_projected):
        raise RuntimeError("Original Geometry replay differs.")
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError("Nonfinite allocation-factor features.")
    return features, diagnostics


@torch.inference_mode()
def encode_single(backbone, rgb, method):
    backbone._validate_rgb(rgb)
    with backbone._autocast():
        visual = backbone.model.visual_model
        normalized = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        dtype = torch.bfloat16 if backbone.use_amp else raw.dtype
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                                   geometry_patch_conditional=relation, block_index=1)
        return read_head(visual.head, prepared, method)[0]
