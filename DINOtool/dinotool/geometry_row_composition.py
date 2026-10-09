"""Compose geometric neighbors with native self-conditional attention rows."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .geometry_semantic_response import read_head as response_head
from .matched_readout_controls import qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head


IMPLEMENTATION = "geometry-native-row-composition-v1-20261005"
PRIMARY = "Geometry_RowCompose"
BASELINES = ("Geometry", "Geometry_BlockPrefix", "SCLIP_Two", "VIPProxy_Two")
FACTORS = {
    "Geometry_FPRead": "original",
    PRIMARY: "full",
    "Geometry_BudgetOnly": "budget",
    "Geometry_SpecialOnly": "special",
    "Geometry_GroupCompose": "group",
    "Geometry_ConditionalOnly": "conditional",
    "Geometry_GlobalBudget": "global",
}
METHODS = (*BASELINES, *FACTORS, "Geometry_DonorBefore")


def settings():
    return {"implementation": IMPLEMENTATION,
            "primary_rule": "G @ (A_patch_to_special @ V_special + native_patch_mass * V_patch)",
            "effective_row": "special coefficients=G@A_special; patch coefficients=G_ij*m_j",
            "geometry_relation": "original cosine/.10 and spatial sigma .25; normalized in fp32",
            "precision": "fp32 native softmax, row normalization and read; cast attended to native Value dtype before unchanged projection",
            "prefix": "native full attention for prefix queries at each evolving input state",
            "controls": FACTORS,
            "budget_factor": "G@m versus own m; unchanged patch conditional G",
            "special_factor": "conditional special read G@special/(G@(1-m)) versus own special/(1-m)",
            "conditional_factor": "G_ij*m_j/(G@m)_i versus G; full versus GroupCompose",
            "global_control": "per-crop per-head mean native patch mass; original G and own conditional special read",
            "zero_group_fallback": "uniform special-key conditional if own special mass=0; own conditional if mixed special mass=0; G patch conditional if G@m=0",
            "attribution": "DonorBefore is a developed prior control; row composition is not claimed first in literature",
            "shell": "unchanged frozen Value/projection/LayerScale/residual/MLP/final projection; two evolving head blocks",
            "text": "unchanged20 aliases/six RS templates/normalized LME .07",
            "view": "native512/128/Hann probability blending",
            "source": "one backbone/one selected two-block head; no additional encoder, view or MLP",
            "interpretation": "native patch mass is allocation, NOT semantic reliability"}


def normalize_rows(weights):
    weights = weights.float()
    mass = weights.sum(-1, keepdim=True)
    identity = torch.eye(weights.shape[-1], device=weights.device)
    return torch.where(mass > 0, weights/mass.clamp_min(1e-30), identity)


def composed_attended(native, relation, value, prefix, mode, *, trace=False):
    if mode not in FACTORS.values() or not 0 < prefix < value.shape[-2]:
        raise ValueError("Declared composition mode and nonempty special/patch groups required.")
    if relation.shape != (value.shape[0], value.shape[-2]-prefix, value.shape[-2]-prefix):
        raise ValueError("Geometry relation does not match patch grid.")
    # Keep the whole attention row stochastic, not just its patch conditional.
    with torch.autocast(device_type=value.device.type, enabled=False):
        a, g, v = normalize_rows(native), normalize_rows(relation)[:, None], value.float()
        special_weights = a[..., prefix:, :prefix]
        m = a[..., prefix:, prefix:].sum(-1, keepdim=True)
        w = special_weights.sum(-1, keepdim=True)
        special = special_weights @ v[..., :prefix, :]
        if mode != "full":
            patch = g @ v[..., prefix:, :]
        if mode in ("budget", "special", "group", "global") or trace:
            own_cond = torch.where(w > 0, special/w.clamp_min(1e-30),
                                   v[..., :prefix, :].mean(-2, keepdim=True))
        if mode in ("budget", "group", "conditional") or trace:
            moved_m = g @ m
        if mode in ("special", "group") or trace:
            mixed_special, moved_w = g @ special, g @ w
            mixed_cond = torch.where(moved_w > 0, mixed_special/moved_w.clamp_min(1e-30), own_cond)
        if mode == "original":
            result = special+m*patch
        elif mode == "full":
            result = g @ (special+m*v[..., prefix:, :])
        elif mode == "budget":
            result = (1-moved_m)*own_cond+moved_m*patch
        elif mode == "special":
            result = w*mixed_cond+m*patch
        elif mode == "group":
            result = moved_w*mixed_cond+moved_m*patch
        elif mode == "conditional":
            conditional_patch = torch.where(moved_m > 0,
                (g @ (m*v[..., prefix:, :]))/moved_m.clamp_min(1e-30), patch)
            result = special+m*conditional_patch
        else:
            global_m = m.mean(-2, keepdim=True)
            result = (1-global_m)*own_cond+global_m*patch
        attended = torch.cat((a[..., :prefix, :] @ v, result), dim=2)
        diagnostics = {}
        if trace:
            diagnostics = {"native_patch_mass": float(m.mean()),
                "native_patch_mass_std": float(m.std(unbiased=False)),
                "moved_patch_mass": float(moved_m.mean()),
                "patch_mass_abs_change": float((moved_m-m).abs().mean()),
                "special_read_abs_change": float((mixed_cond-own_cond).abs().mean()),
                "row_mass_max_error": float((moved_m+moved_w-1).abs().max()),
                "zero_own_special_fraction": float((w==0).float().mean()),
                "zero_mixed_patch_fraction": float((moved_m==0).float().mean())}
    return attended.to(value.dtype), diagnostics


@torch.inference_mode()
def read_head(head, prepared, method, *, trace=False):
    if method not in METHODS or prepared.block_index != 1:
        raise ValueError("Declared method and original two-block head required.")
    if method in BASELINES:
        return run_head(head, prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, method, 1)
    if method == "Geometry_DonorBefore":
        return response_head(head, prepared, method, trace=trace)
    current, diagnostics = prepared.backbone_tokens, {}
    for index, block in enumerate(head.blocks[:2]):
        query, key, value = qkv(block, current)
        with torch.autocast(device_type=value.device.type, enabled=False):
            native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1)
        attended, row = composed_attended(native, prepared.geometry_patch_conditional, value,
                                         prepared.prefix_tokens, FACTORS[method], trace=trace)
        current = _finish_attention_block(block, current, attended)
        diagnostics.update({f"block{index}_"+key: val for key, val in row.items()})
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_compositions(geometry, prepared):
    head = geometry.backbone.model.visual_model.head
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        for method in METHODS:
            features[method], row = read_head(head, prepared, method, trace=True)
            diagnostics.update({method+"__"+key: value for key, value in row.items()})
    if not torch.equal(features["Geometry"], prepared.geometry_projected):
        raise RuntimeError("Original Geometry replay differs.")
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError("Nonfinite composition features.")
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
