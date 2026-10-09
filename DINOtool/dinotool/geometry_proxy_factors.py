"""Attribute VIP support, weighting and readout effects inside Geometry."""
from math import isqrt
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .matched_readout_controls import proxy_relation, qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-vip-proxy-factor-transplant-v1-20261004"
PRIMARY = "Geometry_VIPSupport"
BASELINES = ("Geometry", "Geometry_BlockPrefix", "SCLIP_Two", "VIPProxy_Two")
FACTORS = {
    PRIMARY: ("vip", "geometry", "geometry"),
    "Geometry_VIPWeights": ("dense", "weak_spatial", "geometry"),
    "Geometry_VIPSupportWeights": ("vip", "weak_spatial", "geometry"),
    "VIPRelation_GeometryPath": ("vip", "vip", "geometry"),
    "Geometry_VIPPath": ("dense", "geometry", "vip"),
}
METHODS = (*BASELINES, *FACTORS)


def settings():
    return {"implementation": IMPLEMENTATION,
            "factors": {name: list(value) for name, value in FACTORS.items()},
            "support": "pinned VIP global crop mean: S>1.5*mean(S); self only on empty rows",
            "geometry_relation": "original cosine/.10, spatial sigma .25, both head blocks",
            "weak_spatial": "cosine*.5 plus unchanged original spatial logits",
            "vip_relation": "pinned (S-1.5*mean(S))*.5 masked softmax, no spatial term",
            "geometry_path": "native special queries/contribution and native patch mass preserved",
            "vip_path": "unit-mass patch-only read and normalized-prefix attention increment",
            "primary_rule": "support transplant only; no change of geometry weights or readout path",
            "shell": "unchanged frozen V/projection/residual/MLP/final normalization",
            "text": "unchanged20 aliases/six RS templates/normalized LME .07",
            "view": "native512/128/Hann probability blending",
            "attribution": "VIP mechanisms are borrowed controls, not claimed novel",
            "source": "one backbone, one selected two-block head; no extra encoder/view"}


def restrict_relation(relation, support):
    if relation.shape != support.shape or support.dtype != torch.bool:
        raise ValueError("Matching relation and boolean support required.")
    weights = relation.float().masked_fill(~support, 0.)
    mass = weights.sum(-1, keepdim=True)
    diagonal = torch.eye(relation.shape[-1], device=relation.device)[None]
    return torch.where(mass > 0, weights/mass.clamp_min(1e-30), diagonal).to(relation.dtype)


def relations(prepared):
    raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
    side = isqrt(raw.shape[1])
    if side*side != raw.shape[1]:
        raise ValueError("This fixed screen requires a square patch grid.")
    original = prepared.geometry_patch_conditional
    vip_logits, empty, rows = proxy_relation(raw, 1.5, .5, strict_zero=True)
    support = torch.isfinite(vip_logits)
    weak = structural_logits(raw, side, side, temperature=2., spatial_sigma=.25).softmax(-1).to(original.dtype)
    result = {"geometry": original,
              "vip_support": restrict_relation(original, support),
              "weak_spatial": weak,
              "vip_support_weak": restrict_relation(weak, support),
              "vip": vip_logits.to(raw.dtype).softmax(-1)}
    removed_mass = original.float().masked_fill(support, 0.).sum(-1)
    return result, {"vip_empty_row_fraction": empty/rows,
                    "vip_support_fraction": float(support.float().mean()),
                    "geometry_removed_mass": float(removed_mass.mean()),
                    "geometry_removed_mass_max": float(removed_mass.max())}


def vip_block(block, current, value, relation, prefix):
    attended = relation[:, None].to(value.dtype) @ value[..., prefix:, :]
    merged = attended.transpose(1, 2).reshape(current.shape[0], relation.shape[1], -1)
    increment = torch.cat((block.norm1(current)[:, :prefix],
                           block.attn.proj_drop(block.attn.proj(merged))), dim=1)
    after = current+block.ls1(increment)
    return after+block.ls2(block.mlp(block.norm2(after)))


@torch.inference_mode()
def read_head(head, prepared, method, *, relation_cache=None, trace=False):
    if method not in METHODS or prepared.block_index != 1:
        raise ValueError("Declared method and the existing two-block head required.")
    raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
    if method in BASELINES:
        return run_head(head, prepared.backbone_tokens, raw, prepared.geometry_patch_conditional,
                        prepared.prefix_tokens, method, 1)
    cache = relations(prepared)[0] if relation_cache is None else relation_cache
    support, weighting, path = FACTORS[method]
    key = ("vip_support" if weighting == "geometry" and support == "vip" else
           "vip_support_weak" if weighting == "weak_spatial" and support == "vip" else weighting)
    relation = cache[key]
    current, prefix, diagnostics = prepared.backbone_tokens, prepared.prefix_tokens, {}
    for index, block in enumerate(head.blocks[:2]):
        query, keys, value = qkv(block, current)
        if path == "vip":
            current = vip_block(block, current, value, relation, prefix)
        else:
            native = ((query.float() @ keys.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
            attended = _geometry_attended(native, relation, value, prefix)
            current = _finish_attention_block(block, current, attended)
            if trace:
                diagnostics[f"block{index}_native_patch_mass"] = float(native[..., prefix:, prefix:].float().sum(-1).mean())
        if trace:
            diagnostics[f"block{index}_output_norm"] = float(current[:, prefix:].float().norm(dim=-1).mean())
    if trace:
        p = relation.float()
        diagnostics["self_weight"] = float(p.diagonal(dim1=-2, dim2=-1).mean())
        diagnostics["effective_donors"] = float(p.square().sum(-1).reciprocal().mean())
        diagnostics["relation_l1_from_geometry"] = float((p-cache["geometry"].float()).abs().sum(-1).mean())
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prefix:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_factors(geometry, prepared):
    head = geometry.backbone.model.visual_model.head
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        cache, diagnostics = relations(prepared)
        for method in METHODS:
            features[method], current = read_head(head, prepared, method, relation_cache=cache, trace=True)
            diagnostics.update({method+"__"+name: value for name, value in current.items()})
    if not torch.equal(features["Geometry"], prepared.geometry_projected):
        raise RuntimeError("Original Geometry replay differs.")
    if any(not bool(torch.isfinite(feature).all()) for feature in features.values()):
        raise RuntimeError("Nonfinite proxy-factor features.")
    return features, diagnostics


@torch.inference_mode()
def encode_single(backbone, rgb, method):
    backbone._validate_rgb(rgb)
    with backbone._autocast():
        visual = backbone.model.visual_model
        image = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
        cls, raw, registers = visual.get_backbone_features(image)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        value_dtype = torch.bfloat16 if backbone.use_amp else raw.dtype
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(value_dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                                   geometry_patch_conditional=relation, block_index=1)
        # A deployed support-only arm does not construct unrelated control relations.
        cache = {"geometry": relation}
        if method == PRIMARY:
            logits, _, _ = proxy_relation(raw, 1.5, .5, strict_zero=True)
            cache["vip_support"] = restrict_relation(relation, torch.isfinite(logits))
        elif method in FACTORS:
            cache = relations(prepared)[0]
        return read_head(visual.head, prepared, method, relation_cache=cache)[0]
