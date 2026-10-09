"""Test contrast compensation without moving Geometry's aggregate Value read."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .geometry_csa_allocation import read_head as allocation_head
from .matched_readout_controls import qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-conservative-contrast-v1-20261005"
PRIMARY = "Geometry_ConservativeContrast"
BASELINES = ("Geometry", "Geometry_BlockPrefix", "SCLIP_Two", "VIPProxy_Two")
REFERENCES = ("Geometry_DoubleRow", "Geometry_CSAAllocationSum")
FACTORS = {
    PRIMARY: "conservative",
    "Geometry_ContrastUncentered": "uncentered",
    "Geometry_ContrastFixed": "fixed",
    "Geometry_ContrastGainOnly": "gain_only",
    "Geometry_ContrastHeadShuffle": "head_shuffle",
}
METHODS = (*BASELINES, *REFERENCES, *FACTORS)


def settings():
    return {"implementation": IMPLEMENTATION, "primary": PRIMARY,
        "primary_rule": "original Geometry read + centered_query(gain * native_patch_mass * (G_normalized @ V_patch - mean(V_patch)))",
        "gain": "per-window/head max(0,1-Var_spatial(G_normalized @ V_patch)/Var_spatial(V_patch)); zero on constant Values",
        "range": "gain in[0,1], no fitted scalar/temperature/threshold; not class reliability",
        "centering": "all patch queries of the unchanged padded512 crop; before frozen projection, not output probabilities",
        "invariants": "zero mean added read at a fixed input; patch-common Value shifts cancel from correction; identity G gain0; collapsed uniform contrast gives zero correction",
        "precision": "original Geometry baseline bf16 read unedited; extra correction/statistics fp32 then cast before unchanged projection",
        "geometry_relation": "original cosine/.10 and spatial sigma .25; only extra correction uses normalized fp32 rows",
        "controls": FACTORS,
        "gain_only": "same input-derived per-head gain applied to the entire original patch-query read",
        "head_shuffle": "roll input-derived gains by half the head count, preserving gain multiset, not correction-norm distribution",
        "fixed_control": "same centered correction with gain1, not a tuned hyperparameter",
        "shell": "original prefix query attention, special contributions, two evolving blocks, residual/MLP/final normalization",
        "limits": "signed correction is not probability attention; head nonlinearity can change all class areas despite zero intermediate mean",
        "attribution": "contrast enhancement/variance compensation have standard precedents; no originality or CVPR-readiness claim",
        "text": "fixed20 aliases/six RS templates/normalized LME .07",
        "view": "native512/overlap128(step384)/Hann",
        "cost": "one backbone/one selected head; no additional encoder/view/MLP/optimizer"}


def contrast_correction(native, geometry, value, prefix, mode, *, trace=False):
    if mode not in FACTORS.values() or not 0 < prefix < value.shape[-2]:
        raise ValueError("Declared contrast mode and nonempty key groups required.")
    patches = value.shape[-2]-prefix
    if geometry.shape != (value.shape[0], patches, patches):
        raise ValueError("Geometry does not match patch grid.")
    original = _geometry_attended(native, geometry, value, prefix)
    with torch.autocast(device_type=value.device.type, enabled=False):
        v = value[..., prefix:, :].float()
        g = geometry.float()
        mass = g.sum(-1, keepdim=True)
        if bool((mass <= 0).any()):
            raise ValueError("Nonempty Geometry rows required.")
        g = (g/mass)[:, None]
        patch = g @ v
        mean = v.mean(-2, keepdim=True)
        reference_energy = (v-mean).square().sum(-1).mean(-1, keepdim=True)[..., None]
        read_energy = (patch-patch.mean(-2, keepdim=True)).square().sum(-1).mean(-1, keepdim=True)[..., None]
        gain = torch.where(reference_energy > 0,
            (1-read_energy/reference_energy.clamp_min(torch.finfo(torch.float32).tiny)).clamp(0., 1.),
            torch.zeros_like(reference_energy))
        if mode == "fixed":
            applied = torch.ones_like(gain)
        elif mode == "head_shuffle":
            applied = gain.roll(value.shape[1]//2, dims=1)
        else:
            applied = gain
        if mode == "gain_only":
            correction = applied*original[..., prefix:, :].float()
        else:
            patch_mass = native[..., prefix:, prefix:].float().sum(-1, keepdim=True)
            correction = applied*patch_mass*(patch-mean)
            if mode != "uncentered":
                correction = correction-correction.mean(-2, keepdim=True)
        read = original[..., prefix:, :].float()+correction
        diagnostics = {}
        if trace:
            diagnostics = {"reference_spatial_variance": float(reference_energy.mean()),
                "geometry_spatial_variance": float(read_energy.mean()),
                "gain_mean": float(gain.mean()), "gain_std": float(gain.std(unbiased=False)),
                "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                "applied_gain_mean": float(applied.mean()),
                "correction_norm": float(correction.norm(dim=-1).mean()),
                "original_read_norm": float(original[..., prefix:, :].float().norm(dim=-1).mean()),
                "correction_mean_norm": float(correction.mean(-2).norm(dim=-1).mean()),
                "correction_mean_max_error": float(correction.mean(-2).abs().max()),
                "native_patch_mass": float(native[..., prefix:, prefix:].float().sum(-1).mean())}
        attended = torch.cat((original[..., :prefix, :], read.to(value.dtype)), dim=2)
    return attended, diagnostics


@torch.inference_mode()
def read_head(head, prepared, method, *, trace=False):
    if method not in METHODS or prepared.block_index != 1:
        raise ValueError("Declared method and original two-block head required.")
    if method in BASELINES:
        return run_head(head, prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, method, 1)
    if method in REFERENCES:
        return allocation_head(head, prepared, method, trace=trace)
    current, diagnostics = prepared.backbone_tokens, {}
    for index, block in enumerate(head.blocks[:2]):
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended, row = contrast_correction(native, prepared.geometry_patch_conditional, value,
                                            prepared.prefix_tokens, FACTORS[method], trace=trace)
        current = _finish_attention_block(block, current, attended)
        diagnostics.update({f"block{index}_"+key: number for key, number in row.items()})
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_factors(geometry, prepared):
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        for method in METHODS:
            features[method], row = read_head(geometry.backbone.model.visual_model.head, prepared, method, trace=True)
            diagnostics.update({method+"__"+key: number for key, number in row.items()})
    if not torch.equal(features["Geometry"], prepared.geometry_projected):
        raise RuntimeError("Original Geometry replay differs.")
    if any(not bool(torch.isfinite(feature).all()) for feature in features.values()):
        raise RuntimeError("Nonfinite contrast features.")
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
