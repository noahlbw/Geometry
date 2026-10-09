"""Separate context allocation from final dense Geometry evidence readout."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-context-then-patch-readout-v1-20261004"
PRIMARY = "Geometry_Staged"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
POLICIES = {"Geometry": ("preserve", "preserve"),
            "Geometry_BlockPrefix": ("block", "block"),
            PRIMARY: ("preserve", "block"),
            "Geometry_ReverseStage": ("block", "preserve"),
            "Geometry_PrefixUnit": ("prefix_unit", "prefix_unit"),
            "Geometry_PatchMass": ("patch_mass", "patch_mass")}
METHODS = (*BASELINES, *tuple(method for method in POLICIES if method != "Geometry"))


def settings():
    return {"implementation": IMPLEMENTATION, "policies": {key: list(value) for key, value in POLICIES.items()},
            "relation": "unchanged original Geometry, cosine/.10 and spatial sigma .25, shared across heads",
            "primary": "block0 preserves native prefix/patch allocation; block1 reads unit-mass patches only",
            "special_tokens": "native attention for prefix queries in every arm; not VIP normalized-prefix increment",
            "shell": "unchanged frozen Value/projection/residual/MLP/final normalization",
            "factorial": "both-layer preserve vs block vs prefix+unit-patch vs mass-patch-only",
            "text": "unchanged20 aliases/six RS templates/normalized LME .07",
            "view": "native512/128/Hann probability blending",
            "source": "one backbone forward and one selected two-block head; no extra encoder or view"}


def attended_with_policy(native, geometry, value, prefix, policy):
    if policy in ("preserve", "block"):
        return _geometry_attended(native, geometry, value, prefix, policy)
    if policy not in ("prefix_unit", "patch_mass"):
        raise ValueError("Unknown Geometry allocation policy.")
    prefix_value = native[..., :prefix, :] @ value
    patch = geometry[:, None].to(value.dtype) @ value[..., prefix:, :]
    if policy == "prefix_unit":
        patch = patch+native[..., prefix:, :prefix] @ value[..., :prefix, :]
    else:
        patch = native[..., prefix:, prefix:].sum(-1, keepdim=True)*patch
    return torch.cat((prefix_value, patch), dim=2)


@torch.inference_mode()
def read_head(head, prepared, method, *, trace=False):
    if method not in POLICIES or prepared.block_index != 1:
        raise ValueError("Require a declared allocation and the existing two-block DINO.text head.")
    current, prefix = prepared.backbone_tokens, prepared.prefix_tokens
    diagnostics = {}
    for index, block in enumerate(head.blocks[:2]):
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended = attended_with_policy(native, prepared.geometry_patch_conditional, value, prefix, POLICIES[method][index])
        after = _finish_attention_block(block, current, attended)
        if trace:
            merged = attended.transpose(1, 2).reshape(current.shape)
            increment = block.attn.proj_drop(block.attn.proj(merged))
            for name, tensor in (("value_read_norm", merged), ("projected_increment_norm", increment),
                                 ("attention_residual_norm", current+block.ls1(increment)), ("block_output_norm", after)):
                diagnostics[f"block{index}_{name}"] = float(tensor[:, prefix:].float().norm(dim=-1).mean())
            diagnostics[f"block{index}_native_patch_mass"] = float(native[..., prefix:, prefix:].float().sum(-1).mean())
            special = native[..., prefix:, :prefix] @ value[..., :prefix, :]
            diagnostics[f"block{index}_native_special_read_norm"] = float(special.float().norm(dim=-1).mean())
            patch = prepared.geometry_patch_conditional[:, None].to(value.dtype) @ value[..., prefix:, :]
            diagnostics[f"block{index}_unit_geometry_read_norm"] = float(patch.float().norm(dim=-1).mean())
        current = after
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prefix:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_staged(geometry, prepared, *, controls=True):
    head = geometry.backbone.model.visual_model.head
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        for method in (POLICIES if controls else (PRIMARY,)):
            features[method], current = read_head(head, prepared, method, trace=controls)
            diagnostics.update({method+"__"+name: value for name, value in current.items()})
        if controls:
            if not torch.equal(features["Geometry"], prepared.geometry_projected):
                raise RuntimeError("Original Geometry replay changed.")
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens, raw,
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, 1)[0]
    if any(not bool(torch.isfinite(feature).all()) for feature in features.values()):
        raise RuntimeError("Nonfinite staged features.")
    return features, diagnostics


@torch.inference_mode()
def encode_single(backbone, rgb, method):
    backbone._validate_rgb(rgb)
    with backbone._autocast():
        visual = backbone.model.visual_model
        image = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
        cls, raw, registers = visual.get_backbone_features(image)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(raw.dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                                   geometry_patch_conditional=relation, block_index=1)
        return read_head(visual.head, prepared, method)[0]
