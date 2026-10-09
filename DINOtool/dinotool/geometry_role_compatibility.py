"""Condition Geometry edges on frozen semantic-head attention profiles."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, run_head
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-native-role-compatibility-v1-20261002"
PRIMARY = "Geometry_RoleCompatible"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "RoleOnly", "MeanLogit_RoleCompatible", PRIMARY)


@dataclass(frozen=True)
class RoleConfig:
    numerical_epsilon: float = 1e-7

    def signature(self):
        return {"role_compatibility": asdict(self), "implementation": IMPLEMENTATION,
                "relation": "original raw Geometry, conditioned on native attention-profile Hellinger distance",
                "scale": "per-head Geometry-weighted mean role distance; numerical zero gives exact Geometry",
                "mass": "original query patch/prefix mass; original valid-donor relation mass",
                "writeback": "original Geometry attended + m_patch (R-G)V, invalid edges unchanged",
                "head": "both frozen blocks, evolving states, residual/MLP/LN retained",
                "text": "unchanged20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128/Hann probability blending",
                "source": "one frozen DINOv3/DINO.text, no teacher or donor/class quotas"}


def role_relation(native, geometry, valid, prefix, *, mode="coupled", config=RoleConfig()):
    if mode not in ("coupled", "role_only", "geometry") or config.numerical_epsilon <= 0:
        raise ValueError("Invalid role mode or numerical precision.")
    patches = native.shape[-1]-prefix
    if (native.ndim != 4 or native.shape[-2] != native.shape[-1] or prefix < 1
            or geometry.shape != (native.shape[0], patches, patches)
            or valid.shape != (native.shape[0], patches) or valid.dtype != torch.bool):
        raise ValueError("Native attention, Geometry and validity shapes differ.")
    if (not all(bool(torch.isfinite(x).all()) for x in (native, geometry))
            or bool((native < 0).any()) or bool((geometry < 0).any())):
        raise ValueError("Require finite nonnegative attention and Geometry.")
    base = geometry[:, None].float().expand(native.shape[0], native.shape[1], patches, patches)
    if mode == "geometry":
        return base.clone(), {"role_distance": 0., "role_scale": 0., "relation_displacement": 0.,
                              "row_mass_error": 0., "invalid_edge_error": 0.}
    with torch.autocast(native.device.type, enabled=False):
        donors = torch.cat((torch.ones((len(valid), prefix), device=valid.device, dtype=torch.bool), valid), -1)
        profile = native[..., prefix:, :].float().masked_fill(~donors[:, None, None], 0.)
        if bool((profile.sum(-1)[valid[:, None].expand_as(profile[..., 0])] <= 0).any()):
            raise ValueError("A valid query has no eligible attention-profile donor.")
        profile = profile/profile.sum(-1, keepdim=True).clamp_min(1e-30)
        root = profile.sqrt()
        norm = root.square().sum(-1)
        distance = (.5*(norm[..., :, None]+norm[..., None, :])-root @ root.transpose(-1, -2)).clamp_min(0.)
        diagonal = torch.eye(patches, dtype=torch.bool, device=valid.device)
        distance = distance.masked_fill(diagonal, 0.)
        eligible = valid[:, None, :, None] & valid[:, None, None, :]
        old_valid = base.masked_fill(~eligible, 0.)
        row_mass = old_valid.sum(-1, keepdim=True)
        conditional = old_valid/row_mass.clamp_min(1e-30)
        queries = (valid.sum(-1)[:, None]).clamp_min(1)
        scale = (conditional*distance).sum((-1, -2))/queries
        # Equal profiles can accumulate tiny GEMM roundoff. The precision-only
        # branch keeps an uninformative role witness exactly neutral.
        reference_index = valid.long().argmax(-1)
        reference = profile.gather(-2, reference_index[:, None, None, None].expand(
            profile.shape[0], profile.shape[1], 1, profile.shape[-1]))
        equal = ((profile-reference).abs().masked_fill(~valid[:, None, :, None], 0.).amax((-1, -2)) == 0)
        informative = (scale > config.numerical_epsilon) & ~equal
        penalty = -distance/scale[..., None, None].clamp_min(config.numerical_epsilon)
        penalty = torch.where(informative[..., None, None], penalty, 0.)
        prior = old_valid if mode == "coupled" else eligible.float().expand_as(base)
        logits = prior.clamp_min(1e-30).log()+penalty
        # Invalid query rows are not written back; give them finite softmax input.
        logits = logits.masked_fill(~eligible | (prior <= 0), -torch.inf)
        usable = valid[:, None, :, None] & (row_mass > 0)
        logits = torch.where(usable, logits, torch.zeros_like(logits))
        new_valid = logits.softmax(-1)*row_mass
        new_valid = new_valid.masked_fill(~eligible, 0.)
        if mode == "coupled":
            new_valid = torch.where(informative[..., None, None], new_valid, old_valid)
        displacement = new_valid-old_valid
        relation = base+displacement
        diagnostics = {"role_distance": float((conditional*distance).sum()/queries.sum().mul(native.shape[1]).clamp_min(1)),
            "role_scale": float(scale.mean()),
            "relation_displacement": float(displacement.abs().sum()/row_mass.sum().clamp_min(1e-30)),
            "row_mass_error": float((relation.sum(-1)-base.sum(-1)).abs().max()),
            "invalid_edge_error": float(displacement.masked_fill(eligible, 0.).abs().max())}
        if not bool(torch.isfinite(relation).all()) or bool((relation < -1e-7).any()):
            raise RuntimeError("Nonfinite or negative role-conditioned relation.")
        return relation, diagnostics


def role_attended(native, geometry, values, valid, prefix, *, mode="coupled", config=RoleConfig()):
    original = _geometry_attended(native, geometry, values, prefix, "preserve")
    relation, diagnostics = role_relation(native, geometry, valid, prefix, mode=mode, config=config)
    with torch.autocast(values.device.type, enabled=False):
        displacement = (relation-geometry[:, None].float()) @ values[..., prefix:, :].float()
        mass = native[..., prefix:, prefix:].sum(-1, keepdim=True).float()
        displacement = mass*displacement
    attended = original.clone()
    attended[..., prefix:, :] += displacement.to(values.dtype)
    diagnostics["prefix_displacement_max_error"] = float((attended[..., :prefix, :]-original[..., :prefix, :]).abs().max())
    diagnostics["invalid_query_displacement_max_error"] = float(displacement.masked_fill(valid[:, None, :, None], 0.).abs().max())
    return attended, diagnostics


def read_role_head(head, prepared, valid, *, mode="coupled", config=RoleConfig()):
    current, prefix = prepared.backbone_tokens, prepared.prefix_tokens
    totals = {}
    for block in head.blocks[:prepared.block_index+1]:
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended, diagnostics = role_attended(native, prepared.geometry_patch_conditional,
                                              value, valid, prefix, mode=mode, config=config)
        current = _finish_attention_block(block, current, attended)
        for field, number in diagnostics.items():
            totals[field] = max(totals.get(field, 0.), number) if "error" in field else totals.get(field, 0.)+number/(prepared.block_index+1)
    projected = _finish_head(head, current, prepared.block_index)
    return F.normalize(projected[:, prefix:].float(), dim=-1), totals


@torch.inference_mode()
def read_role_compatibility(geometry, prepared, valid, *, controls=True):
    head = geometry.backbone.model.visual_model.head
    with geometry.backbone._autocast():
        primary, diagnostics = read_role_head(head, prepared, valid)
        features = {PRIMARY: primary, "Geometry": prepared.geometry_projected}
        if controls:
            features["RoleOnly"], _ = read_role_head(head, prepared, valid, mode="role_only")
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens, raw,
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)[0]
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise ValueError("Nonfinite role-conditioned head features.")
    return features, diagnostics
