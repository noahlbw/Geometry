"""Per-head CSA relations minimally projected into a frozen Geometry budget.

CSA and KL projection are attributed standard operators. The tested change is
their conditional use in Geometry's native-mass-preserving attention shell.
"""
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, proxy_relation, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-per-head-semantic-budget-v1-20261004"
PRIMARY = "Geometry_SemanticBudget"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "CSA_SameShell", "Proxy_SameShell", PRIMARY)


@dataclass(frozen=True)
class BudgetConfig:
    expectation_tolerance: float = 2e-5
    bisection_steps: int = 24
    bracket_steps: int = 16

    def signature(self):
        return {"implementation": IMPLEMENTATION, "solver": asdict(self),
                "semantic_relation": "per-block per-head mean of conditional softmax(QQ) and softmax(KK), attributed CSA",
                "cost": "rowmax(structural_logits)-structural_logits on valid donors; original .10/.25 Geometry",
                "budget": "original Geometry conditional expected cost on valid donors, no fitted multiplier",
                "projection": "min KL(R||CSA) with row sum1 and expected cost<=Geometry budget",
                "shell": "original native patch mass, prefix paths, residual, MLP, final normalization; both blocks",
                "padding": "modify valid-to-valid edges only; preserve original invalid edges and their mass",
                "text": "unchanged20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128/Hann probability blending"}


def project_relation(log_semantic, cost, budget, config=BudgetConfig()):
    """Solve the one-constraint information projection in float32.

    Inactive rows retain their input softmax exactly. Active rows use the
    feasible side of a bracketed dual root, never a label-dependent threshold.
    """
    if (log_semantic.ndim < 2 or cost.shape != log_semantic.shape
            or budget.shape != log_semantic.shape[:-1]
            or config.expectation_tolerance <= 0 or config.bisection_steps < 1
            or config.bracket_steps < 1):
        raise ValueError("Invalid projection shapes or solver precision.")
    with torch.autocast(log_semantic.device.type, enabled=False):
        logits, distance, limit = log_semantic.float(), cost.float(), budget.float()
        if (bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any())
                or not bool(torch.isfinite(distance).all()) or bool((distance < 0).any())
                or not bool(torch.isfinite(limit).all())
                or bool((~torch.isfinite(logits).any(-1)).any())):
            raise ValueError("Require finite costs/budgets and nonempty semantic rows.")
        minimum = distance.masked_fill(~torch.isfinite(logits), torch.inf).amin(-1)
        if bool((limit < minimum).any()):
            raise ValueError("Infeasible geometric budget.")
        semantic = logits.softmax(-1)
        original_cost = (semantic*distance).sum(-1)
        active = original_cost > limit+config.expectation_tolerance
        relation = semantic.clone()
        dual = torch.zeros_like(limit)
        if bool(active.any()):
            base, d, b = logits[active], distance[active], limit[active]
            low, high = torch.zeros_like(b), torch.ones_like(b)
            for _ in range(config.bracket_steps):
                expectation = ((base-high[:, None]*d).softmax(-1)*d).sum(-1)
                bad = expectation > b
                if not bool(bad.any()):
                    break
                high = torch.where(bad, high*2, high)
            if bool((((base-high[:, None]*d).softmax(-1)*d).sum(-1) > b+config.expectation_tolerance).any()):
                raise RuntimeError("Geometry dual root could not be bracketed.")
            for _ in range(config.bisection_steps):
                mid = .5*(low+high)
                expectation = ((base-mid[:, None]*d).softmax(-1)*d).sum(-1)
                bad = expectation > b
                low = torch.where(bad, mid, low)
                high = torch.where(bad, high, mid)
            relation[active] = (base-high[:, None]*d).softmax(-1)
            dual[active] = high
        final_cost = (relation*distance).sum(-1)
        return relation, {"active_fraction": float(active.float().mean()),
                          "dual_mean": float(dual.mean()),
                          "semantic_expected_cost": float(original_cost.mean()),
                          "projected_expected_cost": float(final_cost.mean()),
                          "geometry_budget": float(limit.mean()),
                          "constraint_violation": float((final_cost-limit).clamp_min(0).max()),
                          "conditional_mass_error": float((relation.sum(-1)-1).abs().max()),
                          "projection_l1": float((relation-semantic).abs().sum(-1).mean())}


def relation_for_head(query, key, geometry, logits, valid, prefix, scale, mode):
    """Keep Geometry's invalid-edge allocation identical in every shell arm."""
    batch, heads, patches = query.shape[0], query.shape[1], geometry.shape[-1]
    if (geometry.shape != (batch, patches, patches) or logits.shape != geometry.shape
            or valid.shape != (batch, patches) or valid.dtype != torch.bool
            or not bool(valid.any(-1).all()) or query.shape[-2] != prefix+patches):
        raise ValueError("Invalid Geometry/head/validity inputs.")
    with torch.autocast(query.device.type, enabled=False):
        base = geometry[:, None].float().expand(batch, heads, patches, patches)
        eligible = (valid[:, None, :, None] & valid[:, None, None, :]).expand_as(base)
        old_valid = base.masked_fill(~eligible, 0.)
        mass = old_valid.sum(-1, keepdim=True)
        usable = valid[:, None].expand(batch, heads, patches)
        q, k = query[..., prefix:, :].float(), key[..., prefix:, :].float()
        donor_mask = ~valid[:, None, None, :]
        if mode == "proxy":
            semantic = logits[:, None].float().expand_as(base)
        elif mode in ("csa", "budget"):
            qq = (q @ q.transpose(-1, -2))*scale
            kk = (k @ k.transpose(-1, -2))*scale
            qq = qq.masked_fill(donor_mask, -torch.inf).log_softmax(-1)
            kk = kk.masked_fill(donor_mask, -torch.inf).log_softmax(-1)
            semantic = torch.logaddexp(qq, kk)-math.log(2.)
        else:
            raise ValueError("Unknown semantic relation mode.")
        semantic = semantic.masked_fill(donor_mask, -torch.inf)
        diagnostics = {}
        if mode == "budget":
            cost = logits.masked_fill(~valid[:, None, :], -torch.inf).amax(-1, keepdim=True)-logits
            cost = cost[:, None].expand_as(base).clamp_min(0)
            old_conditional = old_valid/mass.clamp_min(1e-30)
            budget = (old_conditional*cost).sum(-1)
            selected, diagnostics = project_relation(semantic[usable], cost[usable], budget[usable])
            conditional = semantic.softmax(-1)
            conditional[usable] = selected
        else:
            conditional = semantic.softmax(-1)
        replacement = (conditional*mass).masked_fill(~eligible, 0.)
        relation = base+replacement-old_valid
        diagnostics.update(row_mass_error=float((relation.sum(-1)-base.sum(-1)).abs().max()),
                           invalid_edge_error=float((relation-base).masked_fill(eligible, 0.).abs().max()))
        return relation, diagnostics


def shell_attended(native, relation, value, prefix):
    special = native[..., prefix:, :prefix] @ value[..., :prefix, :]
    mass = native[..., prefix:, prefix:].sum(-1, keepdim=True)
    patch = relation.to(value.dtype) @ value[..., prefix:, :]
    return torch.cat((native[..., :prefix, :] @ value, special+mass*patch), dim=2)


@torch.inference_mode()
def read_head(head, prepared, valid, mode, *, trace=True):
    current, prefix = prepared.backbone_tokens, prepared.prefix_tokens
    diagnostics = {}
    geometry = prepared.geometry_patch_conditional
    if mode == "proxy":
        logits, _, _ = proxy_relation(current[:, prefix:], 1.5, .5, strict_zero=True)
    elif mode == "budget":
        logits = getattr(prepared, "geometry_logits", None)
        if logits is None:
            logits = structural_logits(current[:, prefix:], prepared.grid_height, prepared.grid_width,
                                       temperature=.10, spatial_sigma=.25)
    else:
        logits = geometry
    for index, block in enumerate(head.blocks[:prepared.block_index+1]):
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        if mode == "geometry":
            attended = _geometry_attended(native, geometry, value, prefix)
            stats = {}
        else:
            relation, stats = relation_for_head(query, key, geometry, logits, valid, prefix, block.attn.scale, mode)
            attended = shell_attended(native, relation, value, prefix)
        after = _finish_attention_block(block, current, attended)
        if trace:
            merged = attended.transpose(1, 2).reshape(current.shape)
            increment = block.attn.proj_drop(block.attn.proj(merged))
            for name, tensor in (("value_read_norm", merged), ("projected_increment_norm", increment),
                                 ("attention_residual_norm", current+block.ls1(increment)), ("block_output_norm", after)):
                diagnostics[f"block{index}_{name}"] = float(tensor[:, prefix:].float().norm(dim=-1).mean())
            diagnostics[f"block{index}_patch_mass"] = float(native[..., prefix:, prefix:].float().sum(-1).mean())
        for field, number in stats.items():
            diagnostics[f"block{index}_{field}"] = number
        current = after
    projected = _finish_head(head, current, prepared.block_index)
    return F.normalize(projected[:, prefix:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_semantic_budget(geometry, prepared, valid, *, controls=True):
    head = geometry.backbone.model.visual_model.head
    features, totals = {}, {}
    modes = {PRIMARY: "budget"}
    if controls:
        modes.update(Geometry="geometry", CSA_SameShell="csa", Proxy_SameShell="proxy")
    with geometry.backbone._autocast():
        for method, mode in modes.items():
            features[method], diagnostics = read_head(head, prepared, valid, mode)
            totals.update({method+"__"+key: value for key, value in diagnostics.items()})
        if controls:
            error = float((features["Geometry"]-prepared.geometry_projected).abs().max())
            if error != 0:
                raise RuntimeError("Original Geometry replay changed.")
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens, raw,
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)[0]
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError("Nonfinite head output.")
    return features, totals


@torch.inference_mode()
def encode_single(backbone, rgb, valid, mode):
    """One backbone and one requested head, excluding comparison-head overhead."""
    backbone._validate_rgb(rgb)
    with backbone._autocast():
        visual = backbone.model.visual_model
        image = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
        cls, raw, registers = visual.get_backbone_features(image)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        h, w = rgb.shape[-2]//16, rgb.shape[-1]//16
        logits = structural_logits(raw, h, w, temperature=.10, spatial_sigma=.25)
        relation = logits.softmax(-1).to(raw.dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
            geometry_patch_conditional=relation, geometry_logits=logits,
            grid_height=h, grid_width=w, block_index=len(visual.head.blocks)-1)
        return read_head(visual.head, prepared, valid, mode, trace=False)[0]
