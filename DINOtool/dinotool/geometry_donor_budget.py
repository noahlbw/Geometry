"""Couple Geometry conditionals to frozen head-specific visual donor budgets."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, run_head
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-native-donor-marginal-v1-20261002"
PRIMARY = "Geometry_DonorBudget"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "UniformDonorBudget", "MeanLogit_DonorBudget", PRIMARY)


@dataclass(frozen=True)
class DonorBudgetConfig:
    maximum_iterations: int = 1024
    check_every: int = 4
    marginal_tolerance: float = 1e-5

    def signature(self):
        return {"donor_budget": asdict(self), "implementation": IMPLEMENTATION,
                "relation": "unchanged original raw DINO Geometry G",
                "query_mass": "original Geometry flow into valid donors",
                "donor_mass": "current frozen native head attention column mass, restricted to valid patches",
                "budget_scale": "only total-mass compatibility; no per-class quota or fitted coefficient",
                "objective": "min_T generalized_KL(T||Q), T1=Q1, T^T1=native_donor_mass",
                "solver": "FP32 proportional fitting, max1024, marginal L1 tolerance1e-5",
                "writeback": "original Geo attended + (T-Q)V; zero displacement exactly replays Geometry",
                "head": "both original blocks; evolving Q/K/V, prefix queries, residual/MLP/norm retained",
                "padding": "no displacement to/from invalid patches",
                "text": "unchanged20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128/Hann probability blending"}


def marginal_error(flow, rows, columns):
    total = rows.sum(-1).clamp_min(1e-30)
    row_error = (flow.sum(-1)-rows).abs().sum(-1)/total
    col_error = (flow.sum(-2)-columns).abs().sum(-1)/total
    return max(float(row_error.max()), float(col_error.max()))


def project_donor_flow(reference, columns, config=DonorBudgetConfig()):
    """KL projection onto visual-token marginals, not class occupancy constraints."""
    if reference.ndim < 2 or columns.shape != reference.shape[:-2]+reference.shape[-1:]:
        raise ValueError("Flow and column-budget shapes differ.")
    if (config.maximum_iterations < 1 or config.check_every < 1 or config.marginal_tolerance <= 0
            or not all(bool(torch.isfinite(x).all()) for x in (reference, columns))
            or bool((reference < 0).any()) or bool((columns < 0).any())):
        raise ValueError("Require valid solver settings and finite nonnegative flows/budgets.")
    with torch.autocast(reference.device.type, enabled=False):
        base, columns = reference.float(), columns.float()
        rows = base.sum(-1)
        total = rows.sum(-1)
        if not bool(torch.isclose(total, columns.sum(-1), atol=1e-7, rtol=2e-6).all()):
            raise ValueError("Query and donor total masses must match.")
        before = marginal_error(base, rows, columns)
        if before <= config.marginal_tolerance:
            return base.clone(), {"iterations": 0., "marginal_error": before, "flow_kl": 0.}
        possible = (base > 0) & (rows[..., :, None] > 0) & (columns[..., None, :] > 0)
        if bool(((rows > 0) & ~possible.any(-1)).any()) or bool(((columns > 0) & ~possible.any(-2)).any()):
            raise ValueError("Positive marginal has no eligible visual edge.")
        kernel = base.masked_fill(~possible, 0.)
        v = (columns > 0).float()
        # Matrix-vector scaling avoids exponentiating the full edge grid every
        # iteration. Gauge normalization leaves the transport plan unchanged.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            for step in range(config.maximum_iterations):
                u = rows/(kernel @ v[..., None])[..., 0].clamp_min(1e-30)
                v = columns/(kernel.transpose(-1, -2) @ u[..., None])[..., 0].clamp_min(1e-30)
                if (step+1) % config.check_every == 0 or step+1 == config.maximum_iterations:
                    flow = u[..., :, None]*kernel*v[..., None, :]
                    error = marginal_error(flow, rows, columns)
                    if error <= config.marginal_tolerance:
                        break
                    active = columns > 0
                    gauge = ((v.clamp_min(1e-30).log()*active).sum(-1)
                             /active.sum(-1).clamp_min(1)).exp()
                    v = v/gauge[..., None].clamp_min(1e-30)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if not bool(torch.isfinite(flow).all()) or error > config.marginal_tolerance:
            raise RuntimeError(f"Visual donor flow did not reach numerical marginals: {error}")
        divergence = (flow*(flow.clamp_min(1e-30).log()-base.clamp_min(1e-30).log())-flow+base).sum((-1, -2))
        return flow, {"iterations": float(step+1), "marginal_error": error,
                      "flow_kl": float((divergence/total.clamp_min(1e-30)).mean())}


def donor_reference(native, geometry, valid, prefix, mode):
    patches = native.shape[-1]-prefix
    if (native.ndim != 4 or geometry.shape != (len(native), patches, patches)
            or valid.shape != (len(native), patches) or prefix < 1):
        raise ValueError("Native attention, Geometry and patch-validity shapes differ.")
    patch_attention = native[..., prefix:, prefix:]
    mass = patch_attention.sum(-1, keepdim=True).float()
    eligible = valid[:, None, :, None] & valid[:, None, None, :]
    reference = mass*geometry[:, None].float()*eligible
    total = reference.sum((-1, -2), keepdim=False)
    if mode == "native":
        columns = (patch_attention.float()*eligible).sum(-2)
    elif mode == "uniform":
        columns = valid[:, None].float().expand(native.shape[0], native.shape[1], patches)
    elif mode == "geometry":
        columns = reference.sum(-2)
    else:
        raise ValueError("Unknown visual donor-budget source.")
    denominator = columns.sum(-1)
    if bool(((total > 0) & (denominator <= 0)).any()):
        raise ValueError("No native donor mass for a nonempty Geometry flow.")
    columns = columns*(total/denominator.clamp_min(1e-30))[..., None]
    return reference, columns


def budget_attended(native, geometry, values, valid, prefix, mode, config):
    attended = _geometry_attended(native, geometry, values, prefix, "preserve")
    reference, columns = donor_reference(native, geometry, valid, prefix, mode)
    flow, diagnostics = project_donor_flow(reference, columns, config)
    with torch.autocast(values.device.type, enabled=False):
        displacement = (flow-reference) @ values[..., prefix:, :].float()
    changed = attended.clone()
    changed[..., prefix:, :] += displacement.to(values.dtype)
    original = reference.sum(-2)
    total = original.sum(-1).clamp_min(1e-30)
    diagnostics["incoming_mass_displacement"] = float(((columns-original).abs().sum(-1)/total).mean())
    diagnostics["geometry_maximum_donor_fraction"] = float((original.amax(-1)/total).mean())
    diagnostics["native_budget_maximum_donor_fraction"] = float((columns.amax(-1)/total).mean())
    diagnostics["prefix_displacement_max_error"] = float((changed[..., :prefix, :]-attended[..., :prefix, :]).abs().max())
    diagnostics["invalid_query_displacement_max_error"] = float(displacement.masked_fill(valid[:, None, :, None], 0).abs().max())
    return changed, diagnostics


def read_budget_head(head, prepared, valid, *, mode="native", config=DonorBudgetConfig()):
    current, prefix = prepared.backbone_tokens, prepared.prefix_tokens
    totals = {}
    for block in head.blocks[:prepared.block_index+1]:
        query, key, value = qkv(block, current)
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended, diagnostics = budget_attended(native, prepared.geometry_patch_conditional,
                                               value, valid, prefix, mode, config)
        current = _finish_attention_block(block, current, attended)
        for field, number in diagnostics.items():
            totals[field] = max(totals.get(field, 0.), number) if "error" in field else totals.get(field, 0.)+number/(prepared.block_index+1)
    projected = _finish_head(head, current, prepared.block_index)
    return F.normalize(projected[:, prefix:].float(), dim=-1), totals


@torch.inference_mode()
def read_donor_budget(geometry, prepared, valid, *, controls=True):
    head = geometry.backbone.model.visual_model.head
    with geometry.backbone._autocast():
        primary, diagnostics = read_budget_head(head, prepared, valid)
        features = {PRIMARY: primary, "Geometry": prepared.geometry_projected}
        if controls:
            features["UniformDonorBudget"], _ = read_budget_head(head, prepared, valid, mode="uniform")
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens, raw,
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)[0]
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise ValueError("Nonfinite donor-constrained head output.")
    return features, diagnostics
