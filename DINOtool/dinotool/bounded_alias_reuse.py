"""Bounded branch-union contenders; native evidence reuse with no fine forwards."""
import torch

from .fine_alias_view import CONFIG
from .sparse_alias_reuse import survivor_allocation


IMPLEMENTATION = 'geometry-native-reuse-branch-union6-alias-v1-20261005'
PRIMARY = 'SparseNativeUnion6Soft'
METHODS = ('Geometry', 'NoAdmission_Exact', 'SparseNativeSoft', PRIMARY)


def contender_union(base, local, broad, *, baseline_only=False, witness_classes=None):
    classes = base.shape[-1]
    union = torch.zeros_like(base, dtype=torch.bool)
    sources = (base,) if baseline_only else (base, witness_classes) if witness_classes is not None else (base, local, broad)
    for scores in sources:
        union.scatter_(1, scores.argsort(dim=-1, descending=True, stable=True)[:, :2], True)
    width = min(2 * len(sources), classes)
    indices = base.masked_fill(~union, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :width]
    return indices, union.gather(1, indices)


@torch.inference_mode()
def bounded_scores(local, operator, broad, observations, witness, valid, layout, *,
                   soft=True, diagnostics=True, beta=CONFIG.beta, baseline_only=False, chunk=128,
                   witness_contenders=False):
    n, classes = local.shape
    if (classes != len(layout.members) or classes < 2 or broad.shape != local.shape
            or operator.shape != (n, n) or valid.shape != (n,)
            or witness.shape != (n, *layout.members.shape) or beta <= 0 or chunk <= 0 or not observations):
        raise ValueError('Matching frozen local/wide/reuse fields and positive block size required.')
    base = local.double() + operator.double() @ (broad.double() - local.double())
    witness_classes = (beta * witness).logsumexp(-1) / beta if witness_contenders else None
    pairs, active = contender_union(base, local, broad, baseline_only=baseline_only, witness_classes=witness_classes)
    posterior = base.softmax(-1)
    potential = torch.zeros_like(base)
    action_sum = active_edges = rejected = eligible_count = 0.
    mass_error = canonical_error = 0.
    peak_elements = 0
    saved = []
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        indices = pairs[sl]
        rows = torch.arange(start, sl.stop, device=local.device)
        selected = witness[rows[:, None], indices].double()
        alias_valid, canonical = layout.valid[indices], layout.canonical[indices]
        counts = layout.counts[indices].double()
        total_local = (beta * selected).logsumexp(-1) / beta
        mean_local = total_local - counts.log() / beta
        local_margin = selected[:, :, None] - mean_local[:, None, :, None]
        wide_margin = torch.zeros_like(local_margin)
        cached = []
        for crop in observations:
            evidence = crop.evidence[crop.indices[sl, :, None], indices[:, None]].double()
            total = (beta * evidence).logsumexp(-1) / beta
            mean = total - counts[:, None].log() / beta
            margin = evidence[:, :, :, None] - mean[:, :, None, :, None]
            coefficients = crop.coefficients[sl].double()
            wide_margin += (margin.masked_fill(~alias_valid[:, None, :, None], 0.)
                            * coefficients[..., None, None, None]).sum(1)
            cached.append((evidence, total, coefficients))
            peak_elements = max(peak_elements, margin.numel())
        width = indices.shape[1]
        edge = active[sl, :, None] & active[sl, None, :] & valid[sl, None, None]
        edge &= ~torch.eye(width, device=local.device, dtype=torch.bool)[None]
        reject = (wide_margin > 0) & (local_margin < 0) & ~canonical[:, :, None] & edge[..., None]
        keep = alias_valid[:, :, None] & ~reject
        protected = canonical[:, :, None].expand_as(keep)
        source = (beta * selected).softmax(-1)[:, :, None].expand_as(keep)
        allocation = survivor_allocation(source, keep, protected, soft)
        remaining = keep.sum(-1).double()
        normalizer = (counts[:, :, None] / remaining).log()
        untouched = ((allocation == 1) | ~alias_valid[:, :, None]).all(-1)
        directed = torch.zeros_like(normalizer)
        for evidence, total, coefficients in cached:
            changed = (beta * evidence[:, :, :, None] + allocation.log()[:, None]).logsumexp(-1) / beta
            delta = (changed - total[..., None] + normalizer[:, None] / beta).masked_fill(untouched[:, None], 0.)
            directed += (delta * coefficients[..., None, None]).sum(1)
        directed.masked_fill_(~edge, 0.)
        requested = directed - directed.transpose(-1, -2)
        innovation = total_local - broad[sl].gather(1, indices).double()
        target = innovation[:, :, None] - innovation[:, None]
        action = target.sign() * torch.minimum(requested.abs(), target.abs()) * (requested * target > 0)
        weights = posterior[sl].gather(1, indices)
        contribution = (action * weights[:, None]).sum(-1)
        potential[sl].scatter_(1, indices, contribution)
        if diagnostics:
            eligible = alias_valid[:, :, None] & ~protected & edge[..., None]
            rejected += float((reject & eligible).sum())
            eligible_count += float(eligible.sum())
            action_sum += float(action.abs().sum())
            active_edges += float(edge.sum())
            mass_error = max(mass_error, float((allocation.sum(-1) - remaining).abs().max()))
            canonical_error = max(canonical_error, float((allocation[protected] - 1).abs().max()))
        saved.append(action)
    potential -= potential.mean(-1, keepdim=True)
    potential.masked_fill_(~valid[:, None], 0.)
    score = base + operator.double() @ potential
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError('Nonfinite bounded branch-union scores.')
    stats = dict(fine_forwards=0., additional_visual_forwards=0.,
        max_contenders_per_query=float(pairs.shape[1]),
        mean_contenders_per_valid_query=float(active[valid].sum(-1).double().mean()) if bool(valid.any()) else 0.,
        mean_absolute_action=action_sum / max(active_edges, 1.),
        rejected_noncanonical_fraction=rejected / max(eligible_count, 1.),
        survivor_mass_max_error=mass_error, canonical_weight_max_error=canonical_error,
        maximum_pair_alias_elements=float(peak_elements),
        mean_union_posterior_mass=float((posterior.gather(1, pairs) * active)[valid].sum(-1).mean()) if bool(valid.any()) else 1.) if diagnostics else {}
    return score, stats, dict(pairs=pairs, active=active, action=torch.cat(saved), potential=potential, base=base)
