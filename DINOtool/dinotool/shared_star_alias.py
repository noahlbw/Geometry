"""Fixed-budget alias attenuation on an anchored bidirectional star graph.

Reuses SharedLocal's complete local semantic observation and original H.
Only selected edges are observed; unobserved pairs are not zero constraints.
"""
from dataclasses import dataclass
from collections import defaultdict
import math
from types import FunctionType

import torch

from . import shared_local_alias as source
from .calibrated_competitive_alias import profiled_logits
from .fine_alias_view import CONFIG
from .stratified_soft_alias import crop_stencil

IMPLEMENTATION = 'geometry-shared-star-fixedslot-alias-v1-20261008'
PRIMARY = 'SharedStar2_Soft'
BUDGET4 = 'SharedStar4_Soft'
POOLED = 'SharedStar2_PooledClass'
SHUFFLED = 'SharedStar2_AliasShuffle'
BASELINE, MEAN = source.BASELINE, source.MEAN
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, MEAN, PRIMARY, BUDGET4, POOLED, SHUFFLED)
PROTOCOL = {**source.PROTOCOL,
    'implementation': IMPLEMENTATION, 'primary_rivals': 2, 'sensitivity_rivals': 4,
    'competition': 'unmodified observation-mean top1 center and next2/4 rivals; bidirectional star',
    'projection': 'minimum-norm exact star edge constraints; graph-external class corrections zero',
    'semantic_rule': 'same continuous one-sided N/(P+N), canonical protection, fixed-slot mass',
    'maximum_alias_risk_shape': '[query,2*rivals,20]',
    'forbidden_tensors': '[query,all_aliases,all_classes] and [query,classes,classes]',
    'equivalent_to_dense': False, 'alias_shuffle_seed': 20261008}


@dataclass(frozen=True)
class StarCrop:
    evidence: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor


def cache_observations(crops, count, coordinates, image_size, members, beta=CONFIG.beta):
    if beta != 1.:
        raise ValueError('Frozen beta1 contract required.')
    return tuple(StarCrop(profiled_logits(crop, members),
        *crop_stencil(crop, count, coordinates, image_size)) for crop in crops)


def graph(scores, budget):
    if scores.ndim != 2 or budget not in (2, 4) or scores.shape[1] < 2:
        raise ValueError('Multiclass scores and predeclared rival budget required.')
    size = min(budget + 1, scores.shape[1])
    nodes = scores.argsort(dim=-1, descending=True, stable=True)[:, :size]
    center = nodes[:, :1].expand(-1, size-1)
    rivals = nodes[:, 1:]
    return nodes, torch.cat((center, rivals), 1), torch.cat((rivals, center), 1)


def project_star(nodes, directed, classes):
    """Minimum norm under observed differences; no dense graph or solver."""
    rivals = nodes.shape[1]-1
    if directed.shape != (len(nodes), 2*rivals):
        raise ValueError('Both directions of each observed star edge required.')
    margin = directed[:, :rivals].double()-directed[:, rivals:].double()
    center = margin.sum(-1, keepdim=True)/(rivals+1)
    values = torch.cat((center, center-margin), 1)
    result = torch.zeros(len(nodes), classes, dtype=torch.float64, device=directed.device)
    result.scatter_(1, nodes, values)
    return result, margin


def edge_fields(observations, local, edge_from, edge_to):
    """Preserve crop-logmean-before-stencil ordering, gathering only edges."""
    rows = torch.arange(len(local), device=local.device)[:, None]
    slots = local.shape[-1]
    local_mean = local.logsumexp(-1)-math.log(slots)
    local_margin = local[rows, edge_from]-local_mean[rows, edge_to][..., None]
    wide_margin = torch.zeros_like(local_margin)
    wide_class_margin = local.new_zeros(edge_from.shape)
    cached = []
    for crop in observations:
        values = crop.evidence[crop.indices[:, :, None], edge_from[:, None]]
        mean = crop.evidence.logsumexp(-1)-math.log(slots)
        rival = mean[crop.indices[:, :, None], edge_to[:, None]]
        own = mean[crop.indices[:, :, None], edge_from[:, None]]
        coefficient = crop.coefficients
        wide_margin += ((values-rival[..., None])*coefficient[..., None, None]).sum(1)
        wide_class_margin += ((own-rival)*coefficient[..., None]).sum(1)
        cached.append((values.double(), coefficient.double()))
    class_local = local_mean[rows, edge_from]-local_mean[rows, edge_to]
    return wide_margin, local_margin, wide_class_margin, class_local, cached


def edge_risk(wide, local, canonical, valid):
    advantage = wide.clamp_min(0)
    negative = (-local).clamp_min(0)
    eligible = (wide > 0) & (local < 0) & ~canonical & valid[:, None, None]
    return torch.where(eligible, negative/(advantage+negative).clamp_min(CONFIG.epsilon), 0.)


def edge_action(cached, risk, valid):
    """Same fixed-slot mass, with stable log-domain handling of underflow."""
    result = torch.zeros(risk.shape[:2], dtype=torch.float64, device=risk.device)
    allocation = 1-risk.double()
    untouched = (risk == 0).all(-1)
    for values, coefficient in cached:
        mass = (values.softmax(-1)*allocation[:, None]).sum(-1)
        change = mass.clamp_min(torch.finfo(torch.float64).tiny).log()
        underflow = mass == 0
        if bool(underflow.any()):
            stable = (values+allocation.log()[:, None]).logsumexp(-1)-values.logsumexp(-1)
            change = torch.where(underflow, stable, change)
        change.masked_fill_(untouched[:, None], 0.)
        result += (change*coefficient[..., None]).sum(1)
    result.masked_fill_(~valid[:, None], 0.)
    return result


def shuffle_risk(risk, edge_from, canonical_mask):
    generator = torch.Generator().manual_seed(PROTOCOL['alias_shuffle_seed'])
    permutations = torch.arange(canonical_mask.shape[1], device=risk.device).expand_as(canonical_mask).clone()
    for c in range(len(canonical_mask)):
        indices = (~canonical_mask[c]).nonzero().flatten()
        order = torch.randperm(len(indices), generator=generator).to(risk.device)
        permutations[c, indices] = indices[order]
    return risk.gather(-1, permutations[edge_from])


@torch.inference_mode()
def scores(local, operator, broad, evidence, observations, coordinates, valid,
           members, canonical, parents, methods):
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    witness = evidence.logsumexp(-1)
    innovation = .5*(witness.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    output = {MEAN: positive} if MEAN in methods else {}
    stats = dict(canonical_risk_max=0., mean_absolute_admission_potential=0.,
        normal_equation_max_error=0., gauge_max_error=0., retained_count_mean=20.,
        deleted_alias_rival_fraction=0., maximum_risk_elements=0.,
        maximum_directed_edges=0., positive_directed_delta_max=0.)
    canonical_mask = members == canonical[:, None]
    for budget, names in ((2, (PRIMARY, POOLED, SHUFFLED)), (4, (BUDGET4,))):
        requested = tuple(name for name in names if name in methods)
        if not requested:
            continue
        nodes, edge_from, edge_to = graph(positive, budget)
        wm, lm, wc, lc, cached = edge_fields(observations, evidence, edge_from, edge_to)
        protected = canonical_mask[edge_from]
        risk = edge_risk(wm, lm, protected, valid)
        for name in requested:
            use = risk
            if name == POOLED:
                use = edge_risk(wc[..., None].expand_as(risk), lc[..., None].expand_as(risk), protected, valid)
            elif name == SHUFFLED:
                use = shuffle_risk(risk, edge_from, canonical_mask)
            directed = edge_action(cached, use, valid)
            delta, margin = project_star(nodes, directed, local.shape[1])
            output[name] = positive+.5*(operator.double()@delta)
            if name == PRIMARY:
                selected = delta.gather(1, nodes)
                error = (selected[:, :1]-selected[:, 1:]-margin).abs().max()
                stats.update(canonical_risk_max=float(use[protected].abs().max()),
                    mean_absolute_admission_potential=float(delta.abs().mean()),
                    normal_equation_max_error=float(error),gauge_max_error=float(delta.sum(-1).abs().max()),
                    retained_count_mean=float((20-use.sum(-1))[valid].mean()),
                    deleted_alias_rival_fraction=float((use[valid] > 0).float().mean()),
                    maximum_risk_elements=float(use.numel()),maximum_directed_edges=float(use.shape[1]),
                    positive_directed_delta_max=float(directed.max()))
    if not all(bool(torch.isfinite(value).all()) for value in output.values()):
        raise RuntimeError('Nonfinite sparse star prediction.')
    if max(stats['canonical_risk_max'], stats['normal_equation_max_error'], stats['gauge_max_error']) > 1e-10:
        raise RuntimeError('Canonical or star constraint failed.')
    if stats['positive_directed_delta_max'] > 1e-12:
        raise RuntimeError('Fixed-slot action became positive.')
    return output, stats


# Reuse the frozen source observation/stitching body with process-local globals.
# Source module globals are never changed, including when inference raises.
_scope = dict(source.predict_image.__wrapped__.__globals__, METHODS=METHODS,
              shared_scores=scores, cache_observations=cache_observations,
              dict=lambda **fields: defaultdict(float, fields))
_predict = FunctionType(source.predict_image.__wrapped__.__code__, _scope,
                        'shared_star_observation', source.predict_image.__wrapped__.__defaults__)
_predict.__kwdefaults__ = source.predict_image.__wrapped__.__kwdefaults__


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    return _predict(*args, methods=methods, **kwargs)
