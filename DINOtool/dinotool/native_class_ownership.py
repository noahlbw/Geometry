"""Native near-family-excluded class reference for contextual alias ownership."""
from dataclasses import dataclass

import torch

from .competitive_ownership import alias_families
from .excess_alias_rejection import semantic_contradictions
from .shared_family_ownership import SharedOwnershipConfig
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-native-family-class-ownership-source-v1-20261003'


@dataclass(frozen=True)
class NativeOwnershipConfig:
    family_cosine: float = SharedOwnershipConfig().family_cosine
    beta: float = 1.
    family_chunk: int = 16
    epsilon: float = 1e-6


CONFIG = NativeOwnershipConfig()
FEASIBILITY_GATE = {'minimum_domains_separating_foreign_from_paraphrase': 5,
    'minimum_blind_spot_capture_fraction': .1, 'minimum_domains_capturing_blind_spot': 5,
    'exact_heldout_independence_required': True, 'pixel_metrics_required_for_promotion': True,
    'no_automatic_full_rollout': True}


def family_description(bank, config=CONFIG):
    bank.validate()
    families = alias_families(bank, config.family_cosine)
    excluded = torch.zeros(len(families), len(bank.alias_names), dtype=torch.bool, device=bank.features.device)
    assignments = torch.empty(len(bank.alias_names), dtype=torch.long, device=bank.features.device)
    for i, group in enumerate(families):
        excluded[i, group] = True
        assignments[group] = i
    members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
    canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                             for c in range(bank.class_count)])
    conflict = semantic_contradictions(bank.features, bank.parent_indices, canonical, config.epsilon)
    return families, excluded, assignments, members, canonical, conflict


def family_class_field(crops, count, coordinates, members, excluded, valid, config=CONFIG):
    if (excluded.ndim != 2 or excluded.shape[1] != members.numel() or excluded.dtype != torch.bool
            or coordinates.shape != (len(valid), 2) or valid.dtype != torch.bool or config.beta <= 0
            or config.family_chunk < 1):
        raise ValueError('Family masks, grouped aliases and query coordinates required.')
    kept = ~excluded[:, members]
    remaining = kept.sum(-1)
    known = remaining > 0
    field = torch.zeros(len(valid), len(excluded), len(members), device=coordinates.device, dtype=torch.float64)
    coverage = torch.zeros(len(valid), device=coordinates.device, dtype=torch.float32)
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, (512, 512))
        coverage += coeff.sum(-1)
        raw, salience = crop.alias_logits[:, members].double(), crop.salience[members].double()
        if not bool(torch.isfinite(raw).all() and torch.isfinite(salience).all()):
            raise ValueError('Finite cached alias observations required.')
        for start in range(0, len(excluded), config.family_chunk):
            sl = slice(start, start+config.family_chunk)
            keep = kept[sl]
            # Recompute only the held-out REFERENCE salience; the prediction observer is unchanged.
            masked = salience[None].masked_fill(~keep, -torch.inf)
            total = masked.logsumexp(-1, keepdim=True)
            weights = (masked-total.masked_fill(~known[sl, :, None], 0.)).exp()
            evidence = raw[:, None]*remaining[None, sl, :, None]*weights[None]
            score = (config.beta*evidence).masked_fill(~keep[None], -torch.inf).logsumexp(-1)/config.beta
            score -= remaining[sl].clamp_min(1).double().log()[None]/config.beta
            score = score.masked_fill(~known[None, sl], 0.)
            field[:, sl] += (score[ids]*coeff.double()[..., None, None]).sum(1)
    if bool(valid.any()) and float((coverage[valid]-1).abs().max()) > 1e-6:
        raise RuntimeError('Cached native query coverage differs.')
    field.masked_fill_(~valid[:, None, None], 0.)
    if not bool(torch.isfinite(field).all()):
        raise RuntimeError('Nonfinite family reference.')
    return field, known


def ownership_risk(field, known, assignments, parents, canonical, conflict, valid):
    n, _, classes = field.shape
    if (known.shape != field.shape[1:] or assignments.shape != parents.shape
            or conflict.shape != (len(parents), classes) or valid.shape != (n,)
            or not bool(torch.isfinite(field).all() and torch.isfinite(conflict).all())):
        raise ValueError('Matching finite class references and text attachment conflicts required.')
    reference = field[:, assignments]
    own = reference.gather(-1, parents[None, :, None].expand(n, -1, 1))
    contrast = ((reference-own)/2).tanh().clamp_min(0)
    observed = known[assignments] & known[assignments, parents][:, None]
    risk = contrast*conflict.double()[None]*observed[None]
    risk[:, canonical] = 0.
    risk.masked_fill_(~valid[:, None, None], 0.)
    risk.scatter_(-1, parents[None, :, None].expand(n, -1, 1), 0.)
    return risk


def risk_union(view_risk, attachment_risk):
    if view_risk.shape != attachment_risk.shape or not bool(torch.isfinite(view_risk).all() and torch.isfinite(attachment_risk).all()):
        raise ValueError('Matching finite bounded risks required.')
    if bool(((view_risk < 0) | (view_risk > 1) | (attachment_risk < 0) | (attachment_risk > 1)).any()):
        raise ValueError('Risks must lie in [0,1].')
    return (view_risk.double()+(1-view_risk.double())*attachment_risk.double()).clamp(0, 1).to(view_risk.dtype)
