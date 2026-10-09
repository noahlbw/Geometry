"""Rival-conditioned contextual alias evidence with bounded joint class actions."""
from dataclasses import dataclass

import torch

from .calibrated_competitive_alias import profiled_logits
from .stratified_soft_alias import crop_stencil
from .target_context_alias import TargetContextConfig


IMPLEMENTATION = 'geometry-pair-conditioned-context-reader-v1-20261003'
PRIMARY = 'PairContext_Exact'
METHODS = ('Geometry', 'Anchored_Exact', 'TargetContext_Exact', 'MeanLogit_Original', PRIMARY,
    'PairContext_MeanLogit', 'CapacityMean_Exact', *('SpatialShuffle'+str(i)+'_Exact' for i in range(3)),
    'ShuffledSupport_PairExact', 'SingleFill0_PairExact', 'SingleFill1_PairExact', 'TextOnly_PairExact')


@dataclass(frozen=True)
class PairContextConfig:
    query_chunk: int = 128
    maximum_iterations: int = 256
    solver_tolerance: float = 1e-10


def pair_risk(full, kept, removed, parents, canonical, supported, source=TargetContextConfig(), fill=None):
    if (kept.shape != removed.shape or kept.shape != (2, *full.shape) or full.ndim != 2
            or parents.shape != full.shape[1:] or supported.shape != full.shape[:1]
            or fill not in (None, 0, 1)):
        raise ValueError('Matched two-fill alias observations and query validity required.')
    if not bool(torch.isfinite(full).all() and torch.isfinite(kept).all() and torch.isfinite(removed).all()):
        raise ValueError('Finite observations required.')
    contextual = (full[None]-kept).clamp_min(0)/((full[None]-kept).abs()+(full[None]-removed).abs()).clamp_min(source.epsilon)
    p = (kept[:, :, canonical]/source.canonical_temperature).softmax(-1)
    own, rivals = p[:, :, parents, None], p[:, :, None]
    leakage = (rivals-own).clamp_min(0)/(rivals+own).clamp_min(source.epsilon)
    per_fill = contextual[..., None]*leakage
    risk = per_fill.amin(0) if fill is None else per_fill[fill]
    risk[:, canonical] = 0
    risk.masked_fill_(~supported[:, None, None], 0)
    return risk.clamp(0, 1)


def pair_observation(crops, count, coordinates, image_size, members, risk, valid, beta=1., chunk=128):
    if (risk.shape != (len(valid), members.numel(), len(members)) or beta <= 0
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Bounded alias/rival risks required.')
    directed = risk.new_zeros((len(valid), len(members), len(members)))
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        grouped = risk[sl, members]
        for crop in crops:
            indices, coefficients = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = profiled_logits(crop, members)[indices]
            excess = ((beta*evidence).softmax(-1)-1/members.shape[-1]).clamp_min(0)
            rejected = torch.einsum('qsck,qckd->qscd', excess, grouped)
            delta = torch.log1p(-rejected.clamp_max(1-1/members.shape[-1]))/beta
            directed[sl] += (delta*coefficients[..., None, None]).sum(1)
    directed.masked_fill_(~valid[:, None, None], 0)
    strength = risk[:, members].amax(2)
    weight = torch.maximum(strength, strength.transpose(-1, -2))
    return directed-directed.transpose(-1, -2), weight, directed


def reconcile(margin, weight, cap, valid, config=PairContextConfig()):
    n, classes = cap.shape
    if (margin.shape != (n, classes, classes) or weight.shape != margin.shape or valid.shape != (n,)
            or not bool(torch.isfinite(margin).all() and torch.isfinite(weight).all() and torch.isfinite(cap).all())
            or bool((cap < 0).any()) or bool(((weight < 0) | (weight > 1)).any())
            or not torch.allclose(margin, -margin.transpose(-1, -2), atol=1e-7, rtol=0)
            or not torch.equal(weight, weight.transpose(-1, -2))):
        raise ValueError('Antisymmetric margins, symmetric bounded weights and capacities required.')
    with torch.autocast(device_type=margin.device.type, enabled=False):
        w, e, bounds = weight.double().clone(), margin.double(), cap.double().masked_fill(~valid[:, None], 0)
        ids = torch.arange(classes, device=margin.device)
        w[:, ids, ids] = 0
        w.masked_fill_(~valid[:, None, None], 0)
        diagonal = 1+w.sum(-1)
        right = (w*e).sum(-1)
        x = torch.zeros_like(bounds)
        residual = float('inf')
        # Standard cyclic box-coordinate descent; I+L(w) is positive definite.
        for iteration in range(1, config.maximum_iterations+1):
            for c in range(classes):
                candidate = (right[:, c]+(w[:, c]*x).sum(-1))/diagonal[:, c]
                x[:, c] = torch.maximum(candidate.clamp_max(0), -bounds[:, c])
            gradient = diagonal*x-(w @ x[..., None])[..., 0]-right
            projected = torch.maximum((x-gradient).clamp_max(0), -bounds)
            residual = float((x-projected).abs().max())
            if residual <= config.solver_tolerance:
                break
        if residual > config.solver_tolerance:
            raise RuntimeError('Joint class action did not reach its numerical KKT tolerance.')
        realized = x[:, :, None]-x[:, None]
        mismatch = ((realized-e).square()*w).sum()*.5
        requested = (e.square()*w).sum()*.5
        objective = .5*x.square().sum()+.5*mismatch
        if objective > .5*requested+1e-10:
            raise RuntimeError('Joint action increased its objective over identity.')
        stats = {'iterations': iteration, 'projected_kkt_max_error': residual,
            'requested_pair_energy': float(requested), 'remaining_pair_energy': float(mismatch),
            'objective': float(objective), 'identity_objective': float(.5*requested),
            'mean_suppression': float(x[valid].abs().mean()) if bool(valid.any()) else 0.,
            'capacity_saturation_fraction': float(((x <= -bounds+config.solver_tolerance) & (bounds > 0))[valid].double().mean()) if bool(valid.any()) else 0.,
            'active_pair_fraction': float((w[valid] > 0).double().mean()) if bool(valid.any()) else 0.}
    return x, stats
