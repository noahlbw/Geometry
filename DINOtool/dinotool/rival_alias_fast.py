"""Cached execution of the retained reader and opt-in soft admission writers."""
from dataclasses import dataclass

import torch

from .calibrated_competitive_alias import profiled_logits
from .fine_alias_view import CONFIG
from .matched_contribution_alias import contribution_margins
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'frozen-geometry-rival-alias-cached-soft-v1-20261005'
SOFT_METHODS = ('FineSoft_Weighted', 'FineSoft_Excess', 'ReuseSoft_Weighted',
                'ReuseSoft_Excess', 'ReuseClassMean_Weighted')
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', *SOFT_METHODS)


@dataclass(frozen=True)
class CachedCrop:
    evidence: torch.Tensor
    margins: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor


def cache_observations(crops, count, coordinates, image_size, members, beta=CONFIG.beta):
    output = []
    for crop in crops:
        evidence = profiled_logits(crop, members)
        indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
        output.append(CachedCrop(evidence, contribution_margins(evidence, beta), indices, coefficients))
    return output


def sampled_cached(observations, coordinates):
    if not observations:
        raise ValueError('At least one observation is required.')
    first = observations[0]
    output = coordinates.new_zeros((len(coordinates), *first.margins.shape[1:]))
    for crop in observations:
        output += (crop.margins[crop.indices] * crop.coefficients[..., None, None]).sum(1)
    return output


def directed_cached(observations, members, risk, valid, writer='hard',
                    beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (writer not in ('hard', 'weighted', 'excess') or beta <= 0 or chunk < 1
            or risk.shape != (len(valid), members.numel(), len(members))
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Bounded matching risk and a declared positive-temperature writer required.')
    dtype = risk.dtype if writer == 'excess' else torch.float64
    output = torch.zeros((len(valid), len(members), len(members)), dtype=dtype, device=risk.device)
    k = members.shape[-1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start + chunk)
        grouped = risk[sl, members]
        if writer == 'hard':
            keep = grouped == 0
            remaining = keep.sum(2)
            if bool((remaining == 0).any()):
                raise ValueError('Every pair needs a surviving alias.')
            normalizer = (k / remaining.double()).log()[:, None]
        elif writer == 'weighted':
            weights = 1 - grouped.double()
            remaining = weights.sum(2)
            if bool((remaining == 0).any()):
                raise ValueError('Every pair needs positive alias mass.')
            log_weights = weights.log()[:, None]
            normalizer = (k / remaining).log()[:, None]
        for crop in observations:
            ids, coefficients = crop.indices[sl], crop.coefficients[sl]
            sampled = crop.evidence[ids]
            if writer == 'excess':
                excess = ((beta * sampled).softmax(-1) - 1 / k).clamp_min(0)
                rejected = torch.einsum('qsck,qckd->qscd', excess, grouped)
                delta = torch.log1p(-rejected.clamp_max(1 - 1 / k)) / beta
                # Preserve the historical soft writer's float32 arithmetic.
                output[sl] += (delta * coefficients[..., None, None]).sum(1)
            else:
                evidence = (beta * sampled).double()
                full = evidence.logsumexp(-1)
                if writer == 'hard':
                    changed = evidence[..., None].masked_fill(~keep[:, None], -torch.inf).logsumexp(3)
                    untouched = keep.all(2)[:, None]
                else:
                    changed = (evidence[..., None] + log_weights).logsumexp(3)
                    untouched = (grouped == 0).all(2)[:, None]
                delta = (changed - full[..., None] + normalizer) / beta
                delta = delta.masked_fill(untouched, 0.)
                output[sl] += (delta * coefficients.double()[..., None, None]).sum(1)
    return output.masked_fill(~valid[:, None, None], 0.).double()


def risk_statistics(risk, potential, valid, members, canonical):
    rivals = ~torch.eye(len(members), device=risk.device, dtype=torch.bool)
    retained = (risk[valid][:, members] == 0).sum(2)[:, rivals]
    return {'retained_count_mean': float(retained.double().mean()) if retained.numel() else 20.,
            'deleted_alias_rival_fraction': float((retained < members.shape[1]).double().mean()) if retained.numel() else 0.,
            'canonical_risk_max': float(risk[:, canonical].abs().max()),
            'mean_absolute_admission_potential': float(potential.abs().mean())}


@torch.inference_mode()
def cached_risk(wide, wide_count, fine, fine_count, coordinates, fine_coordinates,
                valid, members, canonical, parents, image_size):
    wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    full = sampled_cached(wide, coordinates)
    native = sampled_cached(fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(full, native, native, parents, canonical, valid, CONFIG)
    return risk, wide


@torch.inference_mode()
def retained_scores_fast(local, operator, broad, wide, wide_count, fine, fine_count,
                         coordinates, fine_coordinates, valid, members, canonical, parents, image_size):
    risk, wide = cached_risk(wide, wide_count, fine, fine_count, coordinates, fine_coordinates,
                             valid, members, canonical, parents, image_size)
    directed = directed_cached(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    values = {'Geometry': local.double(), 'NoAdmission_Exact': baseline,
              'RivalFineHard_Exact': baseline + operator.double() @ potential}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Nonfinite cached reader scores.')
    return values, risk, {**risk_statistics(risk, potential, valid, members, canonical), **consistency}


def class_mean_risk(risk, members, canonical):
    result = risk.clone()
    for group in members:
        noncanonical = group[~torch.isin(group, canonical)]
        result[:, noncanonical] = risk[:, noncanonical].mean(1, keepdim=True)
    result[:, canonical] = 0.
    return result


@torch.inference_mode()
def soft_scores(local, operator, broad, wide, wide_count, coordinates, valid, members,
                canonical, parents, image_size, local_aliases, fine_risk=None, methods=SOFT_METHODS,
                observations=None):
    if (not set(methods).issubset(SOFT_METHODS) or local_aliases.shape != (len(valid), members.numel())
            or not bool(torch.isfinite(local_aliases).all())):
        raise ValueError('Declared soft methods and finite already-computed local alias scores required.')
    if observations is None:
        observations = cache_observations(wide, wide_count, coordinates, image_size, members)
    if any(m.startswith('Reuse') for m in methods):
        full = sampled_cached(observations, coordinates)
        # Local cosine/.07 matches the original Geometry class log-mean-exp units.
        native = contribution_margins(local_aliases[:, members] / .07)
        reused = native_risk(full, native, native, parents, canonical, valid, CONFIG)
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    values, diagnostics = {}, {}
    for method in methods:
        if method.startswith('Fine'):
            if fine_risk is None:
                raise ValueError('Fine-risk methods require the existing fine observation.')
            risk = fine_risk
        else:
            risk = class_mean_risk(reused, members, canonical) if method.startswith('ReuseClass') else reused
        writer = 'excess' if method.endswith('Excess') else 'weighted'
        directed = directed_cached(observations, members, risk, valid, writer)
        potential, consistency = signed_potential(directed, valid)
        values[method] = baseline + operator.double() @ potential
        if not bool(torch.isfinite(values[method]).all()):
            raise RuntimeError('Nonfinite soft scores: ' + method)
        active = risk[valid][:, members]
        rivals = ~torch.eye(len(members), device=risk.device, dtype=torch.bool)
        diagnostics[method] = {**consistency,
            'canonical_risk_max': float(risk[:, canonical].abs().max()),
            'mean_absolute_admission_potential': float(potential.abs().mean()),
            'effective_alias_mass_mean': float((1 - active.double()).sum(2)[:, rivals].mean()),
            'active_alias_rival_fraction': float((active > 0).double().mean()),
            'risk_mean': float(active.double().mean())}
    return values, diagnostics
