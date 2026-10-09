"""Alias-contribution mass admits fine innovation into the frozen reconstruction."""
import torch

from .fine_alias_view import CONFIG
from .rival_alias_fast import cached_risk, cache_observations
from .rival_competition_admission import (retained_competition_scores, sampled_classes,
    posterior_potential, settings as competition_settings)
from .rival_alias_attribution import alias_null


IMPLEMENTATION = 'geometry-frozen-alias-mass-fine-admission-v1-20261005'
BASE = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', 'FineBudgetOnly_Exact')
PRIMARY = 'DirectedMassFine_Exact'
CANDIDATES = (PRIMARY, 'PairMassFine_Exact', 'DirectedMassAligned_Exact')
CONTROLS = ('ClassMeanMassFine', 'CountMassFine', 'CountMassStrengthMatched')
SHUFFLES = tuple('MassAliasShuffle'+str(i) for i in range(3))
METHODS = (*BASE, *CANDIDATES, *CONTROLS, *SHUFFLES)


def settings():
    return {**competition_settings(), 'mass': 'original wide logsumexp responsibilities on risk-positive aliases',
        'writer': 'fine-minus-wide class innovation admitted in proportion to alias-specific rejected evidence mass',
        'directed_center': 'remove posterior-weighted class mean of fine-minus-wide innovation',
        'pair_alternative': 'max of the two endpoint rejected masses times fine pair innovation',
        'aligned_alternative': 'accept directed mass action only if aligned with original hard pair action',
        'new_parameters': 0, 'zero_mass_fallback': 'unchanged no-admission coupling',
        'no_global_word_blacklist': True, 'standalone_primary_hard_writer_required': False}


def rejected_mass(observations, risk, members, valid, beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (risk.shape != (len(valid), members.numel(), len(members)) or beta <= 0 or chunk < 1
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Matching bounded risks and positive reader settings required.')
    result = torch.zeros(len(valid), len(members), len(members), dtype=torch.float64, device=risk.device)
    for crop in observations:
        responsibility = (beta*crop.evidence).double().softmax(-1)
        for start in range(0, len(valid), chunk):
            sl = slice(start, start+chunk)
            sampled = responsibility[crop.indices[sl]]
            rejected = (risk[sl, members] > 0).double()
            mass = torch.einsum('qsck,qckd->qscd', sampled, rejected)
            result[sl] += (mass*crop.coefficients[sl].double()[..., None, None]).sum(1)
    result.masked_fill_(~valid[:, None, None], 0.)
    if not bool(torch.isfinite(result).all()) or bool(((result < -1e-6) | (result > 1+1e-6)).any()):
        raise RuntimeError('Rejected logsumexp mass is not bounded.')
    return result.clamp(0, 1)


def mass_actions(mass, innovation, posterior, valid, kind='directed'):
    if (mass.shape != (len(valid), innovation.shape[-1], innovation.shape[-1])
            or posterior.shape != innovation.shape or kind not in ('directed', 'pair')
            or not bool(torch.isfinite(mass).all() and torch.isfinite(innovation).all())
            or bool(((mass < 0) | (mass > 1)).any())):
        raise ValueError('Bounded pair mass and matching finite class innovation required.')
    if kind == 'directed':
        centered = innovation-(posterior*innovation).sum(-1, keepdim=True)
        directed = mass*centered[..., None]
        action = directed-directed.transpose(-1, -2)
    else:
        symmetric = torch.maximum(mass, mass.transpose(-1, -2))
        action = symmetric*(innovation[..., None]-innovation[:, None])
    return action.masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def mass_admission_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                          coordinates, fine_coordinates, valid, members, canonical, parents,
                          image_size, methods=METHODS):
    if not set(methods).issubset(METHODS): raise ValueError('Undeclared mass-admission endpoint.')
    original_methods = tuple(m for m in methods if m in BASE)
    values, stats = {}, {}
    if original_methods:
        required = tuple(dict.fromkeys((*original_methods, 'RivalFineHard_Exact'))) if CANDIDATES[2] in methods else original_methods
        values, risk, stats = retained_competition_scores(local, operator, broad, wide, wide_count,
            fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
            image_size, methods=required)
        observed = cache_observations(wide, wide_count, coordinates, image_size, members)
    else:
        risk, observed = cached_risk(wide, wide_count, fine, fine_count, coordinates, fine_coordinates,
                                    valid, members, canonical, parents, image_size)
    if set(methods).issubset(BASE): return values, risk, stats
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    posterior = baseline.softmax(-1)
    fine_observed = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    innovation = sampled_classes(fine_observed, fine_coordinates).double()-broad.double()
    mass = rejected_mass(observed, risk, members, valid)
    action = mass_actions(mass, innovation, posterior, valid)
    primary_potential, _ = posterior_potential(action, posterior, valid)
    primary_delta = operator.double()@primary_potential
    values[PRIMARY] = baseline+primary_delta
    stats.update(rejected_alias_mass_mean=float(mass[valid].mean()),
                 canonical_risk_max=float(risk[:, canonical].abs().max()))
    if CANDIDATES[1] in methods:
        potential, _ = posterior_potential(mass_actions(mass, innovation, posterior, valid, 'pair'), posterior, valid)
        values[CANDIDATES[1]] = baseline+operator.double()@potential
    if CANDIDATES[2] in methods:
        from .rival_alias_fast import directed_cached
        old = directed_cached(observed, members, risk, valid)
        old = old-old.transpose(-1, -2)
        aligned = action*(action*old > 0)
        potential, _ = posterior_potential(aligned, posterior, valid)
        values[CANDIDATES[2]] = baseline+operator.double()@potential
    if CONTROLS[0] in methods:
        average = mass.sum(-1)/(len(members)-1)
        changed = average[..., None].expand_as(mass).clone()
        changed.diagonal(dim1=-2, dim2=-1).zero_()
        potential, _ = posterior_potential(mass_actions(changed, innovation, posterior, valid), posterior, valid)
        values[CONTROLS[0]] = baseline+operator.double()@potential
    if any(m in methods for m in CONTROLS[1:]):
        counts = (risk[:, members] > 0).double().mean(2)
        potential, _ = posterior_potential(mass_actions(counts, innovation, posterior, valid), posterior, valid)
        delta = operator.double()@potential
        if CONTROLS[1] in methods: values[CONTROLS[1]] = baseline+delta
        if CONTROLS[2] in methods:
            scale = primary_delta.norm()/delta.norm().clamp_min(CONFIG.epsilon)
            values[CONTROLS[2]] = baseline+delta*scale
            stats['count_strength_norm_error'] = float((delta*scale).norm()-primary_delta.norm())
            stats['count_strength_scale'] = float(scale)
            stats['count_strength_zero_direction'] = float(delta.norm() == 0)
    for i, m in enumerate(SHUFFLES):
        if m not in methods: continue
        changed = alias_null(risk, members, canonical, CONFIG.random_seed+i)
        error = int(((changed[:, members] > 0).sum(2)-(risk[:, members] > 0).sum(2)).abs().max())
        if error: raise RuntimeError('Mass null changed per-query/class/rival deletion counts.')
        null_mass = rejected_mass(observed, changed, members, valid)
        potential, _ = posterior_potential(mass_actions(null_mass, innovation, posterior, valid), posterior, valid)
        values[m] = baseline+operator.double()@potential
        stats[m+'__count_max_error'] = error
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise RuntimeError('Nonfinite mass-admission score.')
    return values, risk, stats
