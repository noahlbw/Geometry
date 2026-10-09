"""Retained observer salience mass replaces uniform survivor-count correction."""
import math

import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached, directed_cached
from .rival_competition_admission import (retained_competition_scores, sampled_classes,
    posterior_potential, project_actions, settings as competition_settings)
from .rival_margin_attenuation import BASE
from .rival_neutral_replacement import replacement_directed
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'geometry-frozen-rival-salience-mass-normalizer-v1-20261005'
PRIMARY = 'FineSalienceMass_Projected'
CANDIDATES = (PRIMARY, 'WideSalienceMass_Projected')
FIXED = 'HardFixedMass_Projected'
FAMILIES = {name: {'mean': name+'__ClassMean', 'strength': name+'__HardStrength',
    'shuffles': tuple(name+'__AliasShuffle'+str(i) for i in range(3))} for name in CANDIDATES}
METHODS = (*BASE, *CANDIDATES, FIXED, *(m for group in FAMILIES.values()
    for m in (group['mean'], group['strength'], *group['shuffles'])))


def settings():
    return {**competition_settings(), 'risk_support': 'original b>0,f<0; canonical/self/invalid protected',
        'writer': 'unchanged hard counterfactual plus retained observer salience-mass/count-mass difference',
        'wide_prior': 'log_softmax of existing per-class wide crop salience, no extra temperature',
        'fine_prior': 'original fine crop salience mixtures under original query crop/stencil coverage',
        'normalizer': '-log(sum of surviving per-class observer prior mass)/beta',
        'not_correctness_probabilities': True, 'salience_inside_profiled_logits_unchanged': True,
        'controls': 'each candidate has canonical-preserving class mean, matched hard strength and three own alias-prior permutations',
        'new_parameters': 0, 'additional_visual_forwards': 0, 'uniform_prior_recovers_count_rule': True}


def sampled_fine_log_prior(crops, observations, members, valid):
    if len(crops) != len(observations) or not crops: raise ValueError('Matching nonempty fine observations required.')
    terms = []
    coverage = torch.zeros(len(valid), dtype=torch.float64, device=valid.device)
    for crop, cached in zip(crops, observations):
        alpha = cached.coefficients.double().sum(1)
        coverage += alpha
        log_prior = crop.salience[members].double().log_softmax(-1)
        terms.append(alpha.log()[:, None, None]+log_prior[None])
    if bool((valid & (coverage <= 0)).any()): raise ValueError('Every valid query needs fine prior coverage.')
    log_prior = torch.stack(terms).logsumexp(0)
    log_prior[~valid] = -math.log(members.shape[-1])
    log_prior -= log_prior.logsumexp(-1, keepdim=True)
    if not bool(torch.isfinite(log_prior).all()): raise RuntimeError('Nonfinite sampled fine log prior.')
    return log_prior


def prior_normalizer(log_prior, keep, beta=CONFIG.beta):
    if (keep.ndim != 4 or keep.dtype != torch.bool or log_prior.ndim not in (2, 3) or beta <= 0
            or log_prior.shape[-2:] != keep.shape[1:3]
            or (log_prior.ndim == 3 and len(log_prior) != len(keep))
            or not bool(torch.isfinite(log_prior).all())
            or not torch.allclose(log_prior.logsumexp(-1), torch.zeros_like(log_prior[..., 0]), atol=1e-12, rtol=0)):
        raise ValueError('Normalized finite class/alias log prior and matching binary survival required.')
    prior = log_prior if log_prior.ndim == 3 else log_prior[None].expand(len(keep), -1, -1)
    mass = prior[..., None].masked_fill(~keep, -torch.inf).logsumexp(2)
    if not bool(torch.isfinite(mass).all()): raise ValueError('Each class/rival needs surviving prior mass.')
    return (-mass/beta).clamp_min(0).masked_fill(keep.all(2), 0.)


def prior_adjustment(observations, log_priors, members, risk, valid, beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (len(log_priors) != len(observations) or not observations
            or risk.shape != (len(valid), members.numel(), len(members)) or beta <= 0 or chunk < 1
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Matching priors, observations and bounded frozen risk required.')
    output = torch.zeros(len(valid), len(members), len(members), device=risk.device, dtype=torch.float64)
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        keep = risk[sl, members] == 0
        remaining = keep.sum(2).double()
        if bool((remaining == 0).any()): raise ValueError('Protected surviving aliases required.')
        count = (members.shape[-1]/remaining).log()/beta
        for crop, log_prior in zip(observations, log_priors):
            prior = log_prior[sl] if log_prior.ndim == 3 else log_prior
            changed = prior_normalizer(prior, keep, beta)
            output[sl] += (changed-count)*crop.coefficients[sl].double().sum(1)[:, None, None]
    return output.masked_fill(~valid[:, None, None], 0.)


def mean_noncanonical_prior(log_prior, members, canonical):
    value = log_prior.clone()
    for c, (row, word) in enumerate(zip(members, canonical)):
        slots = (row != word).nonzero().flatten()
        if len(slots):
            mean = log_prior[..., c, slots].logsumexp(-1)-math.log(len(slots))
            value[..., c, slots] = mean[..., None]
    return value


def shuffled_prior(log_prior, members, canonical, seed):
    permutation = alias_permutations(members, canonical, seed)[0]
    if log_prior.ndim == 3: permutation = permutation[None].expand_as(log_prior)
    return log_prior.gather(-1, permutation)


@torch.inference_mode()
def salience_mass_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                         coordinates, fine_coordinates, valid, members, canonical, parents,
                         image_size, methods=METHODS):
    if not set(methods).issubset(METHODS): raise ValueError('Undeclared salience-mass endpoint.')
    old_methods = tuple(m for m in methods if m in BASE)
    values, stats = {}, {}
    if old_methods:
        values, old_risk, stats = retained_competition_scores(local, operator, broad, wide, wide_count,
            fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
            image_size, methods=old_methods)
        if set(methods).issubset(BASE): return values, old_risk, stats
    observed = cache_observations(wide, wide_count, coordinates, image_size, members)
    fine_observed = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    b = sampled_cached(observed, coordinates)
    f = sampled_cached(fine_observed, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(b, f, f, parents, canonical, valid, CONFIG)
    if old_methods and not torch.equal(risk, old_risk): raise RuntimeError('Frozen salience-mass source changed.')
    hard = directed_cached(observed, members, risk, valid)
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    posterior = baseline.softmax(-1)
    innovation = sampled_classes(fine_observed, fine_coordinates).double()-broad.double()
    target = (innovation[..., None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    stats['canonical_risk_max'] = float(risk[:, canonical].abs().max())

    def reconcile(directed):
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, _ = posterior_potential(action, posterior, valid)
        return operator.double()@potential

    hard_delta = None
    if any(FAMILIES[name]['strength'] in methods for name in CANDIDATES):
        hard_delta = values[BASE[3]]-baseline if BASE[3] in values else reconcile(hard)
    if FIXED in methods:
        fixed = replacement_directed(observed, members, risk, valid, kind='fixed_mass')
        values[FIXED] = baseline+reconcile(fixed)
    for name in CANDIDATES:
        family = FAMILIES[name]
        names = (name, family['mean'], family['strength'], *family['shuffles'])
        if not set(methods).intersection(names): continue
        if name == PRIMARY:
            prior = sampled_fine_log_prior(fine, fine_observed, members, valid)
            log_priors = [prior]*len(observed)
        else: log_priors = [crop.salience[members].double().log_softmax(-1) for crop in wide]
        adjusted = prior_adjustment(observed, log_priors, members, risk, valid)
        delta = reconcile(hard+adjusted)
        values[name] = baseline+delta
        active = (risk[:, members] > 0).any(2)[valid]
        stats[name+'__normalizer_change_mean'] = float(adjusted[valid][active].mean()) if bool(active.any()) else 0.
        stats[name+'__larger_than_count_fraction'] = float((adjusted[valid][active] > 1e-12).double().mean()) if bool(active.any()) else 0.
        entropy = [-(p.exp()*p).sum(-1)/math.log(members.shape[-1]) for p in log_priors]
        stats[name+'__prior_entropy_fraction'] = float(torch.stack([v.mean() for v in entropy]).mean())
        if family['mean'] in methods:
            changed = [mean_noncanonical_prior(p, members, canonical) for p in log_priors]
            error = max(float((p.logsumexp(-1)-q.logsumexp(-1)).abs().max()) for p, q in zip(log_priors, changed))
            stats[family['mean']+'__mass_max_error'] = error
            values[family['mean']] = baseline+reconcile(hard+prior_adjustment(observed, changed, members, risk, valid))
        if family['strength'] in methods:
            scale = delta.norm()/hard_delta.norm().clamp_min(CONFIG.epsilon)
            values[family['strength']] = baseline+scale*hard_delta
            stats[family['strength']+'__norm_error'] = float((scale*hard_delta).norm()-delta.norm())
            stats[family['strength']+'__zero_direction'] = float(hard_delta.norm() == 0)
            stats[family['strength']+'__scale'] = float(scale)
        for i, shuffle in enumerate(family['shuffles']):
            if shuffle not in methods: continue
            changed = [shuffled_prior(p, members, canonical, CONFIG.random_seed+i) for p in log_priors]
            error = max(float((p.sort(-1).values-q.sort(-1).values).abs().max()) for p, q in zip(log_priors, changed))
            if error: raise RuntimeError('Prior alias-null changed normalized spectra.')
            stats[shuffle+'__spectrum_max_error'] = error
            values[shuffle] = baseline+reconcile(hard+prior_adjustment(observed, changed, members, risk, valid))
        if family['mean'] in methods:
            uniform = [torch.full_like(p, -math.log(members.shape[-1])) for p in log_priors]
            stats[name+'__uniform_count_adjustment_max'] = float(prior_adjustment(observed, uniform, members, risk, valid).abs().max())
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise RuntimeError('Nonfinite salience-mass score.')
    return values, risk, stats
