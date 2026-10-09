"""Reciprocal scale contradiction, with no survivor-mass redistribution."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)


IMPLEMENTATION = 'geometry-bounded896-reciprocal-fixed-slot-alias-v1-20261006'
PRIMARY = 'ReciprocalAdmission_Soft'
OBSERVATION_MEAN = 'ReciprocalAdmission_ObservationMean'
ONE_SIDED_HARD = 'ReciprocalAdmission_WideHard'
ONE_SIDED_SOFT = 'ReciprocalAdmission_WideSoft'
SURVIVOR = 'ReciprocalAdmission_SurvivorSoft'
CLASS_MEAN = 'ReciprocalAdmission_ClassMean'
SHUFFLES = tuple('ReciprocalAdmission_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'ReciprocalAdmission_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, ONE_SIDED_HARD, ONE_SIDED_SOFT, SURVIVOR, CLASS_MEAN,
    *SHUFFLES, SHUFFLED_WRITE)
DIAGNOSTICS = ('fine_risk_mean', 'wide_risk_mean', 'positive_directed_delta_max',
               'mass_log_fallbacks', 'class_budget_max_error')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'reciprocal wide/fine query/class/rival contradiction; continuous1-risk, no G truth gate',
    'aggregation': 'fixed20-slot log mean exp; no survivor-mass compensation',
    'writer': 'equal positive wide/fine plus half each conditional potential through original H',
    'semantic_source': 'unchanged wide and bounded fine observations; no additional visual or text source',
    'controls': METHODS[4:], 'fitted_parameters': 0,
    'limitation': 'cross-scale agreement can be jointly wrong; contradiction is not semantic ground truth'}


def fixed_slot_action(observations, members, risk, valid, *, chunk=CONFIG.query_chunk):
    if (not observations or chunk < 1 or risk.shape != (len(valid), members.numel(), len(members))
            or valid.dtype != torch.bool or not bool(torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Matching bounded risks, validity and cached observations required.')
    output = torch.zeros((len(valid), len(members), len(members)), dtype=torch.float64, device=risk.device)
    fallbacks = 0
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        allocation = 1-risk[sl, members].double()
        if bool((allocation.sum(2) == 0).any()):
            raise ValueError('At least one positive alias weight required per comparison.')
        untouched = (allocation == 1).all(2)[:, None]
        for crop in observations:
            evidence = crop.evidence.double()[crop.indices[sl]]
            responsibility = evidence.softmax(-1)
            mass = torch.einsum('qsck,qckd->qscd', responsibility, allocation)
            underflow = mass == 0
            change = mass.clamp_min(torch.finfo(torch.float64).tiny).log()
            # Exact log-domain fallback only for genuinely underflowed retained mass.
            if bool(underflow.any()):
                stable = (evidence[..., None]+allocation.log()[:, None]).logsumexp(3)
                stable -= evidence.logsumexp(-1)[..., None]
                change = torch.where(underflow, stable, change)
                fallbacks += int(underflow.sum())
            change = change.masked_fill(untouched, 0.)
            output[sl] += (change*crop.coefficients[sl].double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    maximum = float(output.max())
    if maximum > 1e-12 or not bool(torch.isfinite(output).all()):
        raise RuntimeError('Fixed-slot attenuation must be finite and nonpositive before pair projection.')
    return output, dict(positive_directed_delta_max=maximum, mass_log_fallbacks=float(fallbacks))


def reciprocal_risks(wide, fine, parents, canonical, valid):
    first = native_risk(wide, fine, fine, parents, canonical, valid, CONFIG)
    second = native_risk(fine, wide, wide, parents, canonical, valid, CONFIG)
    return first, second


def continuous_controls(risk, members, canonical):
    controls = {CLASS_MEAN: risk.double().clone()}
    for group in members:
        ids = group[~torch.isin(group, canonical)]
        controls[CLASS_MEAN][:, ids] = risk[:, ids].double().mean(1, keepdim=True)
    for index, name in enumerate(SHUFFLES):
        generator = torch.Generator().manual_seed(CONFIG.random_seed+index)
        changed = risk.clone()
        for group in members:
            ids = group[~torch.isin(group, canonical)]
            permutation = torch.randperm(len(ids), generator=generator).to(risk.device)
            changed[:, ids] = risk[:, ids[permutation]]
        controls[name] = changed
    error = float((controls[CLASS_MEAN][:, members].sum(2)-risk[:, members].double().sum(2)).abs().max())
    if error > 1e-10:
        raise RuntimeError('Class-only control must preserve continuous risk mass for every pair.')
    return controls, error


@torch.inference_mode()
def reciprocal_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                      relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared reciprocal admission methods required.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    innovation = .5*(fine_field.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    values = {OBSERVATION_MEAN: positive} if OBSERVATION_MEAN in methods else {}
    stats = dict.fromkeys((*DIAGNOSTICS, 'canonical_risk_max', 'mean_absolute_admission_potential',
        'normal_equation_max_error', 'gauge_max_error', 'deleted_alias_rival_fraction'), 0.)
    stats['retained_count_mean'] = float(members.shape[1])
    if set(methods) == {OBSERVATION_MEAN}:
        return values, stats
    broad_margin = sampled_cached(wide, coordinates)
    fine_margin = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    wrisk, frisk = reciprocal_risks(broad_margin, fine_margin, parents, canonical, valid)

    def write(first, second):
        da, a = fixed_slot_action(wide, members, first, valid)
        db, b = fixed_slot_action(fine, members, second, valid)
        delta, consistency = signed_potential(.5*(da+db), valid)
        return delta, {**consistency,
            'positive_directed_delta_max': max(a['positive_directed_delta_max'], b['positive_directed_delta_max']),
            'mass_log_fallbacks': a['mass_log_fallbacks']+b['mass_log_fallbacks']}

    if PRIMARY in methods or SHUFFLED_WRITE in methods:
        potential, diagnostics = write(wrisk, frisk)
        stats.update({**risk_statistics(wrisk, potential, valid, members, canonical), **diagnostics,
            'fine_risk_mean': float(frisk[valid].double().mean()),
            'wide_risk_mean': float(wrisk[valid].double().mean()),
            'canonical_risk_max': float(torch.maximum(wrisk[:, canonical].abs().max(), frisk[:, canonical].abs().max()))})
        if PRIMARY in methods:
            values[PRIMARY] = positive+operator.double()@potential
        if SHUFFLED_WRITE in methods:
            values[SHUFFLED_WRITE] = positive+permuted_operator(operator, valid).double()@potential
    if ONE_SIDED_HARD in methods:
        delta, _ = signed_potential(directed_cached(wide, members, wrisk, valid), valid)
        values[ONE_SIDED_HARD] = positive+.5*(operator.double()@delta)
    if ONE_SIDED_SOFT in methods:
        directed, _ = fixed_slot_action(wide, members, wrisk, valid)
        delta, _ = signed_potential(directed, valid)
        values[ONE_SIDED_SOFT] = positive+.5*(operator.double()@delta)
    if SURVIVOR in methods:
        directed = .5*(directed_cached(wide, members, wrisk, valid, 'weighted')+
                        directed_cached(fine, members, frisk, valid, 'weighted'))
        delta, _ = signed_potential(directed, valid)
        values[SURVIVOR] = positive+operator.double()@delta
    if any(name in methods for name in (CLASS_MEAN, *SHUFFLES)):
        wcontrols, we = continuous_controls(wrisk, members, canonical)
        fcontrols, fe = continuous_controls(frisk, members, canonical)
        stats['class_budget_max_error'] = max(we, fe)
        for method in (CLASS_MEAN, *SHUFFLES):
            if method in methods:
                delta, _ = write(wcontrols[method], fcontrols[method])
                values[method] = positive+operator.double()@delta
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite reciprocal coupled scores.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=reciprocal_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
