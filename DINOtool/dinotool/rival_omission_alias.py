"""Judge negative alias evidence against a rival's single-word counterfactuals."""
import math

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS_MEAN,
    SHUFFLES as OLD_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-rival-word-omission-witness-v1-20261006'
PRIMARY = 'RivalOmission_RivalSoft'
OBSERVATION_MEAN = 'RivalOmission_ObservationMean'
OLD_SOFT = 'RivalOmission_PreviousSoft'
OLD_HARD = 'RivalOmission_PreviousHard'
GUARD_ONLY = 'RivalOmission_GuardOnly'
REFERENCE_SHUFFLE = 'RivalOmission_ReferenceShuffle'
MATCHED_OLD = 'RivalOmission_MatchedPrevious'
CLASS_MEAN = 'RivalOmission_ClassMean'
SHUFFLES = tuple('RivalOmission_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'RivalOmission_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN,
    *SHUFFLES, SHUFFLED_WRITE, GUARD_ONLY, REFERENCE_SHUFFLE)
DIAGNOSTICS = ('previous_risk_mean', 'witness_risk_mean', 'risk_support_change_fraction',
    'risk_support_expansion_fraction', 'risk_increase_max', 'unknown_source_risk_max',
    'omission_changed_eligible_fraction', 'rival_omission_shift_mean', 'class_budget_max_error',
    'positive_directed_delta_max', 'mass_log_fallbacks', 'matched_previous_norm_relative_error',
    'matched_previous_unmatchable', 'matched_previous_scale')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'negative fine evidence must survive every single-word omission from the rival class',
    'witness': 'one identical rival word omitted across every contributing fine token/crop; no truth teacher',
    'negative_magnitude': 'worst surviving rival mean, unchanged P/(P+N) and canonical prediction protection',
    'reference_normalizer': 'K-1 for each counterfactual; prediction stays fixed20 with no redistribution',
    'reference_words': 'all rival words including canonical; omission probes are not prediction deletion',
    'writer': 'unchanged positive wide/fine information and half word potential through original H',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'redundant wrong words can survive; a useful discriminative rival word can dominate legitimately'}


def omission_deltas(evidence):
    if evidence.ndim != 3 or evidence.shape[-1] < 2 or not bool(torch.isfinite(evidence).all()):
        raise ValueError('Finite [token,class,alias] evidence with at least two aliases required.')
    values = evidence.double()
    edge = torch.full_like(values[..., :1], -torch.inf)
    left = torch.cat((edge, values[..., :-1].logcumsumexp(-1)), -1)
    right = torch.cat((values[..., 1:].flip(-1).logcumsumexp(-1).flip(-1), edge), -1)
    remaining = torch.logaddexp(left, right)
    k = values.shape[-1]
    result = remaining-values.logsumexp(-1, keepdim=True)+math.log(k/(k-1))
    uniform = values.amax(-1, keepdim=True) == values.amin(-1, keepdim=True)
    return result.masked_fill(uniform, 0.)


def rival_omission_shift(observations, valid):
    if not observations or valid.ndim != 1 or valid.dtype != torch.bool:
        raise ValueError('Existing fine observations and Boolean query validity required.')
    first = observations[0].evidence
    if first.ndim != 3:
        raise ValueError('Grouped original fine evidence required.')
    changes = first.new_zeros((len(valid), *first.shape[1:]), dtype=torch.float64)
    mass = torch.zeros(len(valid), dtype=torch.float64, device=valid.device)
    for crop in observations:
        if (crop.evidence.ndim != 3 or crop.evidence.shape[1:] != first.shape[1:]
                or crop.indices.ndim != 2 or crop.indices.shape[0] != len(valid)
                or crop.indices.dtype != torch.long or crop.coefficients.shape != crop.indices.shape
                or not bool(torch.isfinite(crop.coefficients).all()) or bool((crop.coefficients < 0).any())
                or bool((crop.indices < 0).any()) or bool((crop.indices >= len(crop.evidence)).any())):
            raise ValueError('Matching finite original interpolation stencils required.')
        delta = omission_deltas(crop.evidence)
        changes += (delta[crop.indices]*crop.coefficients.double()[..., None, None]).sum(1)
        mass += crop.coefficients.double().sum(-1)
    complete = valid & ((mass-1).abs() <= 1e-5)
    # Minimize AFTER interpolation so one actual word is omitted throughout the footprint.
    shift = changes.amin(-1).clamp_max(0.)
    shift.masked_fill_(~complete[:, None], 0.)
    return shift, complete


def omission_risks(wide, fine, shift, complete, parents, canonical, valid):
    if (shift.shape != (len(valid), fine.shape[-1]) or complete.shape != valid.shape
            or complete.dtype != torch.bool or not bool(torch.isfinite(shift).all())
            or bool((shift > 0).any())):
        raise ValueError('Nonpositive rival omission shifts and complete coverage required.')
    previous = native_risk(wide, fine, fine, parents, canonical, valid, CONFIG)
    robust = fine.double()-shift[:, None, :]
    risk = native_risk(wide.double(), robust, robust, parents, canonical, complete, CONFIG)
    risk.masked_fill_(previous <= 0, 0.)
    guard = previous.masked_fill(risk <= 0, 0.)
    return risk, previous, guard


@torch.inference_mode()
def omission_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                    relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared rival-omission methods required.')
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
    wm = sampled_cached(wide, coordinates)
    fm = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    shift, complete = rival_omission_shift(fine, valid)
    risk, previous, guard = omission_risks(wm, fm, shift, complete, parents, canonical, valid)

    def write(source):
        directed, diagnostics = fixed_slot_action(wide, members, source, valid)
        potential, consistency = signed_potential(directed, valid)
        return operator.double()@potential, potential, {**diagnostics, **consistency}

    old_write = None
    if any(m in methods for m in (OLD_SOFT, MATCHED_OLD)):
        old_write, _, _ = write(previous)
        if OLD_SOFT in methods:
            values[OLD_SOFT] = positive+.5*old_write
    if OLD_HARD in methods:
        potential, _ = signed_potential(directed_cached(wide, members, previous, valid), valid)
        values[OLD_HARD] = positive+.5*(operator.double()@potential)
    if set(methods).intersection((PRIMARY, MATCHED_OLD, CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE)):
        written, potential, diagnostics = write(risk)
        if PRIMARY in methods:
            values[PRIMARY] = positive+.5*written
        if MATCHED_OLD in methods:
            matched, extra = match_previous(old_write, written, valid)
            values[MATCHED_OLD] = positive+.5*matched
            stats.update(extra)
        if SHUFFLED_WRITE in methods:
            values[SHUFFLED_WRITE] = positive+.5*(permuted_operator(operator, valid).double()@potential)
        stats.update({**risk_statistics(risk, potential, valid, members, canonical), **diagnostics})
    if GUARD_ONLY in methods:
        written, _, _ = write(guard)
        values[GUARD_ONLY] = positive+.5*written
    if REFERENCE_SHUFFLE in methods:
        generator = torch.Generator().manual_seed(CONFIG.random_seed)
        permutation = torch.randperm(shift.shape[-1], generator=generator).to(shift.device)
        shuffled, _, _ = omission_risks(wm, fm, shift[:, permutation], complete, parents, canonical, valid)
        written, _, _ = write(shuffled)
        values[REFERENCE_SHUFFLE] = positive+.5*written
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, OLD_CLASS_MEAN), *zip(SHUFFLES, OLD_SHUFFLES)):
            if new in methods:
                written, _, _ = write(controls[old])
                values[new] = positive+.5*written
    if bool(valid.any()):
        eligible = (previous > 0) & valid[:, None, None]
        withdrawn = eligible & (risk <= 0) & complete[:, None, None]
        stats.update(previous_risk_mean=float(previous[valid].double().mean()),
            witness_risk_mean=float(risk[valid].double().mean()),
            risk_support_change_fraction=float(((risk[valid] > 0) != (previous[valid] > 0)).double().mean()),
            risk_support_expansion_fraction=float(((risk[valid] > 0) & (previous[valid] == 0)).double().mean()),
            risk_increase_max=float((risk-previous).clamp_min(0).max()),
            unknown_source_risk_max=float(risk.masked_fill(complete[:, None, None], 0.).abs().max()),
            omission_changed_eligible_fraction=float(withdrawn.double().sum()/eligible.double().sum().clamp_min(1)),
            rival_omission_shift_mean=float((-shift[complete]).mean()) if bool(complete.any()) else 0.)
    if stats['risk_increase_max'] > 1e-5 or not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Invalid rival-omission attenuation.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=omission_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
