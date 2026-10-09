"""Treat mixed fine interpolation footprints as inconclusive negative witnesses."""
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


IMPLEMENTATION = 'geometry-bounded896-footprint-negative-witness-v1-20261006'
PRIMARY = 'FootprintWitness_RivalSoft'
OBSERVATION_MEAN = 'FootprintWitness_ObservationMean'
OLD_SOFT = 'FootprintWitness_PreviousSoft'
OLD_HARD = 'FootprintWitness_PreviousHard'
GUARD_ONLY = 'FootprintWitness_GuardOnly'
MATCHED_OLD = 'FootprintWitness_MatchedPrevious'
CLASS_MEAN = 'FootprintWitness_ClassMean'
SHUFFLES = tuple('FootprintWitness_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'FootprintWitness_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN, *SHUFFLES,
    SHUFFLED_WRITE, GUARD_ONLY)
DIAGNOSTICS = ('previous_risk_mean', 'witness_risk_mean', 'risk_support_change_fraction',
    'risk_support_expansion_fraction', 'risk_increase_max', 'unknown_source_risk_max',
    'fully_negative_implication_violation', 'mixed_eligible_fraction', 'footprint_width_mean',
    'class_budget_max_error', 'positive_directed_delta_max', 'mass_log_fallbacks',
    'matched_previous_norm_relative_error', 'matched_previous_unmatchable', 'matched_previous_scale')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'wide advantage requires every positive-coefficient fine stencil margin to be negative',
    'witness': 'actual interpolation contributor interval, not Geometry semantic truth or pseudo labels',
    'negative_magnitude': 'negative upper endpoint; unchanged P/(P+N) attenuation and canonical protection',
    'unknown': 'mixed sign or missing complete fine coverage stays neutral',
    'aggregation': 'unchanged outside-exponent fixed20-slot attenuation; no survivor redistribution',
    'writer': 'unchanged positive wide/fine information and half word potential through original H',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'all-negative footprints can still be jointly wrong; bounds are not statistical confidence intervals'}


def footprint_interval(observations, valid):
    if not observations or valid.dtype != torch.bool:
        raise ValueError('Existing fine observations and Boolean validity required.')
    shape = (len(valid), *observations[0].margins.shape[1:])
    first = observations[0].margins
    lower = first.new_full(shape, torch.inf)
    upper = first.new_full(shape, -torch.inf)
    mass = torch.zeros(len(valid), dtype=torch.float64, device=valid.device)
    for crop in observations:
        if (crop.margins.ndim != 3 or crop.margins.shape[1:] != shape[1:]
                or crop.indices.ndim != 2 or crop.indices.shape[0] != len(valid)
                or crop.indices.dtype != torch.long or crop.coefficients.shape != crop.indices.shape
                or not bool(torch.isfinite(crop.margins).all() and torch.isfinite(crop.coefficients).all())
                or bool((crop.coefficients < 0).any()) or bool((crop.indices < 0).any())
                or bool((crop.indices >= len(crop.margins)).any())):
            raise ValueError('Finite matching actual fine stencils required.')
        samples = crop.margins[crop.indices]
        active = crop.coefficients > 0
        lower = torch.minimum(lower, samples.masked_fill(~active[..., None, None], torch.inf).amin(1))
        upper = torch.maximum(upper, samples.masked_fill(~active[..., None, None], -torch.inf).amax(1))
        mass += crop.coefficients.double().sum(-1)
    complete = valid & ((mass-1).abs() <= 1e-5)
    lower = lower.masked_fill(~complete[:, None, None], 0.)
    upper = upper.masked_fill(~complete[:, None, None], 0.)
    return lower, upper, complete


def footprint_risks(wide, mean, upper, complete, parents, canonical, valid):
    previous = native_risk(wide, mean, mean, parents, canonical, valid, CONFIG)
    bounded = native_risk(wide, upper, upper, parents, canonical, complete, CONFIG)
    bounded = bounded.masked_fill(previous <= 0, 0.)
    supported = complete[:, None, None] & (upper < 0) & (previous > 0)
    guard = previous.masked_fill(~supported, 0.)
    return bounded, previous, guard


@torch.inference_mode()
def footprint_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                     relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared footprint-witness methods required.')
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
    lower, upper, complete = footprint_interval(fine, valid)
    risk, previous, guard = footprint_risks(wm, fm, upper, complete, parents, canonical, valid)
    old_write = None
    if any(m in methods for m in (OLD_SOFT, MATCHED_OLD)):
        directed, _ = fixed_slot_action(wide, members, previous, valid)
        potential, _ = signed_potential(directed, valid)
        old_write = operator.double()@potential
        if OLD_SOFT in methods:
            values[OLD_SOFT] = positive+.5*old_write
    if OLD_HARD in methods:
        potential, _ = signed_potential(directed_cached(wide, members, previous, valid), valid)
        values[OLD_HARD] = positive+.5*(operator.double()@potential)
    needs_witness = bool(set(methods).intersection((PRIMARY, MATCHED_OLD, CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE)))
    if needs_witness:
        directed, diagnostics = fixed_slot_action(wide, members, risk, valid)
        potential, consistency = signed_potential(directed, valid)
        written = operator.double()@potential
        if PRIMARY in methods:
            values[PRIMARY] = positive+.5*written
        if MATCHED_OLD in methods:
            matched, extra = match_previous(old_write, written, valid)
            values[MATCHED_OLD] = positive+.5*matched
            stats.update(extra)
        if SHUFFLED_WRITE in methods:
            values[SHUFFLED_WRITE] = positive+.5*(permuted_operator(operator, valid).double()@potential)
        stats.update({**risk_statistics(risk, potential, valid, members, canonical), **diagnostics, **consistency})
    if GUARD_ONLY in methods:
        directed, _ = fixed_slot_action(wide, members, guard, valid)
        potential, _ = signed_potential(directed, valid)
        values[GUARD_ONLY] = positive+.5*(operator.double()@potential)
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, OLD_CLASS_MEAN), *zip(SHUFFLES, OLD_SHUFFLES)):
            if new in methods:
                directed, _ = fixed_slot_action(wide, members, controls[old], valid)
                potential, _ = signed_potential(directed, valid)
                values[new] = positive+.5*(operator.double()@potential)
    if bool(valid.any()):
        eligible = (previous > 0) & valid[:, None, None]
        mixed = eligible & (upper >= 0) & complete[:, None, None]
        stats.update(previous_risk_mean=float(previous[valid].double().mean()),
            witness_risk_mean=float(risk[valid].double().mean()),
            risk_support_change_fraction=float(((risk[valid] > 0) != (previous[valid] > 0)).double().mean()),
            risk_support_expansion_fraction=float(((risk[valid] > 0) & (previous[valid] == 0)).double().mean()),
            risk_increase_max=float((risk-previous).clamp_min(0).max()),
            unknown_source_risk_max=float(risk.masked_fill(complete[:, None, None], 0.).abs().max()),
            fully_negative_implication_violation=float(((risk > 0) & (upper >= 0)).double().sum()),
            mixed_eligible_fraction=float(mixed.double().sum()/eligible.double().sum().clamp_min(1)),
            footprint_width_mean=float((upper[complete]-lower[complete]).double().mean()) if bool(complete.any()) else 0.)
    if stats['risk_increase_max'] > 1e-5 or not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Invalid footprint-bounded attenuation.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=footprint_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
