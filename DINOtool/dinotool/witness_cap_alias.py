"""Cap disputed wide alias evidence by its existing fine likelihood contrast."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS_MEAN,
    SHUFFLES as OLD_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)


IMPLEMENTATION = 'geometry-bounded896-witness-likelihood-cap-v1-20261006'
PRIMARY = 'WitnessCap_RivalSoft'
OBSERVATION_MEAN = 'WitnessCap_ObservationMean'
OLD_SOFT = 'WitnessCap_PreviousSoft'
OLD_HARD = 'WitnessCap_PreviousHard'
MATCHED_OLD = 'WitnessCap_MatchedPrevious'
CLASS_MEAN = 'WitnessCap_ClassMean'
SHUFFLES = tuple('WitnessCap_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'WitnessCap_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN,
    *SHUFFLES, SHUFFLED_WRITE)
DIAGNOSTICS = ('previous_risk_mean', 'witness_risk_mean', 'risk_support_change_fraction',
    'class_budget_max_error', 'positive_directed_delta_max', 'mass_log_fallbacks',
    'matched_previous_norm_relative_error', 'matched_previous_unmatchable',
    'matched_previous_scale')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'same wide-positive/fine-negative slots; retention exp(fine alias-minus-rival margin)',
    'witness': 'unchanged profiled fine evidence and pooled rival; broad amplitude only chooses support',
    'aggregation': 'fixed20 slots, outside-exponent attenuation, no survivor redistribution',
    'writer': 'unchanged positive wide/fine observation and half word potential through original H',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'likelihood contrast is not semantic truth; shared visual/text errors can survive'}


def witness_risk(wide, fine, parents, canonical, valid):
    previous = native_risk(wide, fine, fine, parents, canonical, valid, CONFIG)
    # The existing margins are log alias-to-rival-mean exponential evidence ratios.
    risk = torch.where(previous > 0, -torch.expm1(fine.double().clamp_max(0)), 0.)
    return risk, previous


def match_previous(previous, reference, valid):
    if (previous.shape != reference.shape or previous.ndim != 2 or valid.shape != previous.shape[:1]
            or valid.dtype != torch.bool or not bool(torch.isfinite(previous).all() and torch.isfinite(reference).all())):
        raise ValueError('Matching finite class-field interventions required.')
    old_power = previous[valid].double().square().sum()
    new_power = reference[valid].double().square().sum()
    unmatchable = bool(old_power == 0 and new_power > 0)
    scale = torch.sqrt(new_power/old_power) if bool(old_power > 0) else old_power.new_tensor(1.)
    matched = previous.double()*scale
    error = float((matched[valid].square().sum()-new_power).abs()/new_power.clamp_min(1e-30))
    return matched, dict(matched_previous_norm_relative_error=error,
        matched_previous_unmatchable=float(unmatchable), matched_previous_scale=float(scale))


@torch.inference_mode()
def witness_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                   relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared witness-cap methods required.')
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
    risk, previous = witness_risk(wm, fm, parents, canonical, valid)
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
        stats.update({**risk_statistics(risk, potential, valid, members, canonical),
                      **diagnostics, **consistency})
        if bool(valid.any()):
            stats.update(previous_risk_mean=float(previous[valid].double().mean()),
                witness_risk_mean=float(risk[valid].mean()),
                risk_support_change_fraction=float(((risk[valid] > 0) != (previous[valid] > 0)).double().mean()))
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, OLD_CLASS_MEAN), *zip(SHUFFLES, OLD_SHUFFLES)):
            if new in methods:
                directed, _ = fixed_slot_action(wide, members, controls[old], valid)
                potential, _ = signed_potential(directed, valid)
                values[new] = positive+.5*(operator.double()@potential)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite witness-cap prediction.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=witness_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
