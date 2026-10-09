"""Constrain competitive word writeback to its observed class-pair support."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS_MEAN,
    SHUFFLES as OLD_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)


IMPLEMENTATION = 'geometry-bounded896-evidence-local-competitive-write-v1-20261006'
PRIMARY = 'EvidenceLocal_Soft'
OBSERVATION_MEAN = 'EvidenceLocal_ObservationMean'
GLOBAL_SOFT = 'EvidenceLocal_GlobalSoft'
GLOBAL_HARD = 'EvidenceLocal_GlobalHard'
UNMATCHED = 'EvidenceLocal_Unmatched'
QUERY_ONLY = 'EvidenceLocal_QueryOnly'
DIRECT = 'EvidenceLocal_Direct'
CLASS_MEAN = 'EvidenceLocal_ClassMean'
SHUFFLES = tuple('EvidenceLocal_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'EvidenceLocal_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, GLOBAL_SOFT, GLOBAL_HARD, UNMATCHED, QUERY_ONLY, DIRECT,
    CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE)
DIAGNOSTICS = ('class_budget_max_error',
    'write_norm_relative_error', 'write_scale', 'inactive_class_displacement_max',
    'pair_support_fraction', 'unrestricted_spill_power_fraction',
    'positive_directed_delta_max', 'mass_log_fallbacks')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'unchanged one-sided profiled wide-positive/fine-negative continuous1-risk',
    'writer': 'H transports each antisymmetric pair, then observed pair support restricts output; class projection follows',
    'support': 'either direction has positive alias risk; source-derived before masks, not semantic truth',
    'strength': 'one valid-patch class-field Frobenius norm matched to original global soft intervention',
    'positive_observation': 'unchanged equal positive wide/fine observation through original H',
    'aggregation': 'unchanged fixed20-slot LME without survivor redistribution',
    'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'controls': METHODS[4:],
    'limitation': 'token-logit intervention support, not native-pixel probability fidelity or certified word correctness'}


def pair_support(risk, members, valid):
    if (risk.shape != (len(valid), members.numel(), len(members))
            or valid.dtype != torch.bool or not bool(torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Finite bounded risk and actual class membership required.')
    directed = (risk[:, members] > 0).any(2)
    support = (directed | directed.transpose(-1, -2)) & valid[:, None, None]
    return support.masked_fill(torch.eye(len(members), device=risk.device, dtype=torch.bool)[None], False)


def match_norm(correction, reference):
    target, actual = reference.double().norm(), correction.double().norm()
    if float(target) == 0:
        return torch.zeros_like(correction), dict(write_norm_relative_error=0., write_scale=0.)
    if float(actual) == 0:
        raise RuntimeError('Nonzero reference intervention has no supported write direction.')
    scale = target/actual
    changed = correction*scale
    error = float((changed.norm()-target).abs()/target)
    if error > 1e-10 or not bool(torch.isfinite(changed).all()):
        raise RuntimeError('Whole-field intervention norm matching failed.')
    return changed, dict(write_norm_relative_error=error, write_scale=float(scale))


def supported_write(margins, operator, support, valid, reference=None, *, matched=True):
    n, c = margins.shape[:2]
    if (margins.shape != (n, c, c) or operator.shape != (n, n)
            or support.shape != margins.shape or support.dtype != torch.bool
            or valid.shape != (n,) or valid.dtype != torch.bool
            or not bool(torch.isfinite(margins).all() and torch.isfinite(operator).all())
            or not torch.allclose(margins, -margins.transpose(-1, -2), atol=1e-12, rtol=0)
            or not torch.equal(support, support.transpose(-1, -2))
            or bool(support[~valid].any()) or bool(margins[~valid].any())
            or bool(support.diagonal(dim1=-2, dim2=-1).any())):
        raise ValueError('Finite antisymmetric action and symmetric valid pair support required.')
    i, j = torch.triu_indices(c, c, 1, device=margins.device)
    # Transport upper-triangle edges once, then project only supported arrivals.
    transported = operator.double()@margins[:, i, j].double()
    kept = transported.masked_fill(~support[:, i, j], 0.)/c
    changed = torch.zeros((n, c), dtype=torch.float64, device=margins.device)
    changed.index_add_(1, i, kept)
    changed.index_add_(1, j, -kept)
    changed.masked_fill_(~valid[:, None], 0.)
    if reference is None:
        reference = operator.double()@margins.double().mean(-1)
    if reference.shape != changed.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError('Matching finite class-field norm reference required.')
    diagnostics = dict(write_norm_relative_error=0., write_scale=1.)
    if matched:
        changed, diagnostics = match_norm(changed, reference)
    inactive = ~support.any(-1)
    leak = float(changed.masked_select(inactive).abs().max()) if bool(inactive.any()) else 0.
    gauge = float(changed.sum(-1).abs().max())
    if leak != 0 or gauge > 1e-10:
        raise RuntimeError('Inactive class fidelity or competitive class gauge failed.')
    return changed, {**diagnostics, 'inactive_class_displacement_max': leak, 'gauge_max_error': gauge}


@torch.inference_mode()
def local_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                 relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared evidence-local writeback methods required.')
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
    risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    directed, diagnostics = fixed_slot_action(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    margins = directed-directed.transpose(-1, -2)
    support = pair_support(risk, members, valid)
    reference = operator.double()@potential
    stats.update({**risk_statistics(risk, potential, valid, members, canonical), **diagnostics, **consistency})
    if GLOBAL_SOFT in methods:
        values[GLOBAL_SOFT] = positive+.5*reference
    if GLOBAL_HARD in methods:
        hard, _ = signed_potential(directed_cached(wide, members, risk, valid), valid)
        values[GLOBAL_HARD] = positive+.5*(operator.double()@hard)
    needs_primary = bool(set(methods).intersection((PRIMARY, UNMATCHED, QUERY_ONLY, DIRECT, SHUFFLED_WRITE)))
    if needs_primary:
        delta, extra = supported_write(margins, operator, support, valid, reference)
        stats.update(extra)
        if PRIMARY in methods:
            values[PRIMARY] = positive+.5*delta
        unrestricted = reference.masked_fill(~valid[:, None], 0.)
        outside = unrestricted.masked_fill(support.any(-1), 0.)
        stats['unrestricted_spill_power_fraction'] = float(outside.square().sum()/unrestricted.square().sum().clamp_min(1e-30))
    upper = torch.ones((len(members), len(members)), device=valid.device, dtype=torch.bool).triu(1)
    stats['pair_support_fraction'] = float(support[valid][:, upper].double().mean()) if bool(valid.any()) else 0.
    if UNMATCHED in methods:
        delta, _ = supported_write(margins, operator, support, valid, reference, matched=False)
        values[UNMATCHED] = positive+.5*delta
    if QUERY_ONLY in methods:
        query = support.any((-1, -2))[:, None, None] & (~torch.eye(len(members), device=valid.device, dtype=torch.bool))[None]
        delta, _ = supported_write(margins, operator, query, valid, reference)
        values[QUERY_ONLY] = positive+.5*delta
    if DIRECT in methods:
        delta, _ = supported_write(margins, torch.eye(len(valid), device=valid.device), support, valid, reference)
        values[DIRECT] = positive+.5*delta
    if SHUFFLED_WRITE in methods:
        delta, _ = supported_write(margins, permuted_operator(operator, valid), support, valid, reference)
        values[SHUFFLED_WRITE] = positive+.5*delta
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, OLD_CLASS_MEAN), *zip(SHUFFLES, OLD_SHUFFLES)):
            if new not in methods:
                continue
            changed = controls[old]
            if not torch.equal(pair_support(changed, members, valid), support):
                raise RuntimeError('Word control changed observed pair support.')
            action, _ = fixed_slot_action(wide, members, changed, valid)
            delta, _ = supported_write(action-action.transpose(-1, -2), operator, support, valid, reference)
            values[new] = positive+.5*delta
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite evidence-local writeback scores.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=local_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
