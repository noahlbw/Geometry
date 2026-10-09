"""Distinguish an existing word-derived dose from deployable class calibration."""
import torch

from . import one_sided_alias_audit as audit
from .native_alias_noise import signed_potential
from .supported_positive_alias import predict_image as source_predict
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-existing-alias-budget-audit-v1-20261006'
OLD_NAMES = {name: name.replace('OneSide_', 'BudgetAudit_') for name in audit.METHODS[3:]}
NEW_NAMES = {new: old for old, new in OLD_NAMES.items()}
PRIMARY = OLD_NAMES[audit.PRIMARY]
OBSERVATION_MEAN = OLD_NAMES[audit.OBSERVATION_MEAN]
POOLED = 'BudgetAudit_PooledClass'
MATCHED_POOLED = 'BudgetAudit_MatchedPooledClass'
METHODS = (*audit.METHODS[:3], *OLD_NAMES.values(), POOLED, MATCHED_POOLED)
DIAGNOSTICS = (*audit.DIAGNOSTICS, 'pooled_class_risk_mean',
    'pooled_word_write_norm_ratio', 'pooled_word_write_direction_cosine',
    'pooled_zero_with_nonzero_word_write', 'pooled_matched_unmatchable',
    'pooled_matched_norm_relative_error')
PROTOCOL = {**audit.PROTOCOL,
    'alias_admission': 'EXACT existing one-sided soft rule; additional class-calibration controls only',
    'control_source': 'already-computed wide/fine class logits, no primary word risk or inherited dose',
    'class_control': 'wide-positive/fine-negative class margin; N/(P+N); uniform noncanonical attenuation',
    'class_control_execution': 'analytic canonical-plus-noncanonical retained mass; no per-alias risk',
    'controls': METHODS[4:], 'new_primary_equations': False,
    'interpretation': 'developed96 attribution audit, not selector promotion',
    'matching': 'matched pooled control inherits WORD-ONLY primary post-H norm; diagnostic only'}


def pooled_class_risk(broad, fine, valid):
    if (broad.ndim != 2 or fine.shape != broad.shape or broad.shape[1] < 2
            or valid.shape != broad.shape[:1] or valid.dtype != torch.bool
            or not bool(torch.isfinite(broad).all() and torch.isfinite(fine).all())):
        raise ValueError('Matching finite class observations and validity required.')
    wide_margin = broad.double()[:, :, None]-broad.double()[:, None, :]
    fine_margin = fine.double()[:, :, None]-fine.double()[:, None, :]
    positive, negative = wide_margin.clamp_min(0), (-fine_margin).clamp_min(0)
    risk = torch.where((wide_margin > 0) & (fine_margin < 0),
        negative/(positive+negative).clamp_min(audit.CONFIG.epsilon), 0.)
    return risk.masked_fill(~valid[:, None, None], 0.)


def uniform_class_action(observations, members, canonical, risk, valid):
    classes, slots = members.shape
    if (not observations or slots < 2 or risk.shape != (len(valid), classes, classes)
            or canonical.shape != (classes,) or valid.dtype != torch.bool
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())
            or not bool((members == canonical[:, None]).sum(1).eq(1).all())):
        raise ValueError('Finite class risks and exactly one canonical slot per class required.')
    positions = (members == canonical[:, None]).long().argmax(1)
    is_canonical = torch.arange(slots, device=members.device)[None] == positions[:, None]
    result = torch.zeros_like(risk, dtype=torch.float64)
    for crop in observations:
        evidence = crop.evidence.double()[crop.indices]
        total = evidence.logsumexp(-1)
        canonical_log = evidence.gather(-1,
            positions[None, None, :, None].expand(*evidence.shape[:-1], 1)).squeeze(-1)-total
        other_log = evidence.masked_fill(is_canonical[None, None], -torch.inf).logsumexp(-1)-total
        # All noncanonical slots have the same weight: their mass can be read once.
        change = torch.logaddexp(canonical_log[..., None],
            torch.log1p(-risk)[:, None]+other_log[..., None])
        change.masked_fill_(risk[:, None] == 0, 0.)
        result += (change*crop.coefficients.double()[..., None, None]).sum(1)
    result.masked_fill_(~valid[:, None, None], 0.)
    if not bool(torch.isfinite(result).all()) or float(result.max()) > 1e-12:
        raise RuntimeError('Class calibration must be finite and nonpositive before projection.')
    return result


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared unchanged-rule budget audit methods required.')
    args = (local, operator, broad, fine_field, wide, fine, coordinates,
        relation, valid, members, canonical, parents)
    requested = tuple(NEW_NAMES[name] for name in methods if name in NEW_NAMES)
    if MATCHED_POOLED in methods and audit.PRIMARY not in requested:
        requested = (*requested, audit.PRIMARY, audit.OBSERVATION_MEAN)
    values, stats, old = {}, dict.fromkeys(DIAGNOSTICS, 0.), {}
    if requested:
        old, diagnostics = audit.scores(*args, methods=requested)
        stats.update(diagnostics)
        values.update({OLD_NAMES[name]: value for name, value in old.items() if OLD_NAMES[name] in methods})
    if set(methods) & {POOLED, MATCHED_POOLED}:
        baseline = local.double()+operator.double()@(broad.double()-local.double())
        innovation = .5*(fine_field.double()-broad.double())
        innovation.masked_fill_(~valid[:, None], 0.)
        positive = baseline+operator.double()@innovation
        risk = pooled_class_risk(broad, fine_field, valid)
        potential, _ = signed_potential(uniform_class_action(wide, members, canonical, risk, valid), valid)
        written = .5*(operator.double()@potential)
        if POOLED in methods:
            values[POOLED] = positive+written
        stats['pooled_class_risk_mean'] = float(risk[valid].mean()) if bool(valid.any()) else 0.
        if audit.PRIMARY in old:
            primary_positive = old.get(audit.OBSERVATION_MEAN, positive)
            word_write = old[audit.PRIMARY]-primary_positive
            power = word_write[valid].double().square().sum()
            own_power = written[valid].double().square().sum()
            stats['pooled_word_write_norm_ratio'] = float((own_power/power.clamp_min(1e-30)).sqrt())
            stats['pooled_word_write_direction_cosine'] = float(
                (word_write[valid]*written[valid]).sum()/(power*own_power).sqrt().clamp_min(1e-30))
            stats['pooled_zero_with_nonzero_word_write'] = float(bool(own_power == 0 and power > 0))
            if MATCHED_POOLED in methods:
                matched, info = match_previous(written, word_write, valid)
                values[MATCHED_POOLED] = positive+matched
                stats['pooled_matched_unmatchable'] = info['matched_previous_unmatchable']
                stats['pooled_matched_norm_relative_error'] = info['matched_previous_norm_relative_error']
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite budget-audit prediction.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
