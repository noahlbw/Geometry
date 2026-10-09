"""Condition alias use on its own class evidence with that word excluded."""
import torch

from . import alias_budget_attribution as budget
from . import one_sided_alias_audit as audit
from .rival_omission_alias import omission_deltas
from .supported_positive_alias import predict_image as source_predict
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-own-word-excluded-class-evidence-v1-20261006'
OLD_NAMES = {name: name.replace('OneSide_', 'OwnPeer_') for name in audit.METHODS[3:]}
NEW_NAMES = {new: old for old, new in OLD_NAMES.items()}
PRIMARY = OLD_NAMES[audit.PRIMARY]
OBSERVATION_MEAN = OLD_NAMES[audit.OBSERVATION_MEAN]
PREVIOUS_SOFT = 'OwnPeer_PreviousSoft'
PREVIOUS_HARD = 'OwnPeer_PreviousHard'
POOLED = 'OwnPeer_PooledClass'
MATCHED_POOLED = 'OwnPeer_MatchedPooledClass'
METHODS = (*audit.METHODS[:3], *OLD_NAMES.values(), PREVIOUS_SOFT,
    PREVIOUS_HARD, POOLED, MATCHED_POOLED)
DIAGNOSTICS = (*audit.DIAGNOSTICS, 'peer_omission_abs_mean', 'peer_coverage_max_error',
    'peer_unknown_risk_max', 'peer_joint_positive_action_fraction',
    'peer_full_class_positive_action_fraction', 'pooled_matched_norm_relative_error',
    'pooled_matched_unmatchable')
PROTOCOL = {**audit.PROTOCOL,
    'alias_admission': 'wide class positive; own fine class with tested word excluded must be negative against rival',
    'retention': '1-N/(P+N); P=wide class margin, N=negative own-peer fine margin; canonical/self/unknown protected',
    'reference': 'same word omitted from every actual fine contributor; K-1 mean, frozen original salience',
    'prediction': 'all20 remain; original fixed slots, no survivor redistribution',
    'controls': METHODS[4:], 'new_primary_equations': True,
    'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'novelty_boundary': 'leave-word-out and weighting have precedents; test this source, not a new theorem',
    'limitation': 'peer support is not semantic truth; legitimate isolated subtype words can be suppressed'}


def own_omission_shift(observations, members, valid):
    if not observations or valid.ndim != 1 or valid.dtype != torch.bool or members.ndim != 2:
        raise ValueError('Existing grouped fine observations, members and validity required.')
    changes = torch.zeros((len(valid), *members.shape), dtype=torch.float64, device=valid.device)
    mass = torch.zeros(len(valid), dtype=torch.float64, device=valid.device)
    for crop in observations:
        if (crop.evidence.ndim != 3 or crop.evidence.shape[1:] != members.shape
                or crop.indices.shape != crop.coefficients.shape or crop.indices.shape[0] != len(valid)
                or not bool(torch.isfinite(crop.coefficients).all()) or bool((crop.coefficients < 0).any())):
            raise ValueError('Matching original fine contributor stencils required.')
        changes += (omission_deltas(crop.evidence)[crop.indices]*
            crop.coefficients.double()[..., None, None]).sum(1)
        mass += crop.coefficients.double().sum(-1)
    error = float((mass[valid]-1).abs().max()) if bool(valid.any()) else 0.
    known = valid & ((mass-1).abs() <= 1e-5)
    shifts = torch.zeros((len(valid), members.numel()), dtype=torch.float64, device=valid.device)
    # Omit one identical word across contributors, then interpolate; not pixelwise word switching.
    shifts[:, members] = changes
    shifts.masked_fill_(~known[:, None], 0.)
    return shifts, known, error


def own_peer_risk(broad, fine_field, shifts, known, parents, canonical, valid):
    if (broad.ndim != 2 or fine_field.shape != broad.shape or shifts.shape != (len(valid), len(parents))
            or known.shape != valid.shape or known.dtype != torch.bool or valid.dtype != torch.bool
            or not bool(torch.isfinite(broad).all() and torch.isfinite(fine_field).all()
                        and torch.isfinite(shifts).all())):
        raise ValueError('Finite class fields and own-word omission shifts required.')
    wide_margin = broad.double()[:, parents, None]-broad.double()[:, None, :]
    peer_margin = fine_field.double()[:, parents, None]+shifts[..., None]-fine_field.double()[:, None, :]
    positive, negative = wide_margin.clamp_min(0), (-peer_margin).clamp_min(0)
    risk = torch.where((wide_margin > 0) & (peer_margin < 0),
        negative/(positive+negative).clamp_min(audit.CONFIG.epsilon), 0.)
    risk[:, canonical] = 0.
    risk.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 0.)
    return risk.masked_fill(~(valid & known)[:, None, None], 0.)


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared own-peer candidate methods required.')
    args = (local, operator, broad, fine_field, wide, fine, coordinates,
        relation, valid, members, canonical, parents)
    requested = tuple(NEW_NAMES[name] for name in methods if name in NEW_NAMES)
    if MATCHED_POOLED in methods and audit.PRIMARY not in requested:
        requested = (*requested, audit.PRIMARY, audit.OBSERVATION_MEAN)
    values, stats, current = {}, dict.fromkeys(DIAGNOSTICS, 0.), {}

    def build_risk(wm, fm, source_parents, source_canonical, source_valid):
        shifts, known, error = own_omission_shift(fine, members, source_valid)
        risk = own_peer_risk(broad, fine_field, shifts, known, source_parents, source_canonical, source_valid)
        stats['peer_coverage_max_error'] = error
        stats['peer_omission_abs_mean'] = float(shifts[known].abs().mean()) if bool(known.any()) else 0.
        stats['peer_unknown_risk_max'] = float(risk.masked_fill(known[:, None, None], 0.).abs().max())
        if bool(source_valid.any()):
            acted = risk[source_valid] > 0
            count = acted.double().sum().clamp_min(1)
            stats['peer_joint_positive_action_fraction'] = float(
                (acted & (wm[source_valid] > 0) & (fm[source_valid] > 0)).double().sum()/count)
            full_margin = fine_field.double()[:, source_parents, None]-fine_field.double()[:, None, :]
            stats['peer_full_class_positive_action_fraction'] = float(
                (acted & (full_margin[source_valid] >= 0)).double().sum()/count)
        return risk

    if requested:
        current, diagnostics = audit.scores(*args, methods=requested, risk_builder=build_risk)
        stats.update(diagnostics)
        values.update({OLD_NAMES[name]: value for name, value in current.items() if OLD_NAMES[name] in methods})
    if set(methods) & {PREVIOUS_SOFT, PREVIOUS_HARD}:
        legacy_methods = tuple(old for new, old in ((PREVIOUS_SOFT, audit.PRIMARY),
            (PREVIOUS_HARD, audit.HARD)) if new in methods)
        previous, _ = audit.scores(*args, methods=legacy_methods)
        for new, old in ((PREVIOUS_SOFT, audit.PRIMARY), (PREVIOUS_HARD, audit.HARD)):
            if new in methods:
                values[new] = previous[old]
    if set(methods) & {POOLED, MATCHED_POOLED}:
        pooled, _ = budget.scores(*args, methods=(budget.POOLED, budget.OBSERVATION_MEAN))
        if POOLED in methods:
            values[POOLED] = pooled[budget.POOLED]
        if MATCHED_POOLED in methods:
            positive = pooled[budget.OBSERVATION_MEAN]
            word_write = current[audit.PRIMARY]-positive
            matched, info = match_previous(pooled[budget.POOLED]-positive, word_write, valid)
            values[MATCHED_POOLED] = positive+matched
            stats['pooled_matched_unmatchable'] = info['matched_previous_unmatchable']
            stats['pooled_matched_norm_relative_error'] = info['matched_previous_norm_relative_error']
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite own-peer candidate prediction.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
