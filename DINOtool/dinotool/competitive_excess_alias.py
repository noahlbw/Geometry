"""Cap wide alias-to-rival overstatement using the existing fine observation."""
import torch

from . import alias_budget_attribution as budget
from . import one_sided_alias_audit as audit
from . import own_peer_alias as peer
from .native_alias_noise import signed_potential
from .supported_positive_alias import predict_image as source_predict
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-competitive-excess-fixed-slot-v1-20261006'
OLD_NAMES = {name: name.replace('OneSide_', 'ExcessCap_') for name in audit.METHODS[3:]}
NEW_NAMES = {new: old for old, new in OLD_NAMES.items()}
PRIMARY = OLD_NAMES[audit.PRIMARY]
OBSERVATION_MEAN = OLD_NAMES[audit.OBSERVATION_MEAN]
PREVIOUS_SOFT = 'ExcessCap_PreviousSoft'
PREVIOUS_HARD = 'ExcessCap_PreviousHard'
PREVIOUS_PEER = 'ExcessCap_PreviousOwnPeer'
POOLED = 'ExcessCap_PooledClass'
MATCHED_POOLED = 'ExcessCap_MatchedPooledClass'
CLASS_EXCESS = 'ExcessCap_ClassExcess'
MATCHED_EXCESS = 'ExcessCap_MatchedClassExcess'
METHODS = (*audit.METHODS[:3], *OLD_NAMES.values(), PREVIOUS_SOFT, PREVIOUS_HARD,
           PREVIOUS_PEER, POOLED, MATCHED_POOLED, CLASS_EXCESS, MATCHED_EXCESS)
DIAGNOSTICS = (*audit.DIAGNOSTICS, 'excess_risk_mean', 'excess_expanded_support_fraction',
    'excess_joint_positive_action_fraction', 'excess_advantage_loss_mean',
    'class_excess_risk_mean', 'calibration_matched_norm_relative_error',
    'calibration_matched_unmatchable')
PROTOCOL = {**audit.PROTOCOL,
    'alias_admission': 'wide-positive alias/rival advantage exceeds its fine advantage; fine sign unrestricted',
    'retention': 'exp(-max(wide_margin-fine_margin,0)); canonical/self/unknown protected',
    'source': 'same sampled salience-profiled alias-to-rival log-mean-exponential margins, beta1',
    'reference': 'unchanged rival20 pool; no own-word omission or semantic-truth gate',
    'aggregation': 'original20 fixed slots; no survivor compensation or source salience refit',
    'controls': METHODS[4:], 'new_model_equations': True, 'fitted_parameters': 0,
    'additional_visual_or_head_encodings': 0,
    'precedents': 'margin transfer and responsibility intersection precede this scoped fixed-slot test',
    'limitation': 'scale-dependent discrimination is not correctness; legitimate broad context can be attenuated'}


def competitive_excess(wide, fine):
    if (wide.shape != fine.shape or wide.ndim != 3
            or not bool(torch.isfinite(wide).all() and torch.isfinite(fine).all())):
        raise ValueError('Matching finite wide/fine competitive margins required.')
    loss = (wide.double()-fine.double()).clamp_min(0)
    return (-torch.expm1(-loss)).masked_fill(wide <= 0, 0.)


def excess_risk(wide, fine, parents, canonical, valid):
    if (wide.ndim != 3 or fine.shape != wide.shape or parents.shape != wide.shape[1:2]
            or canonical.shape != wide.shape[2:] or valid.shape != wide.shape[:1]
            or parents.dtype != torch.long or canonical.dtype != torch.long or valid.dtype != torch.bool
            or bool((parents < 0).any()) or bool((parents >= wide.shape[-1]).any())
            or bool((canonical < 0).any()) or bool((canonical >= wide.shape[1]).any())
            or len(canonical.unique()) != len(canonical)):
        raise ValueError('Actual class identities and valid queries required.')
    risk = competitive_excess(wide, fine)
    risk[:, canonical] = 0.
    risk.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 0.)
    return risk.masked_fill(~valid[:, None, None], 0.)


def class_excess_risk(broad, fine, valid):
    if (broad.ndim != 2 or fine.shape != broad.shape or broad.shape[-1] < 2
            or valid.shape != broad.shape[:1] or valid.dtype != torch.bool):
        raise ValueError('Matching class observations and Boolean validity required.')
    wide_margin = broad.double()[:, :, None]-broad.double()[:, None]
    fine_margin = fine.double()[:, :, None]-fine.double()[:, None]
    return competitive_excess(wide_margin, fine_margin).masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared frozen competitive-excess endpoints required.')
    args = (local, operator, broad, fine_field, wide, fine, coordinates,
            relation, valid, members, canonical, parents)
    requested = tuple(NEW_NAMES[name] for name in methods if name in NEW_NAMES)
    if set(methods) & {MATCHED_POOLED, MATCHED_EXCESS} and audit.PRIMARY not in requested:
        requested = (*requested, audit.PRIMARY, audit.OBSERVATION_MEAN)
    values, stats, current = {}, dict.fromkeys(DIAGNOSTICS, 0.), {}

    def build_risk(wm, fm, source_parents, source_canonical, source_valid):
        risk = excess_risk(wm, fm, source_parents, source_canonical, source_valid)
        if bool(source_valid.any()):
            active = risk[source_valid] > 0
            count = active.double().sum().clamp_min(1)
            stats['excess_risk_mean'] = float(risk[source_valid].mean())
            stats['excess_expanded_support_fraction'] = float(
                (active & (fm[source_valid] >= 0)).double().sum()/count)
            stats['excess_joint_positive_action_fraction'] = float(
                (active & (wm[source_valid] > 0) & (fm[source_valid] > 0)).double().sum()/count)
            stats['excess_advantage_loss_mean'] = float(
                (wm[source_valid].double()-fm[source_valid].double())[active].mean()) if bool(active.any()) else 0.
        return risk

    if requested:
        current, diagnostics = audit.scores(*args, methods=requested, risk_builder=build_risk)
        stats.update(diagnostics)
        values.update({OLD_NAMES[name]: value for name, value in current.items() if OLD_NAMES[name] in methods})
    if set(methods) & {PREVIOUS_SOFT, PREVIOUS_HARD}:
        old_methods = tuple(old for new, old in ((PREVIOUS_SOFT, audit.PRIMARY),
            (PREVIOUS_HARD, audit.HARD)) if new in methods)
        old, _ = audit.scores(*args, methods=old_methods)
        values.update({new: old[original] for new, original in
            ((PREVIOUS_SOFT, audit.PRIMARY), (PREVIOUS_HARD, audit.HARD)) if new in methods})
    if PREVIOUS_PEER in methods:
        old, _ = peer.scores(*args, methods=(peer.PRIMARY,))
        values[PREVIOUS_PEER] = old[peer.PRIMARY]
    if set(methods) & {POOLED, MATCHED_POOLED, CLASS_EXCESS, MATCHED_EXCESS}:
        baseline = local.double()+operator.double()@(broad.double()-local.double())
        innovation = .5*(fine_field.double()-broad.double())
        innovation.masked_fill_(~valid[:, None], 0.)
        positive = baseline+operator.double()@innovation
        for raw, matched, builder in ((POOLED, MATCHED_POOLED, budget.pooled_class_risk),
                                      (CLASS_EXCESS, MATCHED_EXCESS, class_excess_risk)):
            if not set(methods) & {raw, matched}:
                continue
            risk = builder(broad, fine_field, valid)
            if raw == CLASS_EXCESS and bool(valid.any()):
                stats['class_excess_risk_mean'] = float(risk[valid].mean())
            potential, _ = signed_potential(budget.uniform_class_action(wide, members, canonical, risk, valid), valid)
            written = .5*(operator.double()@potential)
            if raw in methods:
                values[raw] = positive+written
            if matched in methods:
                reference = current[audit.PRIMARY]-positive
                changed, info = match_previous(written, reference, valid)
                values[matched] = positive+changed
                stats['calibration_matched_unmatchable'] += info['matched_previous_unmatchable']
                stats['calibration_matched_norm_relative_error'] = max(
                    stats['calibration_matched_norm_relative_error'], info['matched_previous_norm_relative_error'])
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite competitive-excess prediction.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
