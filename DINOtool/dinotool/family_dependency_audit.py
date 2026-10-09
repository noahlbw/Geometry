"""Mask-free separation of class disagreement from family-omission dependence."""
import torch


IMPLEMENTATION = 'geometry-native-family-dependency-audit-v1-20261003'


def alias_margins(field, known, assignments, parents, valid):
    if (field.ndim != 3 or known.shape != field.shape[1:] or known.dtype != torch.bool
            or assignments.shape != parents.shape or valid.shape != field.shape[:1]
            or valid.dtype != torch.bool or field.shape[-1] < 2
            or not bool(torch.isfinite(field).all())):
        raise ValueError('Finite family/class references and known aliases required.')
    reference, observed = field[:, assignments].double(), known[assignments]
    own = reference.gather(-1, parents[None, :, None].expand(len(valid), -1, 1))[..., 0]
    own_known = observed.gather(-1, parents[:, None])[:, 0]
    rival_known = observed.clone()
    rival_known.scatter_(-1, parents[:, None], False)
    supported = valid[:, None] & own_known[None] & rival_known.any(-1)[None]
    rival = reference.masked_fill(~rival_known[None], -torch.inf).amax(-1)
    return torch.where(supported, own-rival, 0.), supported


def dependency_fields(held, held_known, full, full_known, assignments, parents, valid):
    minus, known = alias_margins(held, held_known, assignments, parents, valid)
    original, original_known = alias_margins(full, full_known, torch.zeros_like(assignments), parents, valid)
    observed = known & original_known
    # Positive margin loss is dependence on a removed family, not semantic correctness.
    loss = (original-minus).masked_fill(~observed, 0.)
    risk_held = -torch.expm1(minus.clamp_max(0.))
    risk_full = -torch.expm1(original.clamp_max(0.))
    added = (risk_held-risk_full).masked_fill(~observed, 0.)
    numerical_guard = 64*torch.finfo(torch.float64).eps*(1+max(float(original.abs().max()), float(minus.abs().max())))
    opposing = observed & (loss*added < -numerical_guard)
    if bool(opposing.any()):
        raise RuntimeError('Monotone omission/risk identity failed.')
    return {'held_margin': minus, 'full_margin': original, 'margin_loss': loss,
        'risk_held': risk_held, 'risk_full': risk_full, 'added_risk': added,
        'known': observed}, numerical_guard


def summarize(fields, native_mass, slots=None, guard=0.):
    if native_mass.shape != fields['known'].shape or not bool(torch.isfinite(native_mass).all()) or bool((native_mass < 0).any()):
        raise ValueError('Finite nonnegative native family contribution required.')
    selection = torch.arange(native_mass.shape[1], device=native_mass.device) if slots is None else slots
    known = fields['known'][:, selection]
    loss, added = fields['margin_loss'][:, selection], fields['added_risk'][:, selection]
    weight = native_mass[:, selection].double()*known
    increased, decreased = known & (added > guard), known & (added < -guard)
    lost = known & (loss > guard)
    native = weight.sum()
    return {'known_comparisons': int(known.sum()), 'increased_risk_comparisons': int(increased.sum()),
        'increased_risk_with_lost_margin': int((increased & lost).sum()),
        'decreased_risk_comparisons': int(decreased.sum()), 'lost_margin_comparisons': int(lost.sum()),
        'native_mass': float(native), 'increased_risk_native_mass': float(weight[increased].sum()),
        'lost_margin_native_mass': float(weight[lost].sum()),
        'added_risk_native_mass': float((weight*added).sum()),
        'held_risk_native_mass': float((weight*fields['risk_held'][:, selection]).sum()),
        'full_risk_native_mass': float((weight*fields['risk_full'][:, selection]).sum()),
        'margin_loss_native_mass': float((weight*loss).sum())}
