"""Fixed decomposition of cached ownership actions, without a new predictor."""
import numpy as np


IMPLEMENTATION = 'geometry-ownership-action-path-audit-v1-20261003'


def action_decomposition(old_directed, new_directed, valid):
    old, new = np.asarray(old_directed, np.float64), np.asarray(new_directed, np.float64)
    valid = np.asarray(valid, bool)
    if (old.shape != new.shape or old.ndim != 3 or old.shape[-1] != old.shape[-2]
            or valid.shape != old.shape[:1] or not np.isfinite(old).all() or not np.isfinite(new).all()
            or np.any(old[~valid] != 0) or np.any(new[~valid] != 0)):
        raise ValueError('Matching finite cached directed actions and query validity required.')
    added = new-old
    edges = added-added.transpose(0, 2, 1)
    potential = edges.mean(-1)
    realized = potential[:, :, None]-potential[:, None, :]
    cycle = edges-realized
    requested_energy = float(np.square(edges).sum()/2)
    realized_energy = float(np.square(realized).sum()/2)
    cycle_energy = float(np.square(cycle).sum()/2)
    error = float(np.abs(cycle.sum(-1)).max())
    identity_error = abs(requested_energy-realized_energy-cycle_energy)
    if error > 1e-10 or identity_error > 1e-10*max(1., requested_energy):
        raise RuntimeError('Class potential decomposition does not replay.')
    return potential, {'requested_energy': requested_energy, 'realized_energy': realized_energy,
        'cycle_energy': cycle_energy, 'normal_equation_max_error': error,
        'orthogonal_energy_max_error': identity_error,
        'mean_added_directed_suppression': float(-added[valid].mean()) if valid.any() else 0.,
        'mean_absolute_potential': float(np.abs(potential[valid]).mean()) if valid.any() else 0.}


def target_margin_signs(before, after, old_scores, target, valid):
    before, after, old = [np.asarray(a, np.float64) for a in (before, after, old_scores)]
    target, valid = np.asarray(target, np.int64), np.asarray(valid, bool)
    if (before.shape != after.shape or old.shape != before.shape or old.ndim != 2
            or target.shape != old.shape[:1] or valid.shape != target.shape
            or not all(np.isfinite(a).all() for a in (before, after, old))):
        raise ValueError('Matching finite increments, old scores and labelled queries required.')
    use = valid & (target >= 0) & (target < old.shape[-1])
    ids, truth = np.flatnonzero(use), target[use]
    competitors = old[ids].copy()
    competitors[np.arange(len(ids)), truth] = -np.inf
    rival = competitors.argmax(-1)
    a = before[ids, truth]-before[ids, rival]
    b = after[ids, truth]-after[ids, rival]
    guard = 64*np.finfo(np.float64).eps*(1+max(float(np.abs(before).max()), float(np.abs(after).max())))
    sign_a = np.where(a > guard, 2, np.where(a < -guard, 0, 1))
    sign_b = np.where(b > guard, 2, np.where(b < -guard, 0, 1))
    classes = []
    for c in range(old.shape[-1]):
        mask = truth == c
        classes.append(np.bincount(3*sign_a[mask]+sign_b[mask], minlength=9).reshape(3, 3))
    return np.stack(classes), {'sign_order': ['negative', 'numerical_zero', 'positive'],
        'numerical_guard': guard, 'valid_labelled_queries': int(use.sum()),
        'mean_prewrite_target_margin_change': float(a.mean()) if len(a) else None,
        'mean_written_target_margin_change': float(b.mean()) if len(b) else None}
