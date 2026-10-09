"""Unlabeled two-mode class-margin calibration on fixed Geometry outputs."""
import numpy as np
from sklearn.mixture import GaussianMixture


IMPLEMENTATION = "geometry-density-margin-calibration-v1-20261001"


def between_mode_boundary(means, variances, priors):
    """Posterior equality between the two ordered Gaussian modes, if identified."""
    order = np.argsort(means)
    low, high = np.asarray(means)[order]
    vl, vh = np.asarray(variances)[order]
    pl, ph = np.asarray(priors)[order]
    if not (np.isfinite([low, high, vl, vh, pl, ph]).all() and high > low
            and min(vl, vh, pl, ph) > 0):
        return None
    a = .5/vl-.5/vh
    b = high/vh-low/vl
    c = -.5*high*high/vh+.5*low*low/vl+np.log(ph/pl)-.5*np.log(vh/vl)
    if abs(a) < 1e-12:
        roots = [-c/b] if abs(b) > 1e-12 else []
    else:
        discriminant = b*b-4*a*c
        roots = [(-b+sign*np.sqrt(discriminant))/(2*a) for sign in (-1, 1)] if discriminant >= 0 else []
    inside = [root for root in roots if np.isfinite(root) and low <= root <= high]
    return min(inside, key=lambda root: abs(root-.5*(low+high))) if inside else None


def margin_mode_shift(margin):
    values = np.asarray(margin, np.float64)
    mean, deviation = values.mean(), values.std()
    if len(values) < 4 or not np.isfinite(values).all() or deviation < 1e-12:
        return 0., {"resolved": False, "reason": "constant_or_insufficient"}
    standardized = ((values-mean)/deviation)[:, None]
    options = dict(covariance_type="full", random_state=20261001, n_init=3,
                   reg_covar=1e-6, max_iter=100)
    null = GaussianMixture(n_components=1, **options).fit(standardized)
    split = GaussianMixture(n_components=2, **options).fit(standardized)
    bic_gain = null.bic(standardized)-split.bic(standardized)
    means, variances = split.means_[:, 0], split.covariances_[:, 0, 0]
    boundary = between_mode_boundary(means, variances, split.weights_)
    resolved = bool(split.converged_ and bic_gain > 0 and boundary is not None)
    shift = float(mean+deviation*boundary) if resolved else 0.
    return shift, {"resolved": resolved, "bic_gain": float(bic_gain),
                   "mode_means": (mean+deviation*means).tolist(), "mode_weights": split.weights_.tolist(),
                   "boundary": shift if resolved else None}


def calibrate_margins(scores, valid, relation=None):
    """Fit from image scores only; no labels, class identities or fitted coefficients.

    Geometry can stabilize the fitting signal, but writes only one class bias
    into the original scores. This does not assert that mixture modes are true
    foreground/background or that their posterior equals semantic correctness.
    """
    scores = np.asarray(scores)
    valid = np.asarray(valid, bool)
    if scores.ndim != 2 or valid.shape != scores.shape[:1] or scores.shape[1] < 2:
        raise ValueError("Expected [N,C] scores and [N] image validity.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    shifts, records = np.zeros(scores.shape[1]), []
    if not valid.any():
        return scores.copy(), {"shifts": shifts.tolist(), "classes": records}
    for cls in range(scores.shape[1]):
        others = np.max(np.delete(scores, cls, axis=1), axis=1)
        margin = scores[:, cls]-others
        if relation is not None:
            weights = np.asarray(relation, np.float64).copy()
            if weights.shape != (len(scores), len(scores)) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Expected finite nonnegative [N,N] Geometry.")
            weights[:, ~valid] = 0
            weights /= np.maximum(weights.sum(-1, keepdims=True), 1e-12)
            margin = weights @ margin
        shift, record = margin_mode_shift(margin[valid])
        shifts[cls] = shift
        records.append(record)
    corrected = scores.astype(np.float64)-shifts[None]
    corrected[~valid] = scores[~valid]
    return corrected, {"shifts": shifts.tolist(), "classes": records}
