"""Budget-matched spatial controls for an already frozen alias suppression field."""
import numpy as np


IMPLEMENTATION = 'geometry-alias-spatial-allocation-audit-v1-20261003'
SEEDS = (20261003, 20261004, 20261005)
METHODS = ('Geometry', 'Anchored_Exact', 'TargetContext_Exact', 'RawMean_Exact',
           'CapacityMean_Exact', *('SpatialShuffle'+str(i)+'_Exact' for i in range(3)))
ROUNDING_GUARD = 1e-6


def validated_magnitudes(delta, cap, valid):
    delta, cap, valid = np.asarray(delta, np.float64), np.asarray(cap, np.float64), np.asarray(valid, bool)
    if delta.ndim != 2 or cap.shape != delta.shape or valid.shape != delta.shape[:1]:
        raise ValueError('Matching donor/class fields and donor validity required.')
    if not np.isfinite(delta).all() or not np.isfinite(cap).all() or (cap < 0).any():
        raise ValueError('Finite fields and nonnegative capacities required.')
    if (delta > ROUNDING_GUARD).any() or (np.abs(delta[~valid]) > ROUNDING_GUARD).any():
        raise ValueError('Suppression-only actions and inert invalid donors required.')
    magnitude = np.maximum(-delta, 0)
    magnitude[~valid] = 0
    if (magnitude-cap > ROUNDING_GUARD).any():
        raise ValueError('Original action exceeds the protected writer capacity.')
    # Original fields/caps were saved in fp32; keep only their measured rounding slack.
    effective_cap = np.maximum(cap, magnitude)
    effective_cap[~valid] = 0
    return magnitude, effective_cap


def feasible_spatial_shuffle(delta, cap, valid, seed):
    magnitude, effective_cap = validated_magnitudes(delta, cap, valid)
    known = np.flatnonzero(valid)
    output = np.zeros_like(magnitude)
    generator = np.random.default_rng(seed)
    for c in range(magnitude.shape[-1]):
        values = np.sort(magnitude[known, c])[::-1]
        available = np.ones(len(known), bool)
        for value in values:
            eligible = np.flatnonzero(available & (effective_cap[known, c] >= value))
            if not len(eligible):
                raise RuntimeError('Feasible sorted assignment failed.')
            chosen = generator.choice(eligible)
            output[known[chosen], c] = value
            available[chosen] = False
    return -output


def capacity_balanced_mean(delta, cap, valid):
    magnitude, effective_cap = validated_magnitudes(delta, cap, valid)
    known = np.flatnonzero(valid)
    output = np.zeros_like(magnitude)
    if not len(known):
        return output
    for c in range(magnitude.shape[-1]):
        budget = magnitude[known, c].sum()
        if budget == 0:
            continue
        ordered = np.sort(effective_cap[known, c])
        prefix = np.r_[0., ordered.cumsum()[:-1]]
        levels = (budget-prefix)/np.arange(len(known), 0, -1)
        feasible = np.flatnonzero(levels <= ordered+1e-12)
        if not len(feasible):
            raise RuntimeError('Feasible capacity-balanced mean failed.')
        level = max(float(levels[feasible[0]]), 0.)
        output[known, c] = np.minimum(effective_cap[known, c], level)
    return -output


def matched_controls(delta, cap, valid):
    magnitude, _ = validated_magnitudes(delta, cap, valid)
    raw = np.zeros_like(magnitude)
    if np.any(valid):
        raw[valid] = -magnitude[valid].mean(0)
    return {'RawMean_Exact': raw, 'CapacityMean_Exact': capacity_balanced_mean(delta, cap, valid),
        **{'SpatialShuffle'+str(i)+'_Exact': feasible_spatial_shuffle(delta, cap, valid, seed)
           for i, seed in enumerate(SEEDS)}}
