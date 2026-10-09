"""Vocabulary-count calibration outside the frozen coupled reader."""
import math

import numpy as np
import torch


IMPLEMENTATION = 'natural-coupled-count-prior-calibration-v1-20261004'


def count_prior_delta(operator, counts, tau=1., strength=1.):
    """Replay a broad log-mean-exp offset through the unchanged reconstruction."""
    if (operator.ndim != 2 or operator.shape[0] != operator.shape[1]
            or counts.ndim != 1 or len(counts) < 2
            or not bool(torch.isfinite(operator).all())
            or not bool(torch.isfinite(counts).all()) or bool((counts < 1).any())
            or bool((counts != counts.round()).any())
            or not math.isfinite(tau) or tau <= 0
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Finite reconstruction, positive integer counts and bounded calibration required.')
    prior = counts.to(device=operator.device, dtype=torch.float64).log() / tau
    return -strength * operator.double().sum(-1, keepdim=True) * prior[None]


def calibrate_coupled_counts(scores, operator, counts, tau=1., strength=1.):
    if (scores.ndim != 2 or scores.shape != (len(operator), len(counts))
            or scores.device != operator.device or not bool(torch.isfinite(scores).all())):
        raise ValueError('Matching finite coupled scores and reconstruction required.')
    delta = count_prior_delta(operator, counts, tau, strength)
    if strength == 0:
        return scores
    return scores.double() + delta


def witness_error(scores, labels, quality):
    """Class-balanced pseudo-witness disagreement, not confidence maximization."""
    if (scores.ndim != 2 or labels.shape != scores.shape[:1]
            or quality.shape != labels.shape or not bool(torch.isfinite(scores).all())
            or not bool(torch.isfinite(quality).all()) or bool((quality < 0).any())
            or labels.dtype != torch.long or bool((labels < 0).any())
            or bool((labels >= scores.shape[-1]).any())):
        raise ValueError('Finite scores, valid witness labels and nonnegative weights required.')
    known = quality > 0
    if not bool(known.any()):
        return None
    error = (scores.argmax(-1) != labels).double()
    values = []
    for c in labels[known].unique().tolist():
        use = known & (labels == c)
        weights = quality[use].double()
        values.append((error[use] * weights).sum() / weights.sum())
    return float(torch.stack(values).mean())


def choose_count_strength(records, counts, tau=1., strengths=(0., .5, 1.), minimum_images=8):
    """Use image blocks; retain zero adjustment without repeated paired evidence."""
    if not strengths or strengths[0] != 0 or len(set(strengths)) != len(strengths):
        raise ValueError('Distinct candidates starting with the unchanged control required.')
    errors = {float(s): [] for s in strengths}
    for row in records:
        for strength in errors:
            scores = calibrate_coupled_counts(row['scores'], row['operator'], counts, tau, strength)
            value = witness_error(scores, row['labels'], row['quality'])
            if value is not None:
                errors[strength].append(value)
    baseline = np.asarray(errors[0.], dtype=np.float64)
    candidates = []
    for strength, values in errors.items():
        difference = np.asarray(values, dtype=np.float64) - baseline
        upper = None
        if len(difference) >= minimum_images and len(difference) >= 2:
            upper = float(difference.mean() + 1.96 * difference.std(ddof=1) / math.sqrt(len(difference)))
        candidates.append(dict(strength=strength, witness_images=len(values),
            mean_witness_error=float(np.mean(values)) if values else None,
            mean_paired_error_delta=float(difference.mean()) if len(difference) else None,
            upper95_paired_error_delta=upper))
    eligible = [c for c in candidates if c['strength'] > 0
                and c['upper95_paired_error_delta'] is not None and c['upper95_paired_error_delta'] < 0]
    chosen = min(eligible, key=lambda c: (c['mean_witness_error'], c['strength']))['strength'] if eligible else 0.
    return dict(strength=chosen, candidates=candidates, target_masks_loaded=False,
                selection_rule='Image-block paired pseudo-witness disagreement; no NLL ranking; zero fallback.',
                interpretation='Self-consistency calibration only, not evidence of ground-truth IoU improvement.')
