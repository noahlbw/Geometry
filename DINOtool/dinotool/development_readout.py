"""Explicit supervised development controls; retained model is not modified."""
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import torch

from .matched_readout_controls import run_head


IMPLEMENTATION = 'bounded-geometry-supervised-readout-development-v1-20261006'
THRESHOLDS = (0., .1, .2, .25, .3, .35, .4, .45, .5, .6, .7, .8, .9, .95)
BACKGROUND_BIASES = (0., -2., 2., 4., 8.)


@dataclass(frozen=True)
class Profile:
    bank: str = 'original'
    strength: object = 2.
    coupling: float = 1.
    temperature: float = .07
    tau: float = 1.
    tem: float = 1.

    def record(self):
        return asdict(self)


def profiles(bank_names):
    result = [Profile()]
    for bank in bank_names:
        for strength in (1., 2., 3., 'original'):
            for coupling in (.5, 1., 2.):
                result.append(Profile(bank, strength, coupling))
        for tau, tem in ((4., 1.), (1., .3), (1., 3.)):
            result.append(Profile(bank=bank, tau=tau, tem=tem))
        for temperature in (.05, .1):
            result.append(Profile(bank=bank, temperature=temperature))
    return tuple(dict.fromkeys(result))


def development_keys(keys, count, seed=20261006):
    if len(keys) != len(set(keys)) or not 0 < count < len(keys):
        raise ValueError('Unique global samples and a nonempty disjoint holdout required.')
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest())
    selected = set(ranked[:count])
    return [k for k in keys if k in selected]


def projected(head, prepared, strength):
    if strength == 'original':
        return prepared.geometry_projected
    if strength not in (1., 2., 3.):
        raise ValueError('Undeclared Geometry read strength.')
    return run_head(head, prepared.backbone_tokens,
        prepared.backbone_tokens[:, prepared.prefix_tokens:],
        prepared.geometry_patch_conditional * strength, prepared.prefix_tokens,
        'Geometry_BlockPrefix', prepared.block_index)[0]


def coupled(local, broad, operator, gain):
    if gain not in (.5, 1., 2.):
        raise ValueError('Undeclared coupled correction strength.')
    return local.double() + gain * (operator.double() @ (broad.double() - local.double()))


def biased_probabilities(probability, bias, background):
    if background is None:
        if bias != 0:
            raise ValueError('Background offsets require a scored residual category.')
        return probability
    if bias == 0:
        return probability
    out = probability.clone()
    out[background] *= np.exp(bias)
    return out / out.sum(0, keepdim=True).clamp_min(1e-12)


def threshold_histogram(probability, target, background, bias=0., thresholds=THRESHOLDS):
    """Exact confusion histograms; threshold ties survive, labels never affect prediction."""
    classes = probability.shape[0]
    if probability.shape[1:] != target.shape:
        raise ValueError('Original-resolution target/probability mismatch.')
    valid = (target >= 0) & (target < classes)
    confidence, prediction = biased_probabilities(probability, bias, background).max(0)
    cuts = torch.tensor(thresholds, dtype=confidence.dtype, device=confidence.device)
    bins = torch.bucketize(confidence[valid].contiguous(), cuts, right=True)
    code = bins.long() * classes * classes + target[valid].long() * classes + prediction[valid].long()
    result = torch.bincount(code, minlength=(len(thresholds) + 1) * classes * classes)
    return result.reshape(len(thresholds) + 1, classes, classes).cpu().numpy()


def threshold_matrices(histogram, background, thresholds=THRESHOLDS):
    total = histogram.sum(0)
    if background is None:
        return {0.: total}
    output = {}
    for index, threshold in enumerate(thresholds):
        rejected = histogram[:index + 1].sum(0)
        matrix = total - rejected
        matrix[:, background] += rejected.sum(1)
        output[threshold] = matrix
    return output


def score(matrix, support=None):
    matrix = np.asarray(matrix)
    tp = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - tp
    selected = union > 0 if support is None else np.asarray(support) & (union > 0)
    return 100 * float(np.mean(tp[selected] / union[selected])) if selected.any() else 0.


def select(histograms, bank_names, background):
    candidates = profiles(bank_names)
    support = histograms[0, 0].sum(0).sum(1) > 0
    ranked = []
    biases = BACKGROUND_BIASES if background is not None else (0.,)
    for index, profile in enumerate(candidates):
        for b, bias in enumerate(biases):
            for threshold, matrix in threshold_matrices(histograms[index, b], background).items():
                ranked.append(dict(profile=profile.record(), background_bias=bias,
                    background_threshold=threshold, development_miou=score(matrix, support),
                    profile_index=index, bias_index=b))
    # Stable ties retain the reference, zero bias and the least rejection.
    ranked.sort(key=lambda r: -r['development_miou'])
    best = ranked[0].copy()
    best['confusion_matrix'] = threshold_matrices(
        histograms[best['profile_index'], best['bias_index']], background)[best['background_threshold']].tolist()
    return best, ranked
