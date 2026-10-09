"""Mask-free alias, template and score calibration outside the frozen model."""
from collections import defaultdict
import math

import numpy as np
import torch

from .natural_evaluation import LEXICON


IMPLEMENTATION = 'natural-semantic-alias-self-calibration-v1-20261003'
CLASS_FILES = dict(voc20='voc20', voc21='voc21', context59='context59', context60='context60',
                   ade150='ade20k', coco_stuff171='coco_stuff', coco_object81='coco_object',
                   cityscapes19='city_scapes')
CONFIG_FILES = {**CLASS_FILES, 'coco_stuff171': 'coco_stuff164k'}
TEMPLATE_FAMILIES = ('seg_template', 'openai_imagenet_template', 'sub_imagenet_template', 'city_template')
TAU_GRID = (1., 3., 5.)
TEM_GRID = (.3, 1., 5.)
SEED = 20261003
CALIBRATION_IMAGES = 64


def key(word):
    return ' '.join(word.replace('_', ' ').replace('-', ' ').casefold().split())


def semantic_pool(names, official):
    if len(names) != len(official):
        raise ValueError('Canonical taxonomy and official query class order differ.')
    canonical_owners = {key(name): c for c, name in enumerate(names)}
    proposals = [tuple(dict.fromkeys((name, *official[c], *LEXICON.get(name, ()))))
                 for c, name in enumerate(names)]
    owners = defaultdict(set)
    for c, group in enumerate(proposals):
        for word in group:
            owners[key(word)].add(c)
    groups, removed = [], []
    for c, group in enumerate(proposals):
        accepted, seen = [], set()
        for word in group:
            identity = key(word)
            reason = None
            if identity in seen:
                reason = 'lexical_duplicate'
            elif identity in canonical_owners and canonical_owners[identity] != c:
                reason = 'other_scored_class_canonical'
            elif len(owners[identity]) > 1 and identity not in canonical_owners:
                reason = 'ambiguous_explicit_parent'
            if reason:
                removed.append(dict(parent=names[c], alias=word, reason=reason))
            else:
                accepted.append(word)
                seen.add(identity)
        if not accepted or accepted[0] != names[c]:
            raise ValueError('Canonical alias must survive in every class.')
        groups.append(tuple(accepted))
    return tuple(groups), removed


def class_logits(alias_logits, salience, parents, classes, tau=1., tem=1.):
    if tau <= 0 or tem <= 0 or not bool(torch.isfinite(alias_logits).all()):
        raise ValueError('Finite alias scores and positive calibration parameters required.')
    result = []
    for c in range(classes):
        members = parents == c
        if not bool(members.any()):
            raise ValueError('Empty scored class.')
        weights = (salience[members] / tem).softmax(0)
        values = alias_logits[:, members] * (weights * int(members.sum()))
        result.append((tau * values).logsumexp(-1) / tau)
    return torch.stack(result, -1)


def canonical_anchor(logits, canonical, parents, classes, background=False):
    result = logits[:, canonical].clone()
    if background:
        # A residual class has no single visual prototype: concrete scene words are its anchors.
        bg = (parents == 0) & ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)
        if bool(bg.any()):
            result[:, 0] = logits[:, bg].amax(-1)
    return result


def trusted_witnesses(geo, wide, canonical, parents, classes, background=False):
    ga = canonical_anchor(geo, canonical, parents, classes, background)
    wa = canonical_anchor(wide, canonical, parents, classes, background)
    gv, gi = ga.topk(2, -1)
    wv, wi = wa.topk(2, -1)
    labels = gi[:, 0]
    quality = (torch.tanh((gv[:, 0] - gv[:, 1]).clamp_min(0) / 2)
               * torch.tanh((wv[:, 0] - wv[:, 1]).clamp_min(0) / 2)).sqrt()
    known = (labels == wi[:, 0]) & (quality > .25)
    return labels, quality * known, known


def choose_aliases(records, parents, canonical, names):
    margins_by_alias = defaultdict(list)
    support_images = defaultdict(set)
    for image_index, row in enumerate(records):
        geo, wide, labels, weight = (row[k] for k in ('geo', 'wide', 'labels', 'quality'))
        evidence = (geo + wide) / 2
        for c in range(len(names)):
            own = (labels == c) & (weight > 0)
            foreign = (labels != c) & (weight > 0)
            if not bool(own.any() and foreign.any()):
                continue
            for alias in (parents == c).nonzero().flatten().tolist():
                positive = (evidence[own, alias] * weight[own]).sum() / weight[own].sum()
                negative = (evidence[foreign, alias] * weight[foreign]).sum() / weight[foreign].sum()
                margins_by_alias[alias].append(float(positive - negative))
                support_images[alias].add(image_index)
    selected, details = [], []
    protected = set(canonical.tolist())
    for alias in range(len(parents)):
        margins = np.asarray(margins_by_alias[alias], dtype=np.float64)
        upper = None
        if len(margins) >= 8:
            upper = float(margins.mean() + 1.96 * margins.std(ddof=1) / math.sqrt(len(margins)))
        # No quota: absent/uncertain concepts survive; only repeated negative evidence deletes.
        reject = alias not in protected and upper is not None and upper < 0
        if not reject:
            selected.append(alias)
        details.append(dict(alias_index=alias, retained=not reject, canonical=alias in protected,
                            witness_images=len(support_images[alias]), mean_margin=float(margins.mean()) if len(margins) else None,
                            upper95_margin=upper))
    return torch.tensor(selected, device=parents.device, dtype=torch.long), details


def balanced_nll(scores, labels, quality):
    log_prob = scores.log_softmax(-1)
    loss = -log_prob.gather(1, labels[:, None])[:, 0]
    supported = []
    for c in labels[quality > 0].unique().tolist():
        use = (labels == c) & (quality > 0)
        supported.append((loss[use] * quality[use]).sum() / quality[use].sum())
    return float(torch.stack(supported).mean()) if supported else float('inf')


def background_threshold(confidence, background, quality):
    confidence = np.asarray(confidence, dtype=np.float64)
    background = np.asarray(background, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    known = quality > 0
    if not np.any(known & background) or not np.any(known & ~background):
        return 0., 'No trusted foreground/background pair; threshold stays off.'
    values, labels, weights = confidence[known], background[known], quality[known]
    candidates = np.unique(np.concatenate(([0.], np.quantile(values, np.linspace(0, 1, 101)))))
    positives, negatives = weights[labels].sum(), weights[~labels].sum()
    scores = [((weights[(values < t) & labels].sum() / positives)
               + (weights[(values >= t) & ~labels].sum() / negatives)) / 2 for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), 'Class-balanced trusted-witness threshold; no masks.'
