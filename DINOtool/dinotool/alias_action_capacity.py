"""Audit the existing alias writer's action space; no deployable label selector."""
from __future__ import annotations

import torch

from .calibrated_competitive_alias import profiled_logits
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = "geometry-alias-action-capacity-audit-v1-20261003"


def evidence_caps(logits, canonical, beta=1.):
    if logits.ndim < 3 or canonical.shape != logits.shape[-2:] or beta <= 0:
        raise ValueError("Grouped evidence and matching canonical mask required.")
    k = logits.shape[-1]
    excess = ((beta * logits).softmax(-1) - 1 / k).clamp_min(0)
    single = -torch.log1p(-excess.clamp_max(1 - 1 / k)) / beta
    protected = -torch.log1p(-excess.masked_fill(canonical, 0.).sum(-1).clamp_max(1 - 1 / k)) / beta
    unrestricted = -torch.log1p(-excess.sum(-1).clamp_max(1 - 1 / k)) / beta
    return single, protected, unrestricted


@torch.inference_mode()
def suppression_caps(crops, count, coordinates, image_size, members, canonical, valid, beta=1., chunk=128):
    single = coordinates.new_zeros((len(coordinates), *members.shape))
    protected = coordinates.new_zeros((len(coordinates), len(members)))
    unrestricted = torch.zeros_like(protected)
    for start in range(0, len(coordinates), chunk):
        sl = slice(start, start + chunk)
        for crop in crops:
            indices, coefficients = crop_stencil(crop, count, coordinates[sl], image_size)
            fields = evidence_caps(profiled_logits(crop, members)[indices], canonical, beta)
            single[sl] += (fields[0] * coefficients[..., None, None]).sum(1)
            protected[sl] += (fields[1] * coefficients[..., None]).sum(1)
            unrestricted[sl] += (fields[2] * coefficients[..., None]).sum(1)
    return tuple(value.masked_fill(~valid.reshape(-1, *([1] * (value.ndim - 1))), 0.)
                 for value in (single, protected, unrestricted))


def reconstruction_operator(relation, valid):
    n = len(valid)
    if relation.shape != (n, n) or valid.dtype != torch.bool or bool((relation < 0).any()):
        raise ValueError("Finite nonnegative square relation and boolean validity required.")
    if not bool(torch.isfinite(relation).all()):
        raise ValueError("Finite relation required.")
    weights = relation.double().masked_fill(~valid[None], 0.)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights.masked_fill(~valid[:, None], 0.)
    gram = weights.T @ weights
    system = torch.eye(n, device=relation.device, dtype=torch.float64) + gram
    operator = torch.cholesky_solve(gram, torch.linalg.cholesky(system))
    operator = (operator + operator.T) * .5
    operator = operator.masked_fill(~valid[:, None] | ~valid[None], 0.)
    residual = float((system @ operator - gram).abs().max())
    return operator, residual


def action_envelope(baseline, operator, cap):
    if baseline.shape != cap.shape or operator.shape != (len(cap), len(cap)) or bool((cap < 0).any()):
        raise ValueError("Matching logits, nonnegative action caps and linear operator required.")
    # Each donor/class independently spans [-cap,0]; signed H entries matter.
    low = baseline.double() - operator.clamp_min(0) @ cap.double()
    high = baseline.double() - operator.clamp_max(0) @ cap.double()
    return low, high


def target_margin_upper(low, high, target):
    if low.shape != high.shape or target.shape != low.shape[:1] or low.shape[-1] < 2:
        raise ValueError("Matching multiclass envelopes and target vector required.")
    known = (target >= 0) & (target < low.shape[-1])
    safe = target.clamp(0, low.shape[-1]-1)
    rival = low.scatter(1, safe[:, None], -torch.inf).amax(-1)
    margin = high.gather(1, safe[:, None])[:, 0] - rival
    return margin.masked_fill(~known, torch.nan)


def label_assisted_action(cap, target):
    if target.shape != cap.shape[:1]:
        raise ValueError("Matching target vector required.")
    known = (target >= 0) & (target < cap.shape[-1])
    classes = torch.arange(cap.shape[-1], device=cap.device)
    reject = known[:, None] & (classes[None] != target[:, None])
    return -cap * reject


def changed_class_prediction(baseline, changed_scores, class_index):
    if baseline.ndim != 2 or changed_scores.shape[-1] != baseline.shape[0]:
        raise ValueError("Original [pixels,classes] and changed class planes required.")
    rivals = baseline.clone()
    rivals[:, class_index] = -torch.inf
    best, winner = rivals.max(-1)
    take = (changed_scores > best) | ((changed_scores == best) & (class_index < winner))
    return torch.where(take, class_index, winner)
