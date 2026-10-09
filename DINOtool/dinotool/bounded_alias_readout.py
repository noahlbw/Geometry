"""Entropy-regularized alias aggregation with a fixed responsibility ceiling."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


IMPLEMENTATION = "geometry-bounded-alias-anchored-v2-exact-inactive-20261002"
CONFIG = {"responsibility_multiplier": 4.0, "local_temperature": 0.07,
          "alias_deletion": False, "cap_scope": "effective alias logits at class aggregation"}
PRIMARY = "Bounded_Anchored"
METHODS = ("Geometry", "Bounded_Local", "BroadVIP", "Bounded_BroadVIP",
           "MeanProb_VIP", "MeanLogit_VIP", "Bounded_MeanLogit",
           "Anchored_VIP", PRIMARY)
FACTORIAL_PAIRS = {"alias_in_fusion": ("MeanLogit_VIP", "Bounded_MeanLogit"),
                   "alias_in_anchored": ("Anchored_VIP", PRIMARY),
                   "geometry_in_original": ("MeanLogit_VIP", "Anchored_VIP"),
                   "geometry_in_bounded": ("Bounded_MeanLogit", PRIMARY),
                   "alias_local": ("Geometry", "Bounded_Local"),
                   "alias_broad": ("BroadVIP", "Bounded_BroadVIP")}


def capped_responsibilities(scores, temperature, multiplier):
    if (scores.ndim < 1 or scores.shape[-1] < 1 or temperature <= 0
            or not math.isfinite(temperature) or multiplier < 1
            or not math.isfinite(multiplier) or not scores.is_floating_point()
            or not bool(torch.isfinite(scores).all())):
        raise ValueError("Finite floating alias scores, positive temperature and multiplier >= 1 required.")
    count = scores.shape[-1]
    if multiplier >= count:
        return (scores / temperature).softmax(-1)
    if multiplier == 1:
        return torch.full_like(scores, 1.0 / count)
    cap = multiplier / count
    logits = (scores.double() - scores.double().amax(-1, keepdim=True)) / temperature
    ordered = logits.sort(dim=-1, descending=True).values
    tails = torch.logcumsumexp(ordered.flip(-1), dim=-1).flip(-1)
    remaining = 1.0 - cap * torch.arange(count, device=scores.device, dtype=logits.dtype)
    log_mass = remaining.clamp_min(torch.finfo(logits.dtype).tiny).log()
    # Select how many largest entries saturate; the remaining entries share a softmax.
    tolerance = 8 * torch.finfo(logits.dtype).eps * (1 + ordered.abs() + tails.abs())
    feasible = (remaining > 0) & (log_mass + ordered - tails <= math.log(cap) + tolerance)
    if not bool(feasible.any(-1).all()):
        raise RuntimeError("No feasible capped simplex solution.")
    saturated = feasible.to(torch.int64).argmax(-1, keepdim=True)
    threshold = tails.gather(-1, saturated) - log_mass[saturated]
    bounded = (logits - threshold).exp().clamp_max(cap).to(scores.dtype)
    original = (scores / temperature).softmax(-1)
    inactive = original.amax(-1, keepdim=True) <= cap
    return torch.where(inactive, original, bounded)


def bounded_score(scores, temperature, multiplier, *, normalize_count):
    weights = capped_responsibilities(scores, temperature, multiplier)
    count = scores.shape[-1]
    original_weights = (scores / temperature).softmax(-1)
    original_score = temperature * (torch.logsumexp(scores / temperature, -1)
                                   - (math.log(count) if normalize_count else 0.0))
    inactive = original_weights.amax(-1) <= multiplier / count
    if multiplier >= count:
        score = original_score
    elif multiplier == 1:
        score = scores.mean(-1) + (0.0 if normalize_count else temperature * math.log(count))
    else:
        peak = scores.amax(-1)
        entropy = -(weights * weights.clamp_min(torch.finfo(scores.dtype).tiny).log()).sum(-1)
        score = peak + (weights * (scores - peak[..., None])).sum(-1) + temperature * entropy
        if normalize_count:
            score = score - temperature * math.log(count)
        # An inactive constraint must not change the original score, even at ties.
        score = torch.where(inactive, original_score, score)
    return score, {
        "capped_patch_class_fraction": float((original_weights.amax(-1) > multiplier / count).float().mean()),
        "original_max_responsibility": float(original_weights.amax(-1).mean()),
        "bounded_max_responsibility": float(weights.amax(-1).mean()),
        "original_effective_aliases": float(original_weights.square().sum(-1).reciprocal().mean()),
        "bounded_effective_aliases": float(weights.square().sum(-1).reciprocal().mean()),
        "mean_absolute_score_change": float((score - original_score).abs().mean()),
        "inactive_max_absolute_score_change": float(torch.where(
            inactive, (score - original_score).abs(), 0.0).amax()),
        "responsibility_mass_error": float((weights.sum(-1) - 1).abs().amax()),
    }


def bounded_classes(alias_scores, parents, classes, temperature=0.07, multiplier=4.0,
                    *, normalize_count=True):
    outputs, statistics = [], {}
    for index in range(classes):
        score, diagnostics = bounded_score(alias_scores[..., parents == index], temperature,
                                           multiplier, normalize_count=normalize_count)
        outputs.append(score)
        for field, value in diagnostics.items():
            statistics[field] = statistics.get(field, 0.0) + value / classes
    return torch.stack(outputs, -1), statistics


def vip_profile_scores(features, queries, settings, multiplier=4.0):
    """Change only final LSE responsibilities; keep VIP template/salience scoring."""
    with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16):
        similarities = torch.einsum("bnd,mtd->bnmt", features, queries.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(queries.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0] / settings.tem).float()
        alias_logits = (similarities * settings.logit_scale).T.reshape(-1, 21, 21)
        original, bounded, statistics = [], [], {}
        for index in range(len(queries.class_names)):
            members = queries.parents == index
            weights = salience[members].softmax(0)
            scaled = alias_logits[members] * (weights / weights.mean())[:, None, None]
            original.append(torch.logsumexp(settings.tau * scaled, dim=0) / settings.tau)
            changed, diagnostics = bounded_score(scaled.permute(1, 2, 0).float(), 1.0 / settings.tau,
                                                 multiplier, normalize_count=False)
            bounded.append(changed)
            for field, value in diagnostics.items():
                statistics[field] = statistics.get(field, 0.0) + value / len(queries.class_names)
        original_map = F.interpolate(torch.stack(original)[None], size=(336, 336),
                                     mode="bilinear", align_corners=False)[0].float()
        bounded_map = F.interpolate(torch.stack(bounded)[None], size=(336, 336),
                                    mode="bilinear", align_corners=False)[0].float()
    return original_map, bounded_map, statistics
