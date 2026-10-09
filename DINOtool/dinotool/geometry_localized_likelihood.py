"""Geometry-conditioned localization likelihoods, not box-to-label painting.

Bayesian likelihood updates and uniform pseudocounts are established machinery.
Useful coupling and originality are separate experimental questions.
"""
from dataclasses import asdict, dataclass

import torch

from .grounded_local_observer import fractional_box_support


IMPLEMENTATION = "geometry-localized-alias-likelihood-v1-20261002"
PRIMARY = "Geometry_LocalLikelihood"
METHODS = ("Geometry", "BoxLocalLikelihood", "MeanProb_BoxLikelihood", "ShuffledLocalLikelihood", PRIMARY)


@dataclass(frozen=True)
class LocalLikelihoodConfig:
    confidence: float = .25
    temperature: float = .07
    uniform_pseudocount: float = 1.
    box_chunk: int = 128

    def signature(self):
        return {**asdict(self), "implementation": IMPLEMENTATION,
            "support": "fractional box intersection times Geometry affinity to box-intersecting valid patches",
            "density": "D_r=N*h_r/sum(h_r), each observed localization has spatial mean1",
            "alias": "e_a=(1+sum_r(score_ra*D_r))/(1+sum_r(score_ra)), at fixed score>0.25",
            "class": "arithmetic mean of all20 alias localization likelihoods, no deletion",
            "prediction": "original cosine/LME scores + .07*log(class density)",
            "uninformative": "uniform whole-window support and empty observations are exactly neutral",
            "identity": "G=I recovers same-source BoxLocalLikelihood exactly",
            "absence": "no detections for an alias supplies density1, never an absence veto",
            "outside_boxes": "positive location observations redistribute relative class likelihood; "
                "lower likelihood outside observed boxes is not a certificate of class absence",
            "precision": "FP32 likelihoods with unchanged frozen Geometry and detector weights",
            "supervision": "additional frozen detection pretraining; not a matched single-encoder model"}


def class_localization_density(rows, parents, classes, valid, relation=None, config=LocalLikelihoodConfig()):
    if (valid.ndim != 1 or valid.dtype != torch.bool or not bool(valid.any())
            or not parents or set(parents) != set(range(classes)) or config.box_chunk < 1
            or config.uniform_pseudocount != 1. or config.confidence != .25 or config.temperature != .07):
        raise ValueError("Invalid/frozen localization likelihood inputs.")
    count, patches = int(valid.sum()), len(valid)
    side = int(patches**.5)
    if side*side != patches:
        raise ValueError("Expected square original patch grid.")
    device = valid.device
    geometry = None
    if relation is not None:
        if relation.shape != (patches, patches) or not torch.isfinite(relation).all() or bool((relation < 0).any()):
            raise ValueError("Invalid Geometry relation.")
        geometry = relation[valid][:, valid].float()
        if bool((geometry.sum(-1) <= 0).any()):
            raise ValueError("Empty valid Geometry row.")
        geometry = geometry/geometry.sum(-1, keepdim=True)
        if torch.equal(geometry, torch.eye(count, device=device)):
            geometry = None
    aliases = len(parents)
    residual = torch.zeros(count, aliases, device=device)
    denominator = torch.ones(aliases, device=device)
    observed = torch.zeros(aliases, dtype=torch.bool, device=device)
    seen = []
    total_boxes = 0
    for row in rows:
        indices = torch.as_tensor(row["alias_indices"], dtype=torch.long, device=device)
        boxes = torch.as_tensor(row["boxes"], dtype=torch.float32, device=device)
        scores = torch.as_tensor(row["alias_scores"], dtype=torch.float32, device=device)
        if (scores.shape != (len(boxes), len(indices)) or not torch.isfinite(scores).all()
                or bool(((scores < 0) | (scores > 1)).any()) or bool(((indices < 0) | (indices >= aliases)).any())):
            raise ValueError("Invalid localized alias observations.")
        seen.extend(indices.tolist())
        weights = torch.where(scores > config.confidence, scores, 0.)
        active = weights.any(-1)
        supports = None
        if "supports" in row:
            supports = torch.as_tensor(row["supports"], dtype=torch.float32, device=device)
            if (supports.shape != (len(boxes), patches) or not torch.isfinite(supports).all()
                    or bool(((supports < 0) | (supports > 1)).any())):
                raise ValueError("Invalid externally observed patch supports.")
            supports = supports[active]
        boxes, weights = boxes[active], weights[active]
        for start in range(0, len(boxes), config.box_chunk):
            support = (fractional_box_support(boxes[start:start+config.box_chunk], side)
                       if supports is None else supports[start:start+config.box_chunk])[:, valid]
            if geometry is not None:
                affinity = (support > 0).float() @ geometry.T
                affinity[(support > 0).all(-1)] = 1.
                support = support*affinity
            mass = support.sum(-1)
            usable = mass > 0
            support, mass = support[usable], mass[usable]
            weight = weights[start:start+config.box_chunk][usable]
            if not len(mass):
                continue
            density = count*support/mass[:, None]
            # Centered accumulation keeps exactly uniform observations bit-exact neutral.
            constant = (support == support[:, :1]).all(-1)
            density[constant] = 1.
            residual[:, indices] += (density-1).T @ weight
            denominator[indices] += weight.sum(0)
            observed[indices] |= weight.any(0)
            total_boxes += int(usable.sum())
    if sorted(seen) != list(range(aliases)):
        raise ValueError("Every unchanged alias must be observed exactly once.")
    alias_density = 1+residual/denominator[None]
    parent = torch.as_tensor(parents, dtype=torch.long, device=device)
    local = torch.stack([alias_density[:, parent == c].mean(-1) for c in range(classes)], -1)
    result = torch.ones(patches, classes, device=device)
    result[valid] = local
    if not torch.isfinite(result).all() or not bool((result > 0).all()):
        raise RuntimeError("Invalid positive localization likelihood.")
    return result, {"observed_alias_fraction": float(observed.float().mean()),
                    "qualified_box_queries": total_boxes,
                    "spatial_density_mean_max_error": float((local.mean(0)-1).abs().max()),
                    "mean_absolute_log_density": float(local.log().abs().mean())}


def apply_localized_likelihood(scores, density, config=LocalLikelihoodConfig()):
    if scores.shape != density.shape or not torch.isfinite(scores).all():
        raise ValueError("Geometry and density shape/value mismatch.")
    if not torch.isfinite(density).all() or not bool((density > 0).all()):
        raise ValueError("Localization densities must be finite and positive.")
    return scores + config.temperature*density.log()
