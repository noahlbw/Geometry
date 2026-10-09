"""All-class soft alias admission from already-computed visual observations."""
from dataclasses import dataclass
import math

import torch

from .sparse_alias_reuse import profile_aliases
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-shared-rival-soft-alias-v1-20261005'
PRIMARY = 'SharedRivalSoft'
METHODS = ('Geometry', 'NoAdmission_Exact', PRIMARY, 'SharedRivalHard', 'SharedRivalShuffle')


@dataclass(frozen=True)
class SharedRivalConfig:
    beta: float = 1.
    epsilon: float = 1e-6
    hard_retention: float = .5
    query_chunk: int = 128
    shuffle_seed: int = 20261005


CONFIG = SharedRivalConfig()


def rival_reference(evidence, layout, beta=CONFIG.beta):
    """Stable leave-class-out LME, without a token x alias x rival array."""
    classes = evidence.shape[-2]
    if classes < 2 or beta <= 0 or evidence.shape[-2:] != layout.members.shape:
        raise ValueError('At least two matching alias groups and positive beta required.')
    logits = (beta * evidence.double()).logsumexp(-1) - layout.counts.double().log()
    prefix = logits.logcumsumexp(-1)
    suffix = logits.flip(-1).logcumsumexp(-1).flip(-1)
    blank = torch.full_like(logits[..., :1], -torch.inf)
    left = torch.cat((blank, prefix[..., :-1]), -1)
    right = torch.cat((suffix[..., 1:], blank), -1)
    return (torch.logaddexp(left, right) - math.log(classes - 1)) / beta


def positive_margin(evidence, layout, beta=CONFIG.beta):
    return (beta * (evidence.double() - rival_reference(evidence, layout, beta)[..., None])).tanh().clamp_min(0)


@dataclass(frozen=True)
class CachedWide:
    crop: object
    log_responsibility: torch.Tensor
    positive: torch.Tensor


def cache_wide(crops, layout, config=CONFIG):
    result = []
    for crop in crops:
        evidence = profile_aliases(crop.alias_logits, crop.salience, layout).double()
        logits = config.beta * evidence
        result.append(CachedWide(crop, logits.log_softmax(-1), positive_margin(evidence, layout, config.beta)))
    return result


def alias_permutation(layout, seed=CONFIG.shuffle_seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.arange(layout.members.shape[-1]).expand_as(layout.members).clone().cpu()
    for c in range(len(permutation)):
        eligible = (layout.valid[c] & ~layout.canonical[c]).nonzero().flatten().cpu()
        permutation[c, eligible] = eligible[torch.randperm(len(eligible), generator=generator)]
    return permutation.to(layout.members.device)


def local_contradiction(native, geometry, layout, config=CONFIG):
    n = (-config.beta * (native.double() - rival_reference(native, layout, config.beta)[..., None])).tanh().clamp_min(0)
    g = (-config.beta * (geometry.double() - rival_reference(geometry, layout, config.beta)[..., None])).tanh().clamp_min(0)
    return torch.minimum(n, g).masked_fill(~layout.valid[None], 0.)


def weights_from_risk(risk, layout, valid, config=CONFIG):
    eligible = layout.valid[None] & ~layout.canonical[None] & valid[:, None, None]
    risk = risk.clamp(0, 1).masked_fill(~eligible, 0.)
    return (1 - risk).clamp_min(config.epsilon).masked_fill(~layout.valid[None], 0.)


def normalized_delta(log_responsibility, weights, layout):
    """Change in weighted LME: equal evidence/constant class weights are neutral."""
    normalizer = layout.counts.double().log() - weights.sum(-1).log()
    result = (log_responsibility + weights.log()[:, None]).logsumexp(-1) + normalizer[:, None]
    constant = ((weights == weights[..., :1]) | ~layout.valid[None]).all(-1)
    return result.masked_fill(constant[:, None], 0.)


@torch.inference_mode()
def shared_scores(local, operator, broad, observations, count, coordinates, image_size,
                  native, geometry, valid, layout, *, methods=(PRIMARY,), config=CONFIG,
                  permutation=None, diagnostics=True):
    if (local.shape != broad.shape or native.shape != geometry.shape
            or native.shape != (len(local), *layout.members.shape)
            or operator.shape != (len(local), len(local)) or not observations
            or not set(methods).issubset(METHODS[2:])):
        raise ValueError('Matching frozen observations and declared admission methods required.')
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    contradiction = local_contradiction(native, geometry, layout, config)
    permutation = alias_permutation(layout, config.shuffle_seed) if permutation is None else permutation
    stencils = [(observation, *crop_stencil(observation.crop, count, coordinates, image_size))
                for observation in observations]
    deltas = {m: torch.zeros_like(baseline) for m in methods}
    risk_sum = changed_sum = eligible_count = 0.
    max_elements = 0
    canonical_error = 0.
    for start in range(0, len(local), config.query_chunk):
        end = min(start + config.query_chunk, len(local))
        sampled = []
        wide_positive = torch.zeros_like(contradiction[start:end])
        for observation, indices, coefficients in stencils:
            indices, coefficients = indices[start:end], coefficients[start:end].double()
            wide_positive += (observation.positive[indices] * coefficients[..., None, None]).sum(1)
            sampled.append((observation.log_responsibility[indices], coefficients))
            max_elements = max(max_elements, sampled[-1][0].numel())
        weights = weights_from_risk(wide_positive * contradiction[start:end], layout, valid[start:end], config)
        choices = {PRIMARY: weights,
                   'SharedRivalHard': (weights >= config.hard_retention).double().masked_fill(~layout.valid[None], 0.),
                   'SharedRivalShuffle': weights.gather(-1, permutation[None].expand_as(weights))}
        for method in methods:
            allocation = choices[method]
            for response, coefficients in sampled:
                change = normalized_delta(response, allocation, layout) / config.beta
                deltas[method][start:end] += (change * coefficients[..., None]).sum(1)
        if diagnostics:
            eligible = layout.valid[None] & ~layout.canonical[None] & valid[start:end, None, None]
            eligible_count += float(eligible.sum())
            risk_sum += float((1 - weights)[eligible].sum())
            changed_sum += float((weights[eligible] < 1).sum())
            canonical_error = max(canonical_error, float((weights[:, layout.canonical] - 1).abs().max()))
    output = {m: baseline + operator.double() @ delta.masked_fill(~valid[:, None], 0.) for m, delta in deltas.items()}
    if any(not bool(torch.isfinite(value).all()) for value in output.values()):
        raise RuntimeError('Nonfinite shared-rival admission.')
    stats = dict(fine_forwards=0., additional_visual_forwards=0., class_coverage=float(local.shape[1]),
                 per_alias_rival_axis=0., maximum_sampled_alias_elements=float(max_elements),
                 canonical_weight_max_error=canonical_error,
                 mean_noncanonical_risk=risk_sum / max(eligible_count, 1),
                 changed_noncanonical_fraction=changed_sum / max(eligible_count, 1)) if diagnostics else {}
    return output, stats
