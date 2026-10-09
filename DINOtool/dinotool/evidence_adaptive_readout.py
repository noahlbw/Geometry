"""Frozen evidence-driven alias weights, branch trust and residual rejection."""
import math
import torch
import torch.nn.functional as F

IMPLEMENTATION = 'geometry-evidence-adaptive-readout-v1-20261007'


def competition_scores(alias, bank, temperature=.07):
    """Canonical anchoring plus rival-conditioned, count-normalized alias evidence."""
    text = F.normalize(bank.features.float(), dim=-1)
    anchors = torch.stack([text[(bank.parent_indices == c) & bank.canonical_mask][0]
                           for c in range(bank.class_count)])
    canonical = torch.stack([alias[:, (bank.parent_indices == c) & bank.canonical_mask][:, 0]
                             for c in range(bank.class_count)], -1)
    semantic = text @ anchors.T
    top = canonical.topk(min(2, bank.class_count), -1).indices
    result = []
    for c in range(bank.class_count):
        members = bank.parent_indices == c
        if bank.class_count == 1:
            weights = torch.ones_like(alias[:, members])
        else:
            rival = torch.where(top[:, 0] == c, top[:, 1], top[:, 0])
            # An alias useful against one rival can retain its evidence elsewhere.
            specificity = semantic[members, c][None] - semantic[members][:, rival].T
            weights = torch.sigmoid(specificity / temperature).clamp_min(1e-6)
        weights = weights / weights.sum(-1, keepdim=True)
        expanded = temperature * torch.logsumexp(alias[:, members] / temperature + weights.log(), -1)
        # Equal canonical/expanded evidence keeps the original semantic anchor.
        result.append(temperature * (torch.logsumexp(torch.stack((canonical[:, c], expanded), -1)
                                                       / temperature, -1) - math.log(2)))
    return torch.stack(result, -1)


def adaptive_coupled(local, broad, operator, relation):
    """Trust combines semantic margin and agreement with existing geometry."""
    reliabilities = []
    for logits in (local, broad):
        probability = logits.float().softmax(-1)
        propagated = relation.float() @ probability
        propagated /= propagated.sum(-1, keepdim=True).clamp_min(1e-12)
        midpoint = (probability + propagated) / 2
        js = .5 * ((probability * (probability.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
                   + (propagated * (propagated.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1))
        values = probability.topk(min(2, probability.shape[-1]), -1).values
        margin = values[:, 0] - values[:, 1] if values.shape[-1] > 1 else values[:, 0]
        reliabilities.append(margin * (1 - js / math.log(2)).clamp(0, 1))
    a, b = reliabilities
    gain = torch.where(a + b > 1e-8, 2 * b / (a + b).clamp_min(1e-8), torch.ones_like(a))
    correction = operator.double() @ (broad.double() - local.double())
    return local.double() + gain[:, None] * correction, gain


def residual_prediction(probability, background):
    """Image-only BIC chooses whether foreground-confidence has two modes.

    The low-confidence component is treated as residual. This is a hypothesis,
    not a labelled background estimator; full precision/recall must test it.
    """
    prediction = probability.argmax(0)
    if background is None:
        return prediction, dict(rejection=False)
    keep = torch.arange(probability.shape[0], device=probability.device) != background
    confidence = probability[keep].amax(0)
    x = F.interpolate(confidence[None, None], (64, 64), mode='area').flatten().double()
    eps = 1e-6
    single_mu, single_var = x.mean(), x.var(unbiased=False).clamp_min(eps)
    single_ll = (-.5 * ((x - single_mu).square() / single_var + (2 * math.pi * single_var).log())).sum()
    mu = x.quantile(torch.tensor([.25, .75], device=x.device, dtype=x.dtype))
    variance = single_var.repeat(2)
    prior = torch.full((2,), .5, device=x.device, dtype=x.dtype)
    for _ in range(20):
        logp = -.5 * ((x[:, None] - mu).square() / variance + (2 * math.pi * variance).log()) + prior.clamp_min(eps).log()
        responsibility = logp.softmax(-1)
        counts = responsibility.sum(0).clamp_min(eps)
        mu = (responsibility * x[:, None]).sum(0) / counts
        variance = (responsibility * (x[:, None] - mu).square()).sum(0) / counts
        variance = variance.clamp_min(eps)
        prior = counts / len(x)
    mixture_ll = torch.logsumexp(-.5 * ((x[:, None] - mu).square() / variance + (2 * math.pi * variance).log())
                                 + prior.clamp_min(eps).log(), -1).sum()
    use = bool(-2 * mixture_ll + 5 * math.log(len(x)) < -2 * single_ll + 2 * math.log(len(x)))
    if use:
        logp = -.5 * ((confidence.double()[..., None] - mu).square() / variance + (2 * math.pi * variance).log()) + prior.clamp_min(eps).log()
        rejected = logp.argmax(-1) == mu.argmin()
        prediction = prediction.masked_fill(rejected, background)
    return prediction, dict(rejection=use, component_means=mu.tolist(), component_weights=prior.tolist())
