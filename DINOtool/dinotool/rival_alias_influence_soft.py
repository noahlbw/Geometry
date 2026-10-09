"""Readout-responsibility-aware attenuation with the unchanged fine-risk source."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .rival_alias_fast import cache_observations


IMPLEMENTATION = 'frozen-geometry-rival-influence-soft-v1-20261005'
PRIMARY = 'FineInfluence_Weighted'


def influence_weights(responsibility, risk, epsilon=CONFIG.epsilon):
    if (responsibility.ndim != 4 or risk.ndim != 4 or risk.shape[:3] != (len(responsibility), *responsibility.shape[2:])
            or not bool(torch.isfinite(responsibility).all() and torch.isfinite(risk).all())
            or bool(((responsibility < 0) | (responsibility > 1)).any())
            or bool(((risk < 0) | (risk > 1)).any()) or not 0 < epsilon < 1):
        raise ValueError('Finite [query,stencil,class,alias] responsibilities and [query,class,alias,rival] risks required.')
    pi = responsibility.double().clamp_max(1-epsilon)[..., None]
    r = risk.double()[:, None]
    other_mass = 1-pi
    return (1-r)*other_mass/(other_mass+r*pi)


def influence_directed(observations, members, risk, valid, beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    output = torch.zeros(len(valid), len(members), len(members), device=risk.device, dtype=torch.float64)
    k = members.shape[-1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        grouped = risk[sl, members]
        for crop in observations:
            ids, coeff = crop.indices[sl], crop.coefficients[sl]
            evidence = (beta*crop.evidence[ids]).double()
            full = evidence.logsumexp(-1)
            weights = influence_weights(evidence.softmax(-1), grouped)
            mass = weights.sum(3)
            if bool((mass <= 0).any()):
                raise RuntimeError('A protected canonical alias must keep positive mass.')
            changed = (evidence[..., None]+weights.log()).logsumexp(3)
            delta = (changed-full[..., None]+(k/mass).log())/beta
            delta.masked_fill_((grouped == 0).all(2)[:, None], 0.)
            output[sl] += (delta*coeff.double()[..., None, None]).sum(1)
    return output.masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def soft_scores(local, operator, broad, wide, wide_count, coordinates, valid, members,
                canonical, parents, image_size, local_aliases, fine_risk=None, methods=(PRIMARY,), observations=None):
    if tuple(methods) != (PRIMARY,) or fine_risk is None:
        raise ValueError('One declared method with unchanged fine risk required.')
    if observations is None:
        observations = cache_observations(wide, wide_count, coordinates, image_size, members)
    directed = influence_directed(observations, members, fine_risk, valid)
    potential, consistency = signed_potential(directed, valid)
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    scores = baseline+operator.double()@potential
    if not bool(torch.isfinite(scores).all()) or bool((fine_risk[:, canonical] != 0).any()):
        raise RuntimeError('Nonfinite scores or changed canonical protection.')
    return {PRIMARY: scores}, {PRIMARY: {**consistency,
        'canonical_risk_max': float(fine_risk[:, canonical].abs().max()),
        'mean_absolute_admission_potential': float(potential.abs().mean())}}
