"""Execution-only caching for the frozen variable-count rival reader."""
from dataclasses import dataclass
import math

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .natural_variable_alias_reader import alias_groups, profile
from .stratified_soft_alias import crop_stencil


@dataclass(frozen=True)
class PreparedObservation:
    values: torch.Tensor
    reference: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor


def prepare_observations(crops, count, coordinates, image_size, groups, beta):
    observations = []
    for crop in crops:
        values = profile(crop, groups)
        reference = torch.stack([
            (beta * values[:, ids]).logsumexp(-1) / beta - math.log(len(ids)) / beta
            for ids in groups
        ], -1)
        indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
        observations.append(PreparedObservation(values, reference, indices, coefficients))
    return observations


def sampled_margins_cached(observations, coordinates, groups, sl):
    output = coordinates.new_zeros((len(coordinates[sl]), sum(map(len, groups)), len(groups)))
    for crop in observations:
        ids, coeff = crop.indices[sl], crop.coefficients[sl]
        # Subtract before interpolation, preserving the original rounding/order.
        margins = crop.values[ids][..., None] - crop.reference[ids][:, :, None]
        output += (margins * coeff[..., None, None]).sum(1)
    return output


def hard_observation_cached(observations, groups, risk, valid, beta, sl):
    output = torch.zeros((len(valid), len(groups), len(groups)), device=risk.device, dtype=torch.float64)
    keeps = [risk[:, members] == 0 for members in groups]
    remaining = [keep.sum(1) for keep in keeps]
    if bool(torch.stack([count.amin() for count in remaining]).eq(0).any()):
        raise ValueError('Every class/rival must retain a canonical alias.')
    normalizers = [(len(members) / count.double()).log()[:, None]
                   for members, count in zip(groups, remaining)]
    untouched = [keep.all(1)[:, None] for keep in keeps]
    for crop in observations:
        ids, coeff = crop.indices[sl], crop.coefficients[sl]
        values = (beta * crop.values[ids]).double()
        for c, members in enumerate(groups):
            evidence = values[:, :, members]
            full = evidence.logsumexp(-1)
            changed = evidence[..., None].masked_fill(~keeps[c][:, None], -torch.inf).logsumexp(2)
            delta = (changed - full[..., None] + normalizers[c]) / beta
            delta = delta.masked_fill(untouched[c], 0.)
            output[:, c] += (delta * coeff.double()[..., None]).sum(1)
    return output.masked_fill(~valid[:, None, None], 0.), remaining


@torch.inference_mode()
def retained_variable_scores_fast(local, operator, broad, wide, wide_count, fine, fine_count,
                                  coordinates, fine_coordinates, valid, parents, canonical,
                                  image_size, beta=1., query_chunk=128):
    if beta <= 0 or query_chunk < 1:
        raise ValueError('Positive aggregation temperature and query block required.')
    groups = alias_groups(parents, local.shape[-1])
    wide = prepare_observations(wide, wide_count, coordinates, image_size, groups, beta)
    fine = prepare_observations(fine, fine_count, fine_coordinates, (512, 512), groups, beta)
    potential = torch.empty_like(local, dtype=torch.float64)
    retained, deleted, actions = [], [], []
    rivals = [torch.arange(len(groups), device=parents.device) != c for c in range(len(groups))]
    for start in range(0, len(valid), query_chunk):
        sl = slice(start, start + query_chunk)
        full = sampled_margins_cached(wide, coordinates, groups, sl)
        native = sampled_margins_cached(fine, fine_coordinates, groups, sl)
        native.masked_fill_(~valid[sl, None, None], 0.)
        risk = native_risk(full, native, native, parents, canonical, valid[sl], CONFIG)
        directed, counts = hard_observation_cached(wide, groups, risk, valid[sl], beta, sl)
        potential[sl], consistency = signed_potential(directed, valid[sl])
        for c, ids in enumerate(groups):
            kept = counts[c][valid[sl]][:, rivals[c]]
            retained.append(kept.flatten())
            deleted.append((kept < len(ids)).flatten())
        actions.append(float(potential[sl].abs().mean()))
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    outputs = {'Geometry': local.double(), 'NoAdmission_Exact': baseline,
               'RivalFineHard_Exact': baseline + operator.double() @ potential}
    if not all(bool(torch.isfinite(v).all()) for v in outputs.values()):
        raise RuntimeError('Nonfinite variable-alias reader output.')
    return outputs, dict(retained_count_mean=float(torch.cat(retained).double().mean()),
                         deleted_alias_rival_fraction=float(torch.cat(deleted).double().mean()),
                         mean_absolute_admission_potential=sum(actions) / len(actions),
                         canonical_risk_max=0., **consistency)
