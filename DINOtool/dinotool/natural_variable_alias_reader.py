"""The retained reader equations for actual, unequal alias group sizes."""
import math

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .stratified_soft_alias import crop_stencil


def alias_groups(parents, classes):
    groups = tuple((parents == c).nonzero().flatten() for c in range(classes))
    if any(not len(g) for g in groups):
        raise ValueError('Every class needs at least one actual alias.')
    if torch.cat(groups).tolist() != list(range(len(parents))):
        raise ValueError('Class-contiguous alias order required.')
    return groups


def profile(crop, groups):
    output = torch.empty_like(crop.alias_logits)
    for ids in groups:
        output[:, ids] = crop.alias_logits[:, ids] * (len(ids) * crop.salience[ids].softmax(0))
    return output


def margins(crop, groups, beta=1.):
    values = profile(crop, groups)
    reference = torch.stack([(beta * values[:, ids]).logsumexp(-1) / beta
                             - math.log(len(ids)) / beta for ids in groups], -1)
    return values[..., None] - reference[:, None]


def sampled_margins(crops, count, coordinates, image_size, groups, beta):
    output = coordinates.new_zeros((len(coordinates), sum(map(len, groups)), len(groups)))
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, image_size)
        output += (margins(crop, groups, beta)[ids] * coeff[..., None, None]).sum(1)
    return output


def hard_observation(crops, count, coordinates, image_size, groups, risk, valid, beta):
    output = torch.zeros((len(valid), len(groups), len(groups)), device=risk.device, dtype=torch.float64)
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, image_size)
        values = (beta * profile(crop, groups)[ids]).double()
        for c, members in enumerate(groups):
            keep = risk[:, members] == 0
            remaining = keep.sum(1)
            if bool((remaining == 0).any()):
                raise ValueError('Every class/rival must retain a canonical alias.')
            evidence = values[:, :, members]
            full = evidence.logsumexp(-1)
            changed = evidence[..., None].masked_fill(~keep[:, None], -torch.inf).logsumexp(2)
            delta = (changed - full[..., None]
                     + (len(members) / remaining.double()).log()[:, None]) / beta
            delta = delta.masked_fill(keep.all(1)[:, None], 0.)
            output[:, c] += (delta * coeff.double()[..., None]).sum(1)
    return output.masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def retained_variable_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                             coordinates, fine_coordinates, valid, parents, canonical,
                             image_size, beta=1., query_chunk=128):
    if beta <= 0 or query_chunk < 1:
        raise ValueError('Positive aggregation temperature and query block required.')
    groups = alias_groups(parents, local.shape[-1])
    potential = torch.empty_like(local, dtype=torch.float64)
    retained, deleted, actions = [], [], []
    for start in range(0, len(valid), query_chunk):
        sl = slice(start, start + query_chunk)
        full = sampled_margins(wide, wide_count, coordinates[sl], image_size, groups, beta)
        native = sampled_margins(fine, fine_count, fine_coordinates[sl], (512, 512), groups, beta)
        native.masked_fill_(~valid[sl, None, None], 0.)
        risk = native_risk(full, native, native, parents, canonical, valid[sl], CONFIG)
        directed = hard_observation(wide, wide_count, coordinates[sl], image_size,
                                    groups, risk, valid[sl], beta)
        potential[sl], consistency = signed_potential(directed, valid[sl])
        for c, ids in enumerate(groups):
            rivals = torch.arange(len(groups), device=parents.device) != c
            counts = (risk[valid[sl]][:, ids] == 0).sum(1)[:, rivals]
            retained.append(counts.flatten())
            deleted.append((counts < len(ids)).flatten())
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
