"""One multiclass contextual observation from native family-excluded evidence."""
from dataclasses import dataclass
import math

import torch

from .calibrated_competitive_alias import profiled_logits
from .native_ownership_reader import METHODS as LEGACY, query_permutation
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-coherent-native-admission-v1-20261003'
PRIMARY = 'CoherentNativeJoint_Exact'
REFERENCE_SHUFFLES = tuple('CoherentReferenceShuffle'+str(i)+'_Exact' for i in range(3))
ALIAS_SHUFFLES = tuple('CoherentAliasShuffle'+str(i)+'_Exact' for i in range(3))
ACTION_SHUFFLES = tuple('CoherentActionShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'CoherentNativeOnly_Exact', 'CoherentViewOnly_Exact',
    'CoherentNoHoldout_Exact', 'CoherentProtected_Exact', *REFERENCE_SHUFFLES,
    *ALIAS_SHUFFLES, 'CoherentMeanLogit', 'CoherentHardDelete_Exact',
    'CoherentActionMean_Exact', *ACTION_SHUFFLES)
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = ('NativeAlias_Exact', 'OwnershipJoint_Exact', 'HardDelete_Exact',
    'OwnershipHardDelete_Exact', 'MeanLogit_Original', 'CoherentNativeOnly_Exact',
    'CoherentViewOnly_Exact', 'CoherentNoHoldout_Exact', 'CoherentProtected_Exact',
    *REFERENCE_SHUFFLES, *ALIAS_SHUFFLES, 'CoherentMeanLogit',
    'CoherentHardDelete_Exact', 'CoherentActionMean_Exact', *ACTION_SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_domain_wins': 5, 'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'development_windows_only': True, 'no_post_result_control_promotion': True,
    'full_image_implementation_must_be_frozen_before_rollout': True}


@dataclass(frozen=True)
class CoherentAdmissionConfig:
    beta: float = 1.
    query_chunk: int = 128
    random_seed: int = 20261003


CONFIG = CoherentAdmissionConfig()


def native_risk(field, known, assignments, parents, valid):
    n, families, classes = field.shape
    if (known.shape != (families, classes) or known.dtype != torch.bool
            or assignments.shape != parents.shape or valid.shape != (n,)
            or valid.dtype != torch.bool or classes < 2
            or not bool(torch.isfinite(field).all())):
        raise ValueError('Finite family references, known classes and alias ownership required.')
    reference = field[:, assignments].double()
    observed = known[assignments]
    own = reference.gather(-1, parents[None, :, None].expand(n, -1, 1))[..., 0]
    own_known = observed.gather(-1, parents[:, None])[:, 0]
    rival_known = observed.clone()
    rival_known.scatter_(-1, parents[:, None], False)
    rivals = reference.masked_fill(~rival_known[None], -torch.inf).amax(-1)
    supported = own_known & rival_known.any(-1)
    margin = torch.where(supported[None], own-rivals, 0.)
    risk = -torch.expm1(margin.clamp_max(0.))
    return risk.masked_fill(~valid[:, None], 0.)


def union_risk(view, native):
    if view.shape != native.shape or not bool(torch.isfinite(view).all() and torch.isfinite(native).all()):
        raise ValueError('Matching finite scalar alias risks required.')
    if bool(((view < 0) | (view > 1) | (native < 0) | (native > 1)).any()):
        raise ValueError('Risks must lie in [0,1].')
    return view.double()+(1-view.double())*native.double()


def source_controls(view_pair, field, known, assignments, parents, canonical,
                    valid, members, no_holdout, no_holdout_known, config=CONFIG):
    view = view_pair.double().amax(-1)
    native = native_risk(field, known, assignments, parents, valid)
    joint = union_risk(view, native)
    protected = joint.clone()
    protected[:, canonical] = 0.
    nonheld = native_risk(no_holdout, no_holdout_known, torch.zeros_like(assignments), parents, valid)
    output = {PRIMARY: joint, 'CoherentNativeOnly_Exact': native,
        'CoherentViewOnly_Exact': view, 'CoherentNoHoldout_Exact': union_risk(view, nonheld),
        'CoherentProtected_Exact': protected}
    for i, name in enumerate(REFERENCE_SHUFFLES):
        index = query_permutation(valid, config.random_seed+i)
        output[name] = union_risk(view, native_risk(field[index], known, assignments, parents, valid))
    generator = torch.Generator().manual_seed(config.random_seed)
    grouped = joint[:, members]
    for name in ALIAS_SHUFFLES:
        permutation = torch.stack([torch.randperm(members.shape[-1], generator=generator)
                                   for _ in members]).to(members.device)
        changed = torch.empty_like(joint)
        changed[:, members] = grouped.gather(-1, permutation[None].expand_as(grouped))
        output[name] = changed
    return output


def coherent_observation(crops, count, coordinates, image_size, members, risk, valid, config=CONFIG):
    if (risk.shape != (len(valid), members.numel()) or config.beta <= 0 or config.query_chunk < 1
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Bounded scalar alias risks and positive writer configuration required.')
    action = torch.zeros(len(valid), len(members), dtype=torch.float64, device=risk.device)
    cap = torch.zeros_like(action)
    k = members.shape[-1]
    for start in range(0, len(valid), config.query_chunk):
        sl = slice(start, start+config.query_chunk)
        for crop in crops:
            ids, coeff = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = profiled_logits(crop, members)[ids].double()
            excess = ((config.beta*evidence).softmax(-1)-1/k).clamp_min(0.)
            removed = (excess*risk[sl, members][:, None]).sum(-1).clamp_max(1-1/k)
            action[sl] += (torch.log1p(-removed)/config.beta*coeff.double()[..., None]).sum(1)
            capacity = -torch.log1p(-excess.sum(-1).clamp_max(1-1/k))/config.beta
            cap[sl] += (capacity*coeff.double()[..., None]).sum(1)
    action.masked_fill_(~valid[:, None], 0.)
    cap.masked_fill_(~valid[:, None], 0.)
    error = float((-action-cap).clamp_min(0.).max())
    if not bool(torch.isfinite(action).all()) or error > 1e-10:
        raise RuntimeError('Coherent class action exceeded its unrestricted excess capacity.')
    stats = {'capacity_excess_max': error, 'maximum_suppression': float(-action.min()),
        'mean_suppression': float(-action[valid].mean()) if bool(valid.any()) else 0.,
        'mean_risk': float(risk[valid].mean()) if bool(valid.any()) else 0.,
        'active_alias_fraction': float((risk[valid] > 0).double().mean()) if bool(valid.any()) else 0.,
        'unrestricted_theoretical_bound': math.log(k)/config.beta}
    return action, cap, stats


def hard_observation(crops, count, coordinates, image_size, members, risk, valid, config=CONFIG):
    action = torch.zeros(len(valid), len(members), dtype=torch.float64, device=risk.device)
    keep, k = risk[:, members] == 0, members.shape[-1]
    remaining = keep.sum(-1)
    usable = (remaining > 0) & (remaining < k)
    for start in range(0, len(valid), config.query_chunk):
        sl = slice(start, start+config.query_chunk)
        # No surviving observation is unknown: do not turn it into a -inf class score.
        safe = keep[sl] | ~usable[sl, :, None]
        for crop in crops:
            ids, coeff = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = (config.beta*profiled_logits(crop, members)[ids]).double()
            delta = (evidence.masked_fill(~safe[:, None], -torch.inf).logsumexp(-1)
                -evidence.logsumexp(-1)+(k/remaining[sl].clamp_min(1).double()).log()[:, None])/config.beta
            delta.masked_fill_(~usable[sl, None], 0.)
            action[sl] += (delta*coeff.double()[..., None]).sum(1)
    return action.masked_fill(~valid[:, None], 0.), {'empty_class_fraction': float((remaining[valid] == 0).double().mean())}


def action_controls(action, valid, config=CONFIG):
    output = {'CoherentActionMean_Exact': torch.zeros_like(action)}
    if bool(valid.any()):
        output['CoherentActionMean_Exact'][valid] = action[valid].mean(0)
    for i, name in enumerate(ACTION_SHUFFLES):
        output[name] = action[query_permutation(valid, config.random_seed+i)]
    return output
