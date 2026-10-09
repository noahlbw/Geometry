"""Frozen class-attachment source with the unchanged excess-evidence writer."""
import torch

from .native_alias_noise import METHODS as LEGACY, GATE as PREVIOUS_GATE
from .native_class_ownership import ownership_risk, risk_union
from .native_query_alias import CONFIG
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'geometry-native-class-ownership-reader-v1-20261003'
PRIMARY = 'OwnershipJoint_Exact'
REFERENCE_SHUFFLES = tuple('OwnershipReferenceShuffle'+str(i)+'_Exact' for i in range(3))
ALIAS_SHUFFLES = tuple('OwnershipAliasShuffle'+str(i)+'_Exact' for i in range(3))
SPATIAL_SHUFFLES = tuple('OwnershipDirectionalShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = (*LEGACY, PRIMARY, 'OwnershipOnly_Exact', 'OwnershipTextUnion_Exact',
    'OwnershipNoHoldout_Exact', *REFERENCE_SHUFFLES, *ALIAS_SHUFFLES,
    'OwnershipMeanLogit', 'OwnershipHardDelete_Exact', 'OwnershipDirectionalMean_Exact', *SPATIAL_SHUFFLES)
GATE_CONTROLS = (*LEGACY[2:], *METHODS[len(LEGACY)+1:])
GATE = {**PREVIOUS_GATE, 'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'fixed_source_feasibility_required': True, 'source_only_pass_is_not_promotion': True}


def query_permutation(valid, seed):
    indices = valid.nonzero().flatten()
    output = torch.arange(len(valid), device=valid.device)
    generator = torch.Generator().manual_seed(seed)
    output[indices] = indices[torch.randperm(len(indices), generator=generator).to(valid.device)]
    return output


def source_controls(view, attachment, field, known, assignments, parents, canonical,
                    conflict, valid, members, no_holdout, no_holdout_known):
    joint = risk_union(view, attachment)
    text = conflict[None].expand_as(view).masked_fill(~valid[:, None, None], 0.)
    text = text.clone()
    text[:, canonical] = 0.
    text.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 0.)
    self_ids = torch.zeros_like(assignments)
    nonheld = ownership_risk(no_holdout, no_holdout_known, self_ids, parents, canonical, conflict, valid)
    output = {PRIMARY: joint, 'OwnershipOnly_Exact': attachment.to(view.dtype),
        'OwnershipTextUnion_Exact': risk_union(view, text),
        'OwnershipNoHoldout_Exact': risk_union(view, nonheld)}
    for i, method in enumerate(REFERENCE_SHUFFLES):
        shuffled = field[query_permutation(valid, CONFIG.random_seed+i)]
        changed = ownership_risk(shuffled, known, assignments, parents, canonical, conflict, valid)
        output[method] = risk_union(view, changed)
    grouped = joint[:, members]
    for method, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members, canonical, CONFIG.random_seed)):
        shuffled = torch.empty_like(joint)
        shuffled[:, members] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
        output[method] = shuffled
    return output
