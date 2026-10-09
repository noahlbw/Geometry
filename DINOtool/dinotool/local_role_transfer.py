"""Matched three-arm transfer of frozen local-role and FamilySUM rules.

Both uniform and role local reductions use the same vectorized implementation.
No new weighting rule, visual observation, or shuffled control is introduced.
"""
from dataclasses import replace
import math

import torch

from . import local_role_alias as role
from .fixed20_task_readout import POLICIES


IMPLEMENTATION = 'geometry-local-role-transfer-potsdam-voc-v1-20261009'
BASE = 'Same20_NoFamilyWeight'
WIDE = 'Same20_WideFamilyOnly'
PRIMARY = 'Same20_LocalRole_WideFamily'
METHODS = (BASE, WIDE, PRIMARY)
TRANSFER = {'potsdam': 'vdd', 'voc20': 'ade150'}


def uniform_plan(plan):
    return replace(plan, log_prior=torch.full_like(plan.log_prior, -math.log(20)),
                   uniform_classes=(True,) * len(plan.uniform_classes))


@torch.inference_mode()
def predict(image, geometry, vip, bank, query, plan, policy, methods=METHODS):
    from eval_development_readout import observations

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Only the three declared treatment arms are allowed.')
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    wanted = tuple(dict.fromkeys('base' if name == BASE else 'sum' for name in methods))
    fields = role.old.wide_fields(source, query, plan.mass, wanted, policy.tau, policy.tem)
    pooled = uniform_plan(plan)
    cosines = {}
    results = {}
    for name in methods:
        broad = fields['base' if name == BASE else 'sum']
        local_plan = plan if name == PRIMARY else pooled
        probability = role.role_probabilities(source, policy.profile(), bank, broad,
                                              local_plan, role.PRIMARY, cosines)
        results[name] = probability.argmax(0).cpu().numpy()
    return results, dict(geometry_encodings=len(source['local']), wide_encodings=len(source['wide']),
        fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
        alias_slots_per_class=20, zero_local_prior_slots=plan.zero_local_slots,
        maximum_local_alias_elements=1024 * plan.members.numel())

