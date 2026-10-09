"""Local-only ablation: inherited all20 wide scores are exactly retained."""
import torch

from . import local_role_transfer as previous


IMPLEMENTATION = 'geometry-local-role-raw-wide-transfer-v1-20261009'
BASE = previous.BASE
PRIMARY = 'Same20_LocalRole_RawWide'
METHODS = (BASE,PRIMARY)
TRANSFER, POLICIES, role = previous.TRANSFER, previous.POLICIES, previous.role


@torch.inference_mode()
def predict(image, geometry, vip, bank, query, plan, policy, methods=METHODS):
    from eval_development_readout import observations, wide_scores

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Only uniform and frozen local-role/raw-wide arms allowed.')
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    broad = wide_scores(source,query,policy.tau,policy.tem)
    pooled = previous.uniform_plan(plan)
    cosines = {};results = {}
    for name in methods:
        local_plan = plan if name == PRIMARY else pooled
        probability = role.role_probabilities(source,policy.profile(),bank,broad,local_plan,role.PRIMARY,cosines)
        results[name] = probability.argmax(0).cpu().numpy()
    return results,dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
        fine_forwards=0,additional_visual_forwards=0,additional_semantic_heads=0,
        alias_slots_per_class=20,zero_local_prior_slots=plan.zero_local_slots,
        maximum_local_alias_elements=1024*plan.members.numel())
