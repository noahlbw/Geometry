"""Prior-developed task profiles, without alias-risk or additional visual heads.

These profiles are configuration controls, not a new alias mechanism. Exact
strings and the original Geometry writeback operator are retained.
"""
from dataclasses import dataclass

import torch

from .development_readout import Profile


IMPLEMENTATION = 'geometry-fixed-original20-task-profile-v1-20261008'
PRIMARY = 'Fixed20_TaskCoupled'
VIP = 'VIP_Task20'


@dataclass(frozen=True)
class TaskPolicy:
    template: str
    strength: object
    wide_policy: str
    coupling: float = .5
    temperature: float = .07
    tau: float = 1.
    tem: float = 1.

    def profile(self):
        return Profile(bank='task', strength=self.strength, coupling=self.coupling,
                       temperature=self.temperature, tau=self.tau, tem=self.tem)


POLICIES = {
    'vdd': TaskPolicy('openai_imagenet_template', 1., 'long448'),
    'ade150': TaskPolicy('seg_template', 'original', 'natural_short336_cap672'),
}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, dataset):
    from eval_development_readout import observations, probabilities, wide_scores

    policy = POLICIES[dataset]
    source = observations(image, geometry, vip, (policy.strength,),
                          wide_policy=policy.wide_policy)
    predictions = {}
    for name, bank in banks.items():
        broad = wide_scores(source, queries[name], policy.tau, policy.tem)
        probability = probabilities(source, policy.profile(), bank, broad, {})
        predictions[name] = {PRIMARY: probability.argmax(0).cpu().numpy()}
    diagnostics = {name: dict(geometry_encodings=len(source['local']),
        wide_encodings=len(source['wide']), fine_forwards=0,
        additional_visual_forwards=0, additional_semantic_heads=0,
        wide_policy=policy.wide_policy, alias_count=20) for name in banks}
    return predictions, diagnostics
