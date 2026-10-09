"""Parameter-free branch-evidence envelope for frozen geometric correction.

Projection is in per-branch normalized log-probability coordinates. It bounds
unsupported extrapolation, not prediction error: two wrong observers remain
wrong. This is one projection of the retained reconstruction, not a claim to
solve a box-constrained geometry energy to convergence.
"""
import torch

IMPLEMENTATION='geometry-normalized-evidence-envelope-v1-20261007'


def envelope_coupled(local,broad,operator,gain):
    if local.ndim!=2 or local.shape!=broad.shape or operator.shape!=(len(local),len(local)):
        raise ValueError('Matching [patch,class] evidence and square geometry operator required.')
    if gain not in (.5,1.,2.):
        raise ValueError('Reuse frozen gain menu only.')
    a=local.double().log_softmax(-1)
    b=broad.double().log_softmax(-1)
    proposal=a+gain*(operator.double()@(b-a))
    lower,upper=torch.minimum(a,b),torch.maximum(a,b)
    output=proposal.clamp(min=lower,max=upper)
    return output
