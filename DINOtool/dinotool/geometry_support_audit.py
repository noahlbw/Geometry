"""Diagnostic only: unsigned leave-self-out influence, not signed correction."""
import torch


def relation_support(local,wide,operator):
    if local.ndim!=2 or wide.shape!=local.shape or operator.shape!=(len(local),len(local)):
        raise ValueError('Matched patch/class scores and square relation required.')
    a,b,h=local.double(),wide.double(),operator.double()
    if not all(bool(torch.isfinite(x).all()) for x in (a,b,h)):
        raise ValueError('Finite scores and relation required.')
    affinity=h.square().clone();affinity.fill_diagonal_(0.)
    mass=affinity.sum(-1,keepdim=True)
    probability=(a.softmax(-1)+b.softmax(-1))*.5
    support=(affinity/mass.clamp_min(torch.finfo(h.dtype).tiny))@probability
    # Without other-patch support return class-neutral evidence, not self confidence.
    return torch.where(mass>0,support,torch.zeros_like(support))
