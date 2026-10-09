"""An opt-in, label-free consistency modulation of the frozen gain.

Squared off-diagonal reconstruction influence is used as a nonnegative
leave-self-out neighborhood. This is not semantic correctness or independent
evidence: correlated wrong predictions can be consistent.
"""
import torch

IMPLEMENTATION='geometry-leave-self-out-consistency-gain-v1-20261007'


def consistency_coupled(local,broad,operator,gain):
    if local.ndim!=2 or local.shape!=broad.shape or operator.shape!=(len(local),len(local)):
        raise ValueError('Matching patch/class scores and square operator required.')
    if gain not in (.5,1.,2.):
        raise ValueError('Reuse the frozen gain ceiling/menu.')
    a,b=local.double(),broad.double()
    h=operator.double()
    affinity=h.square().clone()
    affinity.fill_diagonal_(0.)
    mass=affinity.sum(-1,keepdim=True)
    neighbors=affinity/mass.clamp_min(torch.finfo(affinity.dtype).tiny)
    pa,pb=a.softmax(-1),b.softmax(-1)
    ea=(pa-neighbors@pa).square().sum(-1,keepdim=True)
    eb=(pb-neighbors@pb).square().sum(-1,keepdim=True)
    denominator=ea+eb
    # Equal consistency retains g; no neighbor/evidence defaults exactly to g.
    scale=torch.where((mass>0)&(denominator>0),
        2*ea/denominator.clamp_min(torch.finfo(ea.dtype).tiny),torch.ones_like(ea))
    return a+gain*scale*(h@(b-a))
