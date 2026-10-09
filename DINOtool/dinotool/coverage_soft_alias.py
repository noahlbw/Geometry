"""Rival-specific weights with a floor on the unscreened group's evidence."""
import torch
import torch.nn.functional as F
from .geometry_readout_trace import alias_class_scores

IMPLEMENTATION='geometry-taxonomy-coverage-soft-alias-v1-20261007'


def coverage_scores(alias,bank,temperature=.07):
    baseline=alias_class_scores(alias,bank.parent_indices,bank.class_count,temperature)
    text=F.normalize(bank.features.float(),dim=-1)
    prototypes=F.normalize(torch.stack([text[bank.parent_indices==c].mean(0)
                                       for c in range(bank.class_count)]),dim=-1)
    semantic=text@prototypes.T
    top=baseline.topk(min(2,bank.class_count),-1).indices
    output=[]
    for c in range(bank.class_count):
        members=bank.parent_indices==c
        count=int(members.sum())
        if bank.class_count==1:
            output.append(baseline[:,c])
            continue
        rival=torch.where(top[:,0]==c,top[:,1],top[:,0])
        margin=semantic[members,c][None]-semantic[members][:,rival].T
        learned=torch.sigmoid(margin/temperature).clamp_min(1e-6)
        learned/=learned.sum(-1,keepdim=True)
        # Every alias retains at least half its original uniform mass. Thus
        # S_new >= S_unscreened - temperature*log(2), irrespective of its weights.
        weights=.5/count+.5*learned
        output.append(temperature*torch.logsumexp(alias[:,members]/temperature+weights.log(),-1))
    return torch.stack(output,-1)
