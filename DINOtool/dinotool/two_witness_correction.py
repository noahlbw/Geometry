"""Frozen two-witness veto of unsupported foreground-to-foreground transport."""
import torch
from .development_readout import coupled
from .geometry_support_audit import relation_support

IMPLEMENTATION='geometry-two-witness-foreground-veto-v1-20261007'


def two_witness_coupled(local,wide,operator,gain,background=None):
    proposal=coupled(local,wide,operator,gain)
    support=relation_support(local,wide,operator)
    a=local.argmax(-1);b=proposal.argmax(-1)
    rows=torch.arange(len(a),device=a.device)
    evidence=local.double()+wide.double()
    opposed=(evidence[rows,b]<evidence[rows,a])&(support[rows,b]<support[rows,a])
    veto=(a!=b)&opposed
    if background is not None:
        if not 0<=background<local.shape[-1]:raise ValueError('Invalid background index.')
        veto=veto&(a!=background)&(b!=background)
    return torch.where(veto[:,None],local.double(),proposal)
