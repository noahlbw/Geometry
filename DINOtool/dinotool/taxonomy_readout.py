"""Frozen task calibration from canonical-query rank and image bank separation."""
import math
import torch
import torch.nn.functional as F
from .development_readout import Profile

IMPLEMENTATION='geometry-taxonomy-rank-readout-v1-20261007'


def canonical_features(bank):
    text=F.normalize(bank.features.float(),dim=-1)
    return torch.stack([text[(bank.parent_indices==c)&bank.canonical_mask][0] for c in range(bank.class_count)])


def rank_fraction(bank):
    text=canonical_features(bank)
    centred=text-text.mean(0)
    gram=centred@centred.T
    denominator=gram.square().sum()
    if float(denominator)<1e-12:
        return 0.
    effective=gram.trace().square()/denominator
    return float((effective/len(text)).clamp(0,1))


def natural_profile(bank):
    r=rank_fraction(bank)
    odds=max(r,1e-6)/max(1-r,1e-6)
    # Reuse the previously declared, finite gain menu without a new fitted scalar.
    gain=min((.5,1.,2.),key=lambda g:abs(math.log(g)-math.log(odds)))
    strength='original' if gain==.5 else 2.
    return Profile(bank='semantic_segmentation',strength=strength,coupling=gain),dict(
        canonical_rank_fraction=r,rank_odds=odds,selected_gain=gain,selected_strength=strength)


def bank_separation(source,bank,background):
    anchors=canonical_features(bank)
    scores=[]
    for tile in source['local']:
        cosine=tile['features']['original'].float()@anchors.T
        if background is not None:
            cosine=cosine[:,torch.arange(bank.class_count,device=cosine.device)!=background]
        top=cosine.topk(min(2,cosine.shape[1]),-1).values
        margin=top[:,0]-top[:,1] if top.shape[1]>1 else top[:,0]
        h,w=source['size']
        y,x=torch.meshgrid(torch.arange(32,device=cosine.device)*16+8+tile['top'],
                           torch.arange(32,device=cosine.device)*16+8+tile['left'],indexing='ij')
        valid=((y<h)&(x<w)).flatten()
        scores.append(margin[valid])
    return float(torch.cat(scores).mean())


def rs_profile(source,banks,background):
    # Camera/view descriptions enter through two frozen text banks. Their readout
    # routes are development-derived task priors, not claimed universal optima.
    utilities={k:bank_separation(source,banks[k],background) for k in ('original_imagenet','focused20')}
    selected=max(utilities,key=utilities.get)
    p=Profile(bank=selected,strength=1. if selected=='original_imagenet' else 3.,coupling=.5)
    return p,dict(bank_separation=utilities,selected_bank=selected,selected_strength=p.strength,selected_gain=.5)
