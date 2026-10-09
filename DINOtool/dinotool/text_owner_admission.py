"""Frozen text-nearest-parent admission; not an ontology truth verifier."""
import torch
import torch.nn.functional as F


def own_margin(features,parents,canonical,classes,background=None):
    if features.ndim!=2 or parents.shape!=(len(features),) or canonical.shape!=parents.shape:
        raise ValueError('Matched alias metadata required.')
    if not bool(torch.isfinite(features).all()):raise ValueError('Finite text embeddings required.')
    anchors=[]
    for c in range(classes):
        ids=torch.nonzero((parents==c)&canonical,as_tuple=False).flatten()
        if len(ids)!=1:raise ValueError('Exactly one anchor per category required.')
        anchors.append(int(ids[0]))
    text=F.normalize(features.double(),dim=-1)
    similarity=text@text[anchors].T
    rows=torch.arange(len(features),device=parents.device)
    own=similarity[rows,parents]
    rivals=similarity.clone();rivals[rows,parents]=-torch.inf
    if background is not None:rivals[:,background]=-torch.inf
    rival,index=rivals.max(-1)
    return own-rival,index


def admit(local,wide,parents,canonical,classes,background=None):
    ml,rl=own_margin(local,parents,canonical,classes,background)
    mw,rw=own_margin(wide,parents,canonical,classes,background)
    protect=canonical if background is None else canonical|(parents==background)
    keep=protect|((ml>0)&(mw>0))
    return keep,dict(local_margin=ml,wide_margin=mw,local_rival=rl,wide_rival=rw)
