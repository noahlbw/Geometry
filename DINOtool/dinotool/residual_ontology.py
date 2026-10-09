"""A residual label is a union of unscored concepts, not an ordinary synonym."""
import math
import torch


def complement_queries(label_text,scored_raw_ids):
    labels={}
    for line in label_text.splitlines():
        if not line.strip():
            continue
        number,name=line.split(':',1)
        number=int(number)
        if number in labels or not name.strip():
            raise ValueError('Invalid category-name metadata.')
        labels[number]=name.strip()
    if sorted(labels)!=list(range(1,460)):
        raise ValueError('Expected complete official459 category metadata.')
    scored=set(scored_raw_ids)-{0}
    if len(scored)!=59 or not scored.issubset(labels):
        raise ValueError('Expected official59 scored foreground IDs.')
    residual=[(i,labels[i]) for i in sorted(labels) if i not in scored]
    return ('background',)+tuple(name for _,name in residual),residual


def residual_score(scores,mode,temperature):
    if temperature<=0 or scores.shape[-1]==0:
        raise ValueError('Nonempty residual concepts and positive temperature required.')
    if mode=='max':
        return scores.amax(-1)
    if mode=='mean':
        return temperature*(torch.logsumexp(scores/temperature,-1)-math.log(scores.shape[-1]))
    raise ValueError('Predeclared max or count-normalized mean required.')
