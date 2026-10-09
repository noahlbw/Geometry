"""Post-prediction, label-only audit; never part of inference or selection."""
import numpy as np

FIELDS=('support','changed','fixed','broken','wrong_to_wrong','base_correct','other_correct')


def transition_counts(target,base,other,classes):
    target,base,other=map(np.asarray,(target,base,other))
    if target.shape!=base.shape or target.shape!=other.shape:
        raise ValueError('Matched restored predictions and target required.')
    valid=(target>=0)&(target<classes)
    t,a,b=(x[valid].astype(np.int64) for x in (target,base,other))
    if ((a<0)|(a>=classes)|(b<0)|(b>=classes)).any():
        raise ValueError('Predictions outside class taxonomy.')
    ca,cb=a==t,b==t
    conditions=(np.ones_like(ca),a!=b,~ca&cb,ca&~cb,(a!=b)&~ca&~cb,ca,cb)
    result=np.stack([np.bincount(t[mask],minlength=classes) for mask in conditions],-1)
    if not np.array_equal(result[:,2]-result[:,3],result[:,6]-result[:,5]):
        raise RuntimeError('Correct-pixel transition identity failed.')
    if not np.array_equal(result[:,1],result[:,2:5].sum(1)):
        raise RuntimeError('Changed-pixel partition failed.')
    return result
