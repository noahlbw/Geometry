"""Labelled diagnostic statistics only; no inference decision or fitted gate."""
import numpy as np

SIGNALS=('negative_base_confidence','cross_view_log_evidence','spatial_corroboration')


def exact_auc(score,beneficial):
    score=np.asarray(score,dtype=np.float64);positive=np.asarray(beneficial,dtype=bool)
    if score.ndim!=1 or score.shape!=positive.shape or not np.isfinite(score).all():
        raise ValueError('Finite matched one-dimensional scores required.')
    npos=int(positive.sum());nneg=len(score)-npos
    if not npos or not nneg:return None
    order=np.argsort(score,kind='stable');sorted_score=score[order];p=positive[order]
    starts=np.r_[0,np.flatnonzero(sorted_score[1:]!=sorted_score[:-1])+1]
    sizes=np.diff(np.r_[starts,len(score)])
    positives=np.add.reduceat(p.astype(np.int64),starts)
    negatives=sizes-positives;before=np.cumsum(negatives)-negatives
    return float(np.sum(positives*(before+.5*negatives))/(npos*nneg))


def neighbourhood(probability):
    """3x3 mean on the fixed diagnostic grid, not the Geometry operator."""
    padded=np.pad(probability,((0,0),(1,1),(1,1)),mode='edge')
    h,w=probability.shape[1:]
    return sum(padded[:,dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3))/9


def observations(local,wide,candidate,target,baseline,geometry_support=None):
    """Fixed-vs-broken comparisons; wrong-to-wrong changes are excluded."""
    local,wide,candidate=(np.asarray(p,dtype=np.float64) for p in (local,wide,candidate))
    target=np.asarray(target)
    if local.shape!=wide.shape or local.shape!=candidate.shape or local.ndim!=3 or target.shape!=local.shape[1:]:
        raise ValueError('Matched class-by-grid probabilities required.')
    if baseline not in ('LocalEndpoint','WideEndpoint'):raise ValueError('Declared endpoint baseline required.')
    if not all(np.isfinite(p).all() for p in (local,wide,candidate)):
        raise ValueError('Nonfinite probabilities.')
    base=local if baseline=='LocalEndpoint' else wide
    a=base.argmax(0);b=candidate.argmax(0)
    valid=(target>=0)&(target<len(base));ca=a==target;cb=b==target
    comparable=valid&(ca!=cb)
    y,x=np.nonzero(comparable);aa,bb=a[y,x],b[y,x]
    logs=sum(np.log(np.clip(p,1e-12,1)) for p in (local,wide))
    spatial=neighbourhood(local)+neighbourhood(wide)
    result=dict(true_class=target[y,x].astype(np.int64),beneficial=cb[y,x],
        negative_base_confidence=-base[aa,y,x],
        cross_view_log_evidence=logs[bb,y,x]-logs[aa,y,x],
        spatial_corroboration=spatial[bb,y,x]-spatial[aa,y,x],
        valid_pixels=int(valid.sum()),changed_pixels=int((valid&(a!=b)).sum()))
    if geometry_support is not None:
        geometry_support=np.asarray(geometry_support)
        if geometry_support.shape!=local.shape or not np.isfinite(geometry_support).all():
            raise ValueError('Matched finite Geometry support required.')
        result['geometry_corroboration']=geometry_support[bb,y,x]-geometry_support[aa,y,x]
    return result


def summarize(records,class_names,background,signal_names=SIGNALS):
    result={}
    for base,rows in records.items():
        merged={key:np.concatenate([r[key] for r in rows]) for key in ('true_class','beneficial',*signal_names)}
        y=merged['beneficial'];classes=merged['true_class']
        signals={}
        for name in signal_names:
            per_class={c:exact_auc(merged[name][classes==i],y[classes==i]) for i,c in enumerate(class_names)}
            available=[v for v in per_class.values() if v is not None]
            foreground=[per_class[c] for i,c in enumerate(class_names) if i!=background and per_class[c] is not None]
            signals[name]=dict(micro_auc=exact_auc(merged[name],y),
                macro_class_auc=None if not available else float(np.mean(available)),
                macro_foreground_auc=None if not foreground else float(np.mean(foreground)),
                classes_with_both_outcomes=len(available),per_class_auc=per_class)
        result[base]=dict(valid_grid_pixels=sum(r['valid_pixels'] for r in rows),
            changed_grid_pixels=sum(r['changed_pixels'] for r in rows),fixed=int(y.sum()),broken=int((~y).sum()),signals=signals)
    return result
