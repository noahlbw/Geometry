"""Geov3 path diagnostics: accuracy, boundaries, object scale and cross-view transitions."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
import torch

from eval_geov3_path_ablation import ARMS, build, predict_all
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import write

def boundary_mask(x):
    x = np.asarray(x)
    b = np.zeros(x.shape, dtype=bool)
    b[:-1] |= x[:-1] != x[1:]
    b[1:] |= x[:-1] != x[1:]
    b[:, :-1] |= x[:, :-1] != x[:, 1:]
    b[:, 1:] |= x[:, :-1] != x[:, 1:]
    return b

def dilate(b, radius=2):
    k = np.ones((2*radius+1, 2*radius+1), np.uint8)
    return cv2.dilate(b.astype(np.uint8), k).astype(bool)

def boundary_iou(pred, target, radius=2, valid=None):
    pb, tb = boundary_mask(pred), boundary_mask(target)
    p, t = dilate(pb, radius), dilate(tb, radius)
    if valid is not None:
        p &= valid
        t &= valid
    return float((p & t).sum() / max(1, (p | t).sum()))

def scale_masks(target):
    small = np.zeros(target.shape, bool)
    large = np.zeros(target.shape, bool)
    for c in np.unique(target):
        if c < 0: continue
        mask = (target == c).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= 32*32: small[labels == i] = True
            if area >= 128*128: large[labels == i] = True
    return small, large

def restricted_iou(pred, target, region, nclass):
    if not region.any(): return float('nan')
    cm = confusion(pred[region], target[region], nclass)
    den = cm.sum(1) + cm.sum(0) - np.diag(cm)
    vals = np.divide(np.diag(cm), den, out=np.zeros(nclass,float), where=den>0)
    valid = den > 0
    return float(vals[valid].mean()) if valid.any() else float('nan')

@torch.inference_mode()
def main(a):
    root = json.loads(Path(a.protocol).read_text()); rec = dict(root['datasets'][a.dataset])
    out = Path(a.output) / a.mode / a.dataset
    if out.exists(): raise RuntimeError('preserve existing output '+str(out))
    out.mkdir(parents=True)
    model, opt = build(root, rec)
    ns = argparse.Namespace(**vars(a)); ns.data_root=rec['loader']['data_root']; ns.sample_seed=20260923; ns.vdd_ontology='official'; ns.mode='full'; ns.num_shards=1; ns.shard_index=0
    entry = dict(model.config, banks={'original': next(iter(model.vocabulary['banks'].values()))})
    validation, _, images, masks, names_map = load_samples(ns, entry)
    if len(validation) != rec['total_images']: raise RuntimeError('coverage mismatch')
    names = names_map[a.dataset]; samples = validation[:1] if a.mode == 'smoke' else validation
    mats = {x: np.zeros((len(names),len(names)), np.int64) for x in ARMS}
    totals = {x: dict(correct=0, pixels=0, boundary_iou=[], small_iou=[], large_iou=[]) for x in ARMS}
    transitions = np.zeros((2,2), np.int64)  # rows: local correct {0,1}; cols: wide correct {0,1}
    start=time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image=images(sample); preds=predict_all(model,opt,image,rec['gain']); target=masks(sample,a.dataset,tuple(image.shape[-2:])); target=np.asarray(target)
        valid=target>=0
        sm,lg=scale_masks(target)
        for arm in ARMS:
            pred=np.asarray(preds[arm]); m=valid
            mats[arm] += confusion(pred[m],target[m],len(names))
            totals[arm]['correct'] += int((pred[m]==target[m]).sum()); totals[arm]['pixels'] += int(m.sum())
            totals[arm]['boundary_iou'].append(boundary_iou(pred,target,valid=valid))
            totals[arm]['small_iou'].append(restricted_iou(pred,target,sm & valid,len(names)))
            totals[arm]['large_iou'].append(restricted_iou(pred,target,lg & valid,len(names)))
        lc=(preds['LocalOnly']==target) & valid; wc=(preds['WideOnly']==target) & valid
        for i in (0,1):
            for j in (0,1): transitions[i,j] += int(((lc==(i==1)) & (wc==(j==1))).sum())
        if number==1 or number%25==0 or number==len(samples):
            write(out/'progress.json',dict(dataset=a.dataset,processed=number,total=len(samples),wall_seconds=time.perf_counter()-start)); print(json.dumps(dict(dataset=a.dataset,processed=number,total=len(samples))),flush=True)
    metrics={}
    for arm in ARMS:
        t=totals[arm]
        metrics[arm]=dict(mean_iou_percent=float(np.mean([np.divide(np.diag(mats[arm]), mats[arm].sum(1)+mats[arm].sum(0)-np.diag(mats[arm]), out=np.zeros(len(names)), where=(mats[arm].sum(1)+mats[arm].sum(0)-np.diag(mats[arm]))>0)[i] for i in range(len(names))])), pixel_accuracy=float(t['correct']/max(1,t['pixels'])), boundary_iou=float(np.nanmean(t['boundary_iou'])), small_object_iou=float(np.nanmean(t['small_iou'])), large_region_iou=float(np.nanmean(t['large_iou'])), confusion_matrix=mats[arm].tolist())
    write(out/'results.json',dict(status='complete',implementation='geov3_path_diagnostics_20261101',dataset=a.dataset,arms=ARMS,model='Geov3 QuerySlot24 + DonorBefore + Anchored',g0=rec['gain'],processed_images=len(samples),global_sample_count=len(validation),class_names=names,metrics=metrics,cross_view_transitions=transitions.tolist(),scale_thresholds={'small_max_area':1024,'large_min_area':16384},boundary_tolerance=2,masks_used_only_for_scoring=True,wall_seconds=time.perf_counter()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--protocol',required=True); p.add_argument('--dataset',required=True); p.add_argument('--mode',choices=('smoke','full'),required=True); p.add_argument('--output',required=True); main(p.parse_args())
