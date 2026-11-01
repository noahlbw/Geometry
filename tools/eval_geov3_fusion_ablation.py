"""Geov3 path ablation: local, wide, equal fusion, and Anchored writeback."""
from __future__ import annotations
import argparse, json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from dinotool.frozen_deployment import FrozenGeometry
from dinotool.geov2_wide_optimized import (OptimizedWideReadout, WideReadoutConfig,
    factor_reconstruction, factor_reconstruction_batch, factor_correction)
from dinotool.bounded_physical_coupling import resize_geometry
from dinotool.compact_deployment import _iter_local_observations
from dinotool.efficient_deployment import restore_and_finalize, write_dense
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import metric, write

ARMS = ('LocalOnly','WideOnly','EqualLogitFusion','WeightedFusion','FullGeov3')

def build(root, record):
    from dinov3.eval.text import tokenizer
    tokenizer.get_tokenizer=lambda *_a,**_k: tokenizer.Tokenizer(vocab_path=str(Path(root['checkpoint_dir'])/'bpe_simple_vocab_16e6.txt.gz'))
    m=FrozenGeometry(record['config_path'],checkpoint_dir=root['checkpoint_dir'],dinov3_repo=root['dinov3_repo'],upstream_root=root['upstream_root'],text_cache=record['text_cache'],efficient=True,wide_branch='query_slot',query_slot_capacity=24,wide_alias_aggregation='logsumexp')
    m.profile=replace(m.profile,coupling=float(record['gain']))
    c=WideReadoutConfig(shared_backbone=True,shared_visual=False,direct=True,beta=float(record.get('beta',1.0)),capacity=24,fast_assignment=True,solve_dtype='float32',readout='anchored',batch_anchored=True,local_batch_size=8,compile_visual=True,compile_project=True,weight_dtype='bfloat16',wide_long_edge=int(record.get('wide_long_edge',0) or 0))
    return m,OptimizedWideReadout(m,c)

@torch.inference_mode()
def predict_endpoint(m,opt,image,endpoint,gain):
    resized=resize_geometry(image); h,w=resized.shape[-2:]; memory=opt.wide_memory(image)
    tiles=list(_iter_local_observations(resized,(h,w),m.compact_geometry,('original',),reconstruction=lambda r,v:factor_reconstruction(r,v,torch.float32),include_raw=True,batch_size=8,reconstruction_batch=lambda r,v:factor_reconstruction_batch(r,v,torch.float32)))
    wide_features=opt.tile_wide_features_batch(tiles,memory,(h,w)); p=m.profile; bank=m.banks[p.bank]; reader=m.reader
    with DeviceProbabilityAccumulator(bank.class_count,h,w,reader.blend.device) as acc:
      for tile,wf in zip(tiles,wide_features):
        wide=opt.scorer(wf[None] if wf.ndim==2 else wf)
        alias=tile['features']['original'].float()@reader.text[p.bank].T
        local=m.efficient_local_scorers[p.bank].uniform_reference(alias)
        if p.background_rule=='union_max' and reader.background is not None: local[:,reader.background]=alias[:,reader.members[p.bank]].amax(-1)
        if p.background_rule=='residual_max': local[:,reader.background]=(tile['features']['original'].float()@reader.residual_text.T).amax(-1)
        local=local/p.temperature; wide=wide.float()
        if endpoint=='LocalOnly': logits=local.double()
        elif endpoint=='WideOnly': logits=wide.double()
        elif endpoint=='EqualLogitFusion': logits=.5*(local.double()+wide.double())
        else: logits=local.double()+float(gain)*factor_correction(tile['operator'],wide.double()-local.double()).double()
        ah,aw=min(512,h-tile['top']),min(512,w-tile['left']); write_dense(acc,logits,bank.class_count,ah,aw,reader.blend,tile['left'],tile['top'])
      return restore_and_finalize(acc.probabilities/acc.normalizer[None],tuple(image.shape[-2:]),m.config['background_bias'],m.config['background_threshold'],reader.background)

@torch.inference_mode()
def predict_all(m, opt, image, gain):
    """Evaluate all four endpoints from one shared observation pass.

    The original draft recomputed the wide memory and all DonorBefore tiles
    once per arm, making a four-arm ablation unnecessarily expensive.  The
    endpoint logits differ only after the same local/wide observations, so we
    retain those tensors once and write four independent accumulators.
    """
    resized=resize_geometry(image); h,w=resized.shape[-2:]
    memory=opt.wide_memory(image)
    tiles=list(_iter_local_observations(
        resized,(h,w),m.compact_geometry,('original',),
        reconstruction=lambda r,v:factor_reconstruction(r,v,torch.float32),
        include_raw=True,batch_size=8,
        reconstruction_batch=lambda r,v:factor_reconstruction_batch(r,v,torch.float32)))
    wide_features=opt.tile_wide_features_batch(tiles,memory,(h,w))
    p=m.profile; bank=m.banks[p.bank]; reader=m.reader
    accum={x:DeviceProbabilityAccumulator(bank.class_count,h,w,reader.blend.device)
           for x in ARMS}
    try:
      for tile,wf in zip(tiles,wide_features):
        wide=opt.scorer(wf[None] if wf.ndim==2 else wf).float()
        alias=tile['features']['original'].float()@reader.text[p.bank].T
        local=m.efficient_local_scorers[p.bank].uniform_reference(alias)
        if p.background_rule=='union_max' and reader.background is not None:
            local[:,reader.background]=alias[:,reader.members[p.bank]].amax(-1)
        if p.background_rule=='residual_max':
            local[:,reader.background]=(tile['features']['original'].float()@reader.residual_text.T).amax(-1)
        local=local/p.temperature
        logits={'LocalOnly':local.double(), 'WideOnly':wide.double(),
                'EqualLogitFusion':.5*(local.double()+wide.double()),
                'WeightedFusion':.25*local.double()+.75*wide.double(),
                'FullGeov3':local.double()+float(gain)*factor_correction(tile['operator'],wide.double()-local.double()).double()}
        ah,aw=min(512,h-tile['top']),min(512,w-tile['left'])
        for name,value in logits.items():
            write_dense(accum[name],value,bank.class_count,ah,aw,reader.blend,tile['left'],tile['top'])
      return {name:restore_and_finalize(acc.probabilities/acc.normalizer[None],tuple(image.shape[-2:]),m.config['background_bias'],m.config['background_threshold'],reader.background) for name,acc in accum.items()}
    finally:
      for acc in accum.values(): acc.__exit__(None,None,None)

@torch.inference_mode()
def main(a):
    root=json.loads(Path(a.protocol).read_text()); r=dict(root['datasets'][a.dataset]); out=Path(a.output)/a.mode/a.dataset
    if out.exists(): raise RuntimeError('preserve existing output '+str(out))
    out.mkdir(parents=True); m,opt=build(root,r); ns=argparse.Namespace(**vars(a)); ns.data_root=r['loader']['data_root']; ns.sample_seed=20260923; ns.vdd_ontology='official'; ns.mode='full'; ns.num_shards=1; ns.shard_index=0
    entry=dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))}); validation,_,images,masks,names_map=load_samples(ns,entry)
    if len(validation)!=r['total_images']: raise RuntimeError('coverage mismatch')
    names=names_map[a.dataset]; samples=validation[:1] if a.mode=='smoke' else validation; mats={x:np.zeros((len(names),len(names)),np.int64) for x in ARMS}; keys=[]; start=time.perf_counter()
    for n,s in enumerate(samples,1):
      image=images(s); pred=predict_all(m,opt,image,r['gain']); target=masks(s,a.dataset,tuple(image.shape[-2:]));
      for x in ARMS: mats[x]+=confusion(pred[x],target,len(names))
      keys.append(s.key)
      if n==1 or n%25==0 or n==len(samples): write(out/'progress.json',dict(dataset=a.dataset,processed=n,total=len(samples),wall_seconds=time.perf_counter()-start)); print(json.dumps(dict(dataset=a.dataset,processed=n,total=len(samples))),flush=True)
    write(out/'results.json',dict(status='complete',implementation='geov3_fusion_ablation_20261101',dataset=a.dataset,arms=ARMS,model='Geov3 QuerySlot24 + DonorBefore + Anchored',g0=r['gain'],processed_images=len(keys),global_sample_count=len(validation),sample_keys=keys,class_names=names,metrics={x:dict(mean_iou_percent=metric(mats[x]),confusion_matrix=mats[x].tolist()) for x in ARMS},masks_used_only_for_scoring=True,wall_seconds=time.perf_counter()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--protocol',required=True); p.add_argument('--dataset',required=True); p.add_argument('--mode',choices=('smoke','full'),required=True); p.add_argument('--output',required=True); main(p.parse_args())
