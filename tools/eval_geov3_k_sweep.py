"""Geov3 QuerySlot capacity sweep with fixed banks and g0."""
from __future__ import annotations
import argparse, json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.geov2_wide_optimized import OptimizedWideReadout, WideReadoutConfig
from eval_geov3_path_ablation import predict_all
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import metric, write

@torch.inference_mode()
def main(a):
    root=json.loads(Path(a.protocol).read_text()); rec=dict(root['datasets'][a.dataset]); k=int(a.capacity)
    out=Path(a.output)/a.dataset/f'k{k}'
    if out.exists(): raise RuntimeError('preserve existing output '+str(out))
    out.mkdir(parents=True)
    from dinov3.eval.text import tokenizer
    tokenizer.get_tokenizer=lambda *_a,**_k: tokenizer.Tokenizer(vocab_path=str(Path(root['checkpoint_dir'])/'bpe_simple_vocab_16e6.txt.gz'))
    m=FrozenGeometry(rec['config_path'],checkpoint_dir=root['checkpoint_dir'],dinov3_repo=root['dinov3_repo'],upstream_root=root['upstream_root'],text_cache=rec['text_cache'],efficient=True,wide_branch='query_slot',query_slot_capacity=k,wide_alias_aggregation='logsumexp')
    m.profile=replace(m.profile,coupling=float(rec['gain']))
    c=WideReadoutConfig(shared_backbone=True,shared_visual=False,direct=True,beta=float(rec.get('beta',1.0)),capacity=k,fast_assignment=True,solve_dtype='float32',readout='anchored',batch_anchored=True,local_batch_size=8,compile_visual=True,compile_project=True,weight_dtype='bfloat16',wide_long_edge=int(rec.get('wide_long_edge',0) or 0))
    from dinotool.geov2_wide_optimized import OptimizedWideReadout
    opt=OptimizedWideReadout(m,c)
    ns=argparse.Namespace(**vars(a)); ns.data_root=rec['loader']['data_root']; ns.sample_seed=20260923; ns.vdd_ontology='official'; ns.mode='full'; ns.num_shards=1; ns.shard_index=0
    entry=dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))}); validation,_,images,masks,names_map=load_samples(ns,entry)
    names=names_map[a.dataset]; mats=np.zeros((len(names),len(names)),np.int64); start=time.perf_counter()
    for n,s in enumerate(validation,1):
        image=images(s); pred=predict_all(m,opt,image,rec['gain'])['FullGeov3']; target=masks(s,a.dataset,tuple(image.shape[-2:])); mats+=confusion(pred,target,len(names))
        if n==1 or n%25==0 or n==len(validation): write(out/'progress.json',dict(dataset=a.dataset,capacity=k,processed=n,total=len(validation))); print(json.dumps(dict(dataset=a.dataset,capacity=k,processed=n,total=len(validation))),flush=True)
    write(out/'results.json',dict(status='complete',dataset=a.dataset,capacity=k,model='Geov3 QuerySlot + DonorBefore + Anchored',processed_images=len(validation),global_sample_count=len(validation),g0=rec['gain'],mean_iou_percent=metric(mats),confusion_matrix=mats.tolist(),wall_seconds=time.perf_counter()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--dataset',required=True);p.add_argument('--capacity',type=int,required=True);p.add_argument('--output',required=True);main(p.parse_args())
