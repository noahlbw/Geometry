"""Frozen VIP with the completed Kev experiment's selected language inputs.

The primary arm reuses exact wide text vectors. The second arm changes only
the prompt template to VIP's official template. VIP settings and image views
remain native; neither arm receives Geometry features or selected calibration.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.vip_official_adapter import VIPQueries, _load_upstream, upstream_settings
from eval_alias_finalization import load_samples
from eval_geometry_vip_reliability import summary
from eval_natural_text_adaptation import official_options, official_prediction
from eval_rival_fine_full import save
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT, Confusion, digest

IMPLEMENTATION='vip-kev-selected-same-language-v1-20261010'
METHODS=('VIP_SameText','VIP_SameWordsOfficialTemplate')


def read(path):
    return json.loads(Path(path).read_text())


def frozen_check(vip, state):
    frozen=all(not p.requires_grad for p in vip.backbone.model.parameters())
    unchanged=all(torch.equal(state[k],v.cpu()) for k,v in vip.backbone.model.visual_model.head.state_dict().items())
    if not frozen or not unchanged:raise RuntimeError('Frozen VIP weights changed.')
    return dict(weights_frozen=frozen,head_weights_unchanged=unchanged)


@torch.inference_mode()
def models(args, entry):
    source=Path(args.source_root);upstream=Path(args.upstream_root)
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PINNED_COMMIT:
        raise RuntimeError('Pinned VIP source changed.')
    selection=read(source/'search'/args.dataset/'selection.json')
    if selection['profile']!=entry['selected_profile'] or selection['background_bias']!=entry['selected_background_bias'] or selection['background_threshold']!=entry['selected_background_threshold']:
        raise RuntimeError('Previously frozen development choice changed.')
    config=make_checkpoints(args);manifest=checkpoint_manifest(config)
    prior=read(source/'full'/args.dataset/'merged.json')
    if not prior['coverage_verified'] or prior['signature']['checkpoints']!=manifest:
        raise RuntimeError('Completed source/checkpoint identity changed.')
    vip=FiniteVIPObserver(DINOTextSegmenter(config,device=args.device),upstream)
    if entry['family']=='natural':settings,official_template=official_options(upstream,args.dataset)
    else:settings,official_template=upstream_settings(args.dataset),'openai_imagenet_template'
    cache=Path(args.suite_root)/'text_cache'/(args.dataset+'.pt')
    identity=dict(classes=entry['classes'],source_bank=entry['selected_bank'],source_template=entry['wide_template'],
                  residual_identity=entry.get('residual_identity'),checkpoints=manifest,upstream_commit=PINNED_COMMIT,
                  official_template=official_template)
    if cache.exists():
        stored=torch.load(cache,map_location='cpu',weights_only=True)
        if stored['identity']!=identity:raise RuntimeError('Frozen same-vocabulary cache differs.')
    else:
        packed=torch.load(source/'text_cache'/(args.dataset+'.pt'),map_location='cpu',weights_only=True)
        name=entry['selected_bank'];record=selection['vocabulary'][name]
        if packed['identity']['banks'][name]!=record:raise RuntimeError('Source encoded word bank differs.')
        item=packed['encoded'][name]
        wide=packed['stores'][item['template']][item['indices']] if 'indices' in item else item['wide']
        parents=item['parents'];groups=[list(c['synonyms']) for c in record['classes']]
        if entry.get('residual_identity'):
            residual=torch.load(source.parent/'alias_finalization_20261009'/'residual_cache.pt',map_location='cpu',weights_only=True)
            if residual['identity']!=entry['residual_identity']:raise RuntimeError('PC60 residual ontology differs.')
            rows=[];groups[entry['background_index']]=entry['residual_identity']['names']
            for i in range(len(groups)):
                rows.append(residual['wide'] if i==entry['background_index'] else wide[parents==i])
            wide=torch.cat(rows)
        classes=entry['classes'];names=tuple(c['name'] for c in classes)
        if groups!=[c['synonyms'] for c in classes]:raise RuntimeError('Frozen selected aliases differ.')
        aliases=tuple(a for g in groups for a in g)
        parents=torch.tensor([i for i,g in enumerate(groups) for _ in g],dtype=torch.long)
        prompt=_load_upstream(upstream/'prompts/imagenet_template.py','vip_kev_same_vocabulary_prompts')
        vip.templates=prompt.get_text_template(official_template)
        if entry['wide_template']==official_template:
            official=wide
        else:
            query=vip.encode_queries(names,tuple(tuple(g) for g in groups))
            official=query.features.cpu()
        stored=dict(identity=identity,features={METHODS[0]:wide,METHODS[1]:official},parents=parents,
                    aliases=aliases,names=names,templates_identical=entry['wide_template']==official_template)
        cache.parent.mkdir(parents=True,exist_ok=True)
        torch.save(stored,cache.with_suffix('.tmp'));cache.with_suffix('.tmp').rename(cache)
    queries={m:VIPQueries(f.to(args.device),stored['parents'].to(args.device),tuple(stored['names']),tuple(stored['aliases']))
             for m,f in stored['features'].items() if args.phase!='cost' or m==METHODS[0]}
    protocol_queries={args.dataset:queries}
    if args.dataset=='loveda':
        protocol_queries={'D':queries,'P':{}}
        for m,q in queries.items():
            keep=q.parents!=0
            protocol_queries['P'][m]=VIPQueries(q.features[keep],q.parents[keep]-1,q.class_names[1:],
                                               tuple(a for a,p in zip(q.aliases,q.parents.tolist()) if p!=0))
    return vip,protocol_queries,settings,manifest,stored


def one_prediction(image,vip,query,settings,family):
    return (official_prediction(image,vip,query,settings)[0] if family=='natural'
            else vip.predict(image,query,settings)[0])


@torch.inference_mode()
def predict_all(image,vip,queries,settings,family):
    """Replay exact crop features across text arms and LoveDA protocols."""
    original=vip.crop_patch_features;features=[];position=0;capturing=True
    def observed(crop):
        nonlocal position
        if capturing:features.append(original(crop));return features[-1]
        if position>=len(features):raise RuntimeError('Text arms changed visual crop count.')
        result=features[position];position+=1;return result
    vip.crop_patch_features=observed;predictions={}
    try:
        for p,group in queries.items():
            local_settings=replace(settings,background=False,prob_thd=0.) if p=='P' else settings
            predictions[p]={}
            for m,query in group.items():
                position=0
                predictions[p][m]=one_prediction(image,vip,query,local_settings,family)
                if not capturing and position!=len(features):raise RuntimeError('Text arm visual replay incomplete.')
                capturing=False
    finally:vip.crop_patch_features=original
    return predictions,len(features)


@torch.inference_mode()
def benchmark(args,entry,validation,loader,vip,queries,settings,stored,initialization):
    primary='D' if args.dataset=='loveda' else args.dataset
    query=queries[primary][METHODS[0]];rows=[]
    for i in np.unique(np.linspace(0,len(validation)-1,7,dtype=int)):
        sample=validation[int(i)];image=loader(sample)
        forward=lambda:one_prediction(image,vip,query,settings,entry['family'])
        forward();forward();torch.cuda.synchronize();times=[];peaks=[]
        for _ in range(7):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            pred=forward();torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
            peaks.append(torch.cuda.max_memory_allocated()/1048576)
            if pred.shape!=tuple(image.shape[-2:]):raise RuntimeError('Original size not restored.')
        rows.append(dict(key=sample.key,shape=list(image.shape[-2:]),milliseconds=times,peak_allocated_mib=max(peaks)))
    flat=[v for r in rows for v in r['milliseconds']]
    return dict(status='complete',dataset=args.dataset,method=METHODS[0],images=rows,
                median_ms=float(np.median(flat)),p95_ms=float(np.quantile(flat,.95)),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),settings=asdict(settings),
                text_identity=stored['identity'],initialization_and_text_seconds=initialization,
                standalone_process=True,target_masks_loaded=False,
                boundary='Seven full inputs x seven warmed synchronized repeats. Native VIP resizing/encoding/readout/restoration/CPU output; decode/init/offline selection excluded.')


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);entry=read(root/'protocol.json')['datasets'][args.dataset]
    output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing output preserved: '+str(output))
    args.mode='smoke' if args.phase=='smoke' else 'full'
    validation,samples,loader,load_mask,names=load_samples(args,entry)
    previous=read(Path(args.source_root)/'full'/args.dataset/'merged.json')
    if [s.key for s in validation]!=previous['signature']['sample_keys'] or len(validation)!=entry['total_images']:
        raise RuntimeError('Full source image sequence differs.')
    start=time.perf_counter();vip,queries,settings,manifest,stored=models(args,entry)
    initialization=time.perf_counter()-start
    state={k:v.cpu().clone() for k,v in vip.backbone.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    if args.phase=='cost':
        result=benchmark(args,entry,validation,loader,vip,queries,settings,stored,initialization)
        result.update(**frozen_check(vip,state));save(output/'results.json',result);return
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=list(METHODS),classes=names,
                   selected_profile=entry['selected_profile'],text_identity=stored['identity'],
                   template_counts={m:int(f.shape[1]) for m,f in stored['features'].items()},
                   settings=asdict(settings),checkpoints=manifest,upstream_commit=PINNED_COMMIT,
                   view_policy='short336_cap2048' if entry['family']=='natural' else 'long448',
                   global_sample_count=len(validation),global_sample_keys_sha256=digest([s.key for s in validation]),
                   sample_keys=[s.key for s in samples],num_shards=args.num_shards,shard_index=args.shard_index,
                   residual_rule='PC60 uses the same explicit residual concepts as background aliases with standard VIP aggregation; no Geometry residual gate.',
                   numeric_repair='Self-Value only on completely masked proxy rows; same repaired VIP as prior references.')
    if args.phase=='smoke':
        predictions,crops=predict_all(loader(samples[0]),vip,queries,settings,entry['family'])
        if any(not np.isfinite(pred).all() for group in predictions.values() for pred in group.values()):
            raise RuntimeError('Nonfinite smoke prediction.')
        result=dict(status='complete',processed_images=1,total_images=1,target_masks_loaded=False,
                    visual_encodings=crops,signature=signature,**frozen_check(vip,state))
        save(output/'results.json',result);return
    matrices={p:{m:Confusion(tuple(n),len(n)+int(p!='P' and not settings.background and settings.prob_thd>0))
                 for m in METHODS} for p,n in names.items()}
    arrays={p+'__'+m:[] for p in names for m in METHODS};started=time.perf_counter();visual=0
    for number,sample in enumerate(samples,1):
        image=loader(sample);predictions,crops=predict_all(image,vip,queries,settings,entry['family']);visual+=crops
        for p,group in predictions.items():
            target=load_mask(sample,p,tuple(image.shape[-2:]))
            for m,prediction in group.items():
                cm=matrices[p][m];before=cm.matrix.copy()
                prediction=np.where(prediction==255,len(names[p]),prediction).astype(np.int64)
                cm.update(prediction,target);arrays[p+'__'+m].append(cm.matrix-before)
        if number==1 or number%10==0 or number==len(samples):
            result=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                        metrics={p:{m:summary(cm.matrix,tuple(names[p]),cm.ignored) for m,cm in group.items()} for p,group in matrices.items()},
                        wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                        target_masks_used_only_after_prediction=True,target_label_tuning=False,visual_encodings=visual,
                        empty_proxy_rows=vip.empty_rows,observed_proxy_rows=vip.observed_rows)
            save(output/'results.json',result)
            print(json.dumps(dict(dataset=args.dataset,processed_images=number,total_images=len(samples))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
                        **{k:np.stack(v) for k,v in arrays.items()})
    result.update(status='complete',**frozen_check(vip,state));save(output/'results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('suite-root','source-root','dataset','data-root','dinov3-repo','checkpoint-dir','upstream-root','output-dir'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--phase',choices=('smoke','full','cost'),required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--sample-seed',type=int,default=20260923)
    parser.add_argument('--vdd-ontology',default='official');main(parser.parse_args())
