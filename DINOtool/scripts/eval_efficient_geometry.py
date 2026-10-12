"""Paired old-compact/new-efficient evaluation of the latest frozen task banks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.frozen_deployment import FrozenGeometry
from dinotool.model import checkpoint_manifest
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary


TOOL=Path(__file__).resolve().parents[1]
BASE=TOOL.parent


def read(path):return json.loads(Path(path).read_text())


def save(path, item):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Preserve completed output: '+str(path))
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(item,indent=2)+'\n');temp.rename(path)


def prepare_cache(args, entry, config, vocabulary, cache):
    """Export exact historical encoded subsets; no language re-encoding or search."""
    if cache.exists():return
    packed=torch.load(TOOL/'results'/entry['source_trial']/'text_cache'/f'{args.dataset}.pt',map_location='cpu',weights_only=True)
    expected=checkpoint_manifest(make_checkpoints(args))
    if packed['identity']['checkpoints']!=expected:raise RuntimeError('Historical checkpoint identity differs.')
    encoded={}
    for name, bank in vocabulary['banks'].items():
        actual=packed['identity']['banks']
        key=name if name in actual and actual[name]==bank else 'candidate_pool_'+('imagenet' if bank['template']=='ImageNet80' else 'segmentation')
        source=actual[key];row=packed['encoded'][key]
        wide=packed['stores'][row['template']][row['indices']] if 'indices' in row else row['wide']
        local=F.normalize(wide.float().mean(1),dim=-1) if 'indices' in row else row['local']
        lookup={(c,a):i for i,(c,a) in enumerate(( (c,a) for c,g in enumerate(source['classes']) for a in g['synonyms']))}
        indices=torch.tensor([lookup[c,a] for c,g in enumerate(bank['classes']) for a in g['synonyms']])
        parents=torch.tensor([c for c,g in enumerate(bank['classes']) for a in g['synonyms']])
        canonical=torch.zeros(len(indices),dtype=torch.bool)
        canonical[torch.tensor(np.cumsum([0]+[len(g['synonyms']) for g in bank['classes'][:-1]]))]=True
        encoded[name]=dict(local=local[indices].clone(),wide=wide[indices].clone(),parents=parents,canonical=canonical)
    result=dict(identity=dict(vocabulary=vocabulary,checkpoints=expected,vip_commit=config['upstream']['vip_commit']),encoded=encoded)
    if vocabulary.get('residual'):
        residual=torch.load(TOOL/'results/alias_finalization_20261009/residual_cache.pt',map_location='cpu',weights_only=True)
        if residual['identity']!=vocabulary['residual']:raise RuntimeError('Residual identity differs.')
        result['residual']=residual['wide']
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_suffix('.tmp');torch.save(result,tmp);tmp.rename(cache)


def build(args):
    import yaml
    root=Path(args.suite_root)
    entry=read(root/'protocol.json')['datasets'][args.dataset]
    config=yaml.safe_load((root/entry['config']).read_text())
    vocabulary=read((root/entry['config']).parent/config['vocabulary'])
    cache=root/'text_cache'/f'{args.dataset}.pt'
    prepare_cache(args,entry,config,vocabulary,cache)
    model=FrozenGeometry(root/entry['config'],checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo,upstream_root=args.upstream_root,text_cache=cache)
    entry=dict(entry,**config,banks={'original':next(iter(vocabulary['banks'].values()))})
    args.data_root=entry['data_root'];args.sample_seed=20260923;args.vdd_ontology='official'
    args.mode='full' if args.mode=='full' else args.mode
    validation,samples,images,masks,names=load_samples(args,entry)
    if len(validation)!=entry['total_images']:raise RuntimeError('Full image count differs.')
    return model,entry,(validation,samples,images,masks,names)


def variant(model, name):
    from dinotool.compact_deployment import CompactGeometry,CachedWideScorer
    from dinotool.finite_vip_observer import FiniteVIPObserver
    if name=='compact':
        model.efficient=False;model.compact_geometry=CompactGeometry(model.geometry)
        model.vip.__class__=FiniteVIPObserver;model.wide_scorer=CachedWideScorer()
    else:model.enable_efficient(grouped=name!='efficient_scalar')


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);mode=args.mode
    output=root/mode/(f'{args.dataset}_s{args.shard_index}.json' if mode=='full' else args.dataset+'.json')
    if output.exists():raise RuntimeError('Preserve existing output.')
    model,entry,data=build(args)
    sources={str(p.relative_to(TOOL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (TOOL/'dinotool/efficient_deployment.py',TOOL/'dinotool/frozen_deployment.py',
         TOOL/'dinotool/natural_static_search.py',TOOL/'dinotool/compact_deployment.py')}
    validation,samples,images,masks,names=data
    protocols=list(names)
    if mode in ('smoke','profile','cost'):
        keys=read(root/'cost_keys.json')[args.dataset];lookup={s.key:s for s in validation}
        selected=[lookup[k] for k in keys]
        references={};costs={};changes={}
        variants=('compact','efficient_scalar','efficient') if mode in ('smoke','profile') else ('compact','efficient')
        for name in variants:
            variant(model,name);torch.cuda.empty_cache();rows=[];changed=0
            for sample in selected:
                image=images(sample)
                for protocol in protocols:
                    key=sample.key+'::'+protocol
                    def forward():return model.predict(image,protocol=protocol if args.dataset=='loveda' else None)
                    actual=forward()
                    if name=='compact':references[key]=actual
                    else:changed+=int(np.count_nonzero(actual!=references[key]))
                    repeats=1 if mode=='smoke' else 7
                    for _ in range(2 if mode!='smoke' else 0):forward()
                    times=[];peaks=[];reserved=[]
                    for _ in range(repeats):
                        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                        actual=forward();torch.cuda.synchronize()
                        times.append((time.perf_counter()-start)*1000)
                        peaks.append(torch.cuda.max_memory_allocated()/1048576)
                        reserved.append(torch.cuda.max_memory_reserved()/1048576)
                    rows.append(dict(key=sample.key,protocol=protocol,shape=list(image.shape[-2:]),milliseconds=times,
                        peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved)))
            changes[name]=changed
            costs[name]=dict(median_ms=float(np.median([v for r in rows for v in r['milliseconds']])),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),images=rows)
        save(output,dict(status='complete',dataset=args.dataset,costs=costs,changed_pixels=changes,
            config=model.config,cache_identity=model.cache_identity,target_masks_loaded=False,sources=sources,
            timing='Whole RGB input through CPU IDs; excludes decode/init/offline search. Two warmups, seven repeats for cost/profile.'))
        print(json.dumps(dict(dataset=args.dataset,changed=changes,costs={k:{f:v[f] for f in ('median_ms','peak_allocated_mib')} for k,v in costs.items()})),flush=True)
        return
    matrices={name:{p:np.zeros((len(n),len(n)),np.int64) for p,n in names.items()} for name in ('compact','efficient')}
    changed={p:0 for p in names};per_image={name+'__'+p:[] for name in matrices for p in names}
    start=time.perf_counter();keys=[];changed_keys=[]
    for number,sample in enumerate(samples,1):
        image=images(sample);targets={p:masks(sample,p,tuple(image.shape[-2:])) for p in names}
        reference={}
        for name in matrices:
            # Static caches are reused on successive images, not repeatedly rebuilt.
            if not hasattr(model,'backends'):model.backends={}
            if name in model.backends:
                efficient,geometry,observer,scorer=model.backends[name]
                model.efficient=efficient;model.compact_geometry=geometry;model.vip.__class__=observer;model.wide_scorer=scorer
            else:
                variant(model,name)
                model.backends[name]=(model.efficient,model.compact_geometry,model.vip.__class__,model.wide_scorer)
            for p,n in names.items():
                actual=model.predict(image,protocol=p if args.dataset=='loveda' else None)
                if name=='compact':reference[p]=actual
                else:
                    count=int(np.count_nonzero(actual!=reference[p]));changed[p]+=count
                    if count:changed_keys.append(dict(key=sample.key,protocol=p,pixels=count))
                cm=confusion(actual,targets[p],len(n));matrices[name][p]+=cm;per_image[name+'__'+p].append(cm)
        keys.append(sample.key)
        if number%25==0:print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples),changed=changed)),flush=True)
    npz=output.with_suffix('.npz')
    np.savez_compressed(npz,keys=np.asarray(keys),**{k:np.stack(v) for k,v in per_image.items()})
    save(output,dict(status='complete',dataset=args.dataset,processed_images=len(samples),total_images=len(validation),
        num_shards=args.num_shards,shard_index=args.shard_index,sample_keys=keys,changed_pixels=changed,changed_keys=changed_keys,
        metrics={name:{p:summary(cm,names[p],0) for p,cm in value.items()} for name,value in matrices.items()},
        config=model.config,cache_identity=model.cache_identity,sources=sources,wall_seconds=time.perf_counter()-start,
        per_image_confusion=str(npz),per_image_sha256=hashlib.sha256(npz.read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--mode',choices=('smoke','profile','cost','full'),required=True)
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'))
    p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--device',default='cuda')
    main(p.parse_args())
