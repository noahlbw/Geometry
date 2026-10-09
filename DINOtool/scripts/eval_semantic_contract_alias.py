"""Paired whole-image contract-off/on full evaluation and isolated timing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool import semantic_contract_alias as trial
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.fixed20_task_readout import TaskPolicy
from dinotool.geometry_execution import GeometryExecution
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
from dinotool.prompts import ClassSpec
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_development_readout import observations, probabilities, wide_scores
from eval_gear_ov import digest, protocol as rs_inputs
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


@torch.inference_mode()
def setup(args, protocol):
    root=Path(args.suite_root);entry=protocol['datasets'][args.dataset]
    for name,expected in protocol['source_files'].items():
        if hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source changed: '+name)
    path=root/'vocabularies'/(args.dataset+'.json')
    meta_path=root/'contracts'/(args.dataset+'.json')
    if (hashlib.sha256(path.read_bytes()).hexdigest()!=entry['vocabulary_sha256']
            or hashlib.sha256(meta_path.read_bytes()).hexdigest()!=entry['contract_sha256']):
        raise RuntimeError('Frozen vocabulary/contract differs.')
    classes=json.loads(path.read_text())['classes'];specs=[ClassSpec(v['name'],tuple(v['synonyms'])) for v in classes]
    if entry['family']=='natural':
        samples=discover_samples(args.dataset,args.data_root);loader=load_rgb
        masks=lambda sample,shape:load_target(sample,args.dataset,shape)
    else:
        samples,_,loader,mask_loader=rs_inputs(args,specs)
        masks=lambda sample,shape:mask_loader(sample,args.dataset,shape)
    cached=torch.load(root/'text_cache'/(args.dataset+'.pt'),map_location=args.device,weights_only=True)
    if (tuple(cached['aliases'])!=tuple(a for s in specs for a in s.synonyms)
            or tuple(cached['classes'])!=tuple(s.name for s in specs)
            or cached['vocabulary_sha256']!=entry['vocabulary_sha256']
            or cached['template']!=entry['policy']['template']):
        raise RuntimeError('Selected existing text features differ.')
    commit=subprocess.check_output(['git','-C',args.upstream_root,'rev-parse','HEAD'],text=True).strip()
    if commit!=PINNED_COMMIT:raise RuntimeError('Pinned visual source changed.')
    checkpoints=make_checkpoints(args)
    if cached['checkpoints']!=checkpoint_manifest(checkpoints):raise RuntimeError('Checkpoint identities changed.')
    g=GeometryExecution(TCPRSegmenter(DINOTextSegmenter(checkpoints,device=args.device),
                                     TCPRConfig(maximum_aliases_per_class=20,geometry_depth=2)))
    v=FiniteVIPObserver(DINOTextSegmenter(checkpoints,device=args.device),Path(args.upstream_root))
    parents=torch.arange(len(specs),device=g.device).repeat_interleave(20)
    canonical=torch.zeros(len(parents),device=g.device,dtype=torch.bool);canonical[::20]=True
    bank=TCPRTextBank(cached['local'],parents,canonical,tuple(cached['classes']),tuple(cached['aliases']))
    bank.validate()
    query=VIPQueries(cached['wide'],parents,bank.class_names,bank.alias_names)
    plan=trial.prepare_plan(bank,json.loads(meta_path.read_text()))
    policy=TaskPolicy(**entry['policy'])
    return samples,loader,masks,g,v,bank,query,plan,policy,cached['checkpoints']


@torch.inference_mode()
def main(args):
    output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing worker output; preserve it.')
    device=torch.device(args.device);require_available_gpu(device);lease=torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**kw):save(output/'worker_status.json',dict(status='running',phase=phase,**kw))
    state('loading');root=Path(args.suite_root)
    protocol=json.loads((root/'protocol.json').read_text());entry=protocol['datasets'][args.dataset]
    samples,loader,masks,g,v,bank,query,plan,policy,checkpoints=setup(args,protocol)
    frozen=frozen_state(g,v)
    def call(name,image):return trial.predict(image,g,v,bank,query,plan,policy,(name,))
    if args.mode=='benchmark':
        smoke=[];timing=[]
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            image=loader(samples[index]);state('smoke',sample_key=samples[index].key)
            joint,diag=trial.predict(image,g,v,bank,query,plan,policy,verify=True)
            source=observations(image,g,v,(policy.strength,),wide_policy=policy.wide_policy)
            broad=wide_scores(source,query,policy.tau,policy.tem)
            expected=probabilities(source,policy.profile(),bank,broad,{}).argmax(0).cpu().numpy()
            if not np.array_equal(joint[trial.BASE],expected):raise RuntimeError('Inherited off prediction differs.')
            del source,broad
            for name in trial.METHODS:
                if not np.array_equal(call(name,image)[0][name],joint[name]):raise RuntimeError('Singleton/joint differs: '+name)
            smoke.append(dict(sample_key=samples[index].key,inherited_base_exact=True,
                singleton_joint_exact=True,diagnostics=diag))
            rows={name:dict(seconds=[],peaks=[]) for name in trial.METHODS}
            state('timing',sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order=trial.METHODS[repeat%2:]+trial.METHODS[:repeat%2]
                for name in order:
                    require_available_gpu(device);actual,val=measure(call,(name,image),device,1);require_available_gpu(device)
                    if not np.array_equal(actual[0][name],joint[name]):raise RuntimeError('Warm prediction changed.')
                    rows[name]['seconds']+=val['seconds'];rows[name]['peaks'].append(val['peak_allocated_mib'])
            for val in rows.values():
                val.update(median_seconds=statistics.median(val['seconds']),p90_seconds=float(np.quantile(val['seconds'],.9)),
                           peak_allocated_mib=max(val.pop('peaks')))
            timing.append(dict(sample_key=samples[index].key,timings=rows))
        save(output/'smoke.json',dict(status='complete',images=smoke,target_masks_loaded=False,
                                     **check_frozen(frozen,g,v)))
        save(output/'timing.json',dict(status='complete',images=timing,target_masks_loaded=False,
            whole_image_scope=True,exclusive_gpu_verified=True,
            includes='resize, both visual branches, alias use, writeback, restoration, CPU argmax',
            excludes='decode and model/text setup',new_visual_forwards=0))
        save(output/'worker_status.json',dict(status='complete',phase='complete'));return
    for d in protocol['datasets']:
        smoke=json.loads((root/'benchmark'/d/'smoke.json').read_text())
        if smoke['status']!='complete' or smoke['target_masks_loaded']:raise RuntimeError('Mask-free benchmark required.')
    keys=[s.key for s in samples];selected=samples[args.shard_index::args.num_shards]
    if len(keys)!=entry['total_images'] or len(keys)!=len(set(keys)) or not selected:
        raise RuntimeError('Full inventory differs.')
    nc=bank.class_count
    signature=dict(implementation=trial.IMPLEMENTATION,dataset=args.dataset,methods=trial.METHODS,
        classes={args.dataset:bank.class_names},checkpoints=checkpoints,
        vocabulary=dict(aliases=bank.alias_names,counts=[20]*nc,sha256=entry['vocabulary_sha256']),
        gear=dict(profile=asdict(policy),source_files=protocol['source_files']),
        competitive=dict(contract_sha256=entry['contract_sha256'],sparse_edges=len(plan.edges),
                         shared_weight=.5,canonical_protected=True,unknown_abstention=True,
                         no_survivor_renormalization=True,class_square_alias_tensor=False),
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected],sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args))
    cm={name:np.zeros((nc,nc),np.int64) for name in trial.METHODS}
    per={args.dataset+'__'+name:[] for name in trial.METHODS};transitions=[];ignored=0;totals={}
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image=loader(sample);pred,diag=trial.predict(image,g,v,bank,query,plan,policy)
        target=masks(sample,tuple(image.shape[-2:]));valid=(target>=0)&(target<nc)
        ignored+=int((~valid).sum())
        for name in trial.METHODS:
            if pred[name].shape!=target.shape or np.any((pred[name][valid]<0)|(pred[name][valid]>=nc)):
                raise RuntimeError('Prediction shape/classes differ.')
            current=np.bincount(target[valid].astype(np.int64)*nc+pred[name][valid],minlength=nc*nc).reshape(nc,nc)
            cm[name]+=current;per[args.dataset+'__'+name].append(current)
        b=(pred[trial.BASE][valid]==target[valid]);p=(pred[trial.PRIMARY][valid]==target[valid])
        transitions.append(np.bincount(2*b.astype(np.int64)+p,minlength=4))
        if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards']:
            raise RuntimeError('Visual budget exceeded.')
        for field,val in diag.items():totals[field]=totals.get(field,0.)+val
        if number==1 or number%20==0 or number==len(selected):
            row=dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={args.dataset:{name:summary(mat,bank.class_names,ignored) for name,mat in cm.items()}},
                diagnostics={args.dataset:{k:val/number for k,val in totals.items()}},
                wall_seconds=time.perf_counter()-start,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_masks_used_only_after_prediction=True,target_label_tuning=False)
            save(output/'results.json',row);state('full',processed=number,total=len(selected))
            print(json.dumps(dict(dataset=args.dataset,shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
                        correctness_transitions=np.stack(transitions),**{k:np.stack(v) for k,v in per.items()})
    row.update(status='complete',**check_frozen(frozen,g,v));save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('benchmark','full'),required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official');p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
