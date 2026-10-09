"""Frozen two-observer residual weights, mask-free prepass and paired scoring."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import eval_family_lme_alias as inherited
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool import family_mass_alias as predecessor
from dinotool import region_residual_alias as trial
from dinotool.fixed20_task_readout import POLICIES
from dinotool.vip_official_adapter import upstream_settings
from eval_canonical_rival_trial import (official_speed_queries, official_natural_single, OFFICIAL_SPEED)
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest
from eval_rival_fine_full import check_frozen, frozen_state, save


FULL_METHODS=(*trial.METHODS,trial.VIP)
REPLAY={trial.CURRENT:predecessor.CURRENT_BASE,trial.RAW_BASE:predecessor.COMPLETE_BASE,
        trial.BASE:predecessor.UNIFORM,trial.FAMILY:predecessor.PRIMARY,trial.VIP:predecessor.VIP}


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(args,entry,samples,selected,banks,words,checkpoints,identity,protocol):
    keys=[s.key for s in samples]; shard_keys=[s.key for s in selected]
    return dict(implementation=trial.IMPLEMENTATION,dataset=args.dataset,methods=FULL_METHODS,
        classes={args.dataset:banks['Current20'].class_names},checkpoints=checkpoints,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),task_text_identity=identity,
                  candidate_sha256=protocol['candidate_sha256'][args.dataset]),competitive=None,
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),sample_keys=shard_keys,
        sample_keys_sha256=digest(shard_keys),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args))


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists(): raise RuntimeError('Existing output; preserve it.')
    device=torch.device(args.device); require_available_gpu(device); lease=torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**kw): save(output/'worker_status.json',dict(status='running',phase=phase,**kw))
    state('loading')
    setup=inherited.curated_setup if args.dataset=='ade150' else inherited.original_setup
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,checkpoints,reference,_=setup(args)
    if reference is not None: raise RuntimeError('Unexpected additional model.')
    for name,expected in protocol['method_sources'].items():
        if file_sha(Path(__file__).parents[1]/name)!=expected: raise RuntimeError('Frozen source differs: '+name)
    inherited.prepare_plan=trial.prepare_plan
    banks,queries,plans,identity,text_cost=inherited.input_banks(root,args.dataset,g,v,old_b,old_q,checkpoints,protocol)
    frozen=frozen_state(g,v); policy=POLICIES[args.dataset]
    keys=[s.key for s in samples]; selected=samples[args.shard_index::args.num_shards]
    if len(keys)!=entry['total_images'] or len(keys)!=len(set(keys)) or not selected:
        raise RuntimeError('Unique full inventory/shard differs.')
    sig=signature(args,entry,samples,selected,banks,words,checkpoints,identity,protocol)

    def image_q(image):
        source=trial.observations(image,g,v,policy)
        return trial.evidence(source,banks['Complete20'],queries['Complete20'],plans['Complete20'],policy)

    if args.mode=='benchmark':
        official_q,official_settings,official_identity=official_speed_queries(args.dataset,v,{args.dataset:banks['Complete20']})
        indices=sorted({0,len(samples)//2,len(samples)-1})
        smoke_q=[image_q(loader(samples[i]))[0] for i in indices]
        timing_names=(trial.BASE,trial.PRIMARY,trial.VIP,OFFICIAL_SPEED); timings=[]; smoke=[]
        for j,index in enumerate(indices):
            image=loader(samples[index]); peer=smoke_q[(j+1)%len(smoke_q)]
            def call(name,image):
                if name==trial.VIP:
                    return {name:v.predict(image,queries['Complete20'],upstream_settings(args.dataset))[0]},{}
                if name==OFFICIAL_SPEED:
                    query=official_q[args.dataset]
                    pred=official_natural_single(image,v,query,official_settings) if args.dataset=='ade150' else v.predict(image,query,official_settings)[0]
                    return {name:pred},{}
                return trial.predict(image,g,v,banks,queries,plans,args.dataset,(name,),peer_q=peer)
            state('smoke',sample_key=samples[index].key); require_available_gpu(device)
            joint,diag=trial.predict(image,g,v,banks,queries,plans,args.dataset,peer_q=peer)
            old,_=predecessor.predict(image,g,v,banks,queries,plans,args.dataset,
                                     tuple(REPLAY[name] for name in REPLAY if name!=trial.VIP))
            for name,previous in REPLAY.items():
                if name!=trial.VIP and not np.array_equal(joint[name],old[previous]):
                    raise RuntimeError('Independent predecessor replay differs: '+name)
            for name in trial.METHODS:
                if not np.array_equal(call(name,image)[0][name],joint[name]):
                    raise RuntimeError('Singleton/joint prediction differs: '+name)
            source=trial.observations(image,g,v,policy)
            q,crops,_,_=trial.evidence(source,banks['Complete20'],queries['Complete20'],plans['Complete20'],policy)
            fields=trial.wide_fields(source,crops,plans['Complete20'],torch.zeros_like(q),None,
                                    (trial.BASE,trial.PRIMARY,trial.ALIAS_NULL),policy)
            if not torch.equal(fields[trial.BASE],fields[trial.PRIMARY]) or not torch.equal(fields[trial.BASE],fields[trial.ALIAS_NULL]):
                raise RuntimeError('Exact neutral/class-pooled identity differs.')
            smoke.append(dict(sample_key=samples[index].key,independent_predecessor_equal=True,
                singleton_joint_equal=True,neutral_class_pooled_exact=True,diagnostics=diag))
            expected={name:call(name,image)[0][name] for name in timing_names}
            row={name:dict(seconds=[],peaks=[]) for name in timing_names}; state('timing',sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order=timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]
                for name in order:
                    require_available_gpu(device); actual,t=measure(call,(name,image),device,1); require_available_gpu(device)
                    if not np.array_equal(actual[0][name],expected[name]): raise RuntimeError('Warm prediction changed.')
                    row[name]['seconds']+=t['seconds']; row[name]['peaks'].append(t['peak_allocated_mib'])
            for value in row.values():
                value.update(median_seconds=statistics.median(value['seconds']),peak_allocated_mib=max(value.pop('peaks')))
            ratios={name:row[trial.PRIMARY]['median_seconds']/row[name]['median_seconds'] for name in (trial.VIP,OFFICIAL_SPEED)}
            timings.append(dict(sample_key=samples[index].key,timings=row,ratios=ratios,below6x=max(ratios.values())<=6))
            save(output/'timing.json',dict(status='running',images=timings,target_masks_loaded=False))
        passed=all(im['below6x'] for im in timings)
        save(output/'smoke.json',dict(status='complete',images=smoke,text_cost=text_cost,
            target_masks_loaded=False,image_null_smoke='cyclic fixed-three image q; full trial uses separate complete-inventory derangement',
            **check_frozen(frozen,g,v)))
        save(output/'timing.json',dict(status='complete',images=timings,cost_gate_passed=passed,
            target_masks_loaded=False,official_speed_reference=official_identity,whole_image_scope=True,
            current_image_q_recomputed=True,exclusive_gpu_verified=True,
            includes='resize, both encoders, fresh alias weights, Geometry writeback, restoration, argmax',
            excludes='decode and frozen model/text setup',memory_scope='both backbones and banks resident'))
        save(output/'worker_status.json',dict(status='complete' if passed else 'cost_rejected',phase='complete')); return

    for d in protocol['datasets']:
        if not json.loads((root/'benchmark'/d/'timing.json').read_text())['cost_gate_passed']:
            raise RuntimeError('Both cost gates required before prepass/full.')
    if args.mode=='prepass':
        values=[]; totals={}; started=time.perf_counter()
        for number,sample in enumerate(selected,1):
            image=loader(sample); q,_,_,diag=image_q(image); values.append(q.cpu().numpy())
            for k,val in diag.items(): totals[k]=totals.get(k,0.)+val
            if number==1 or number%20==0 or number==len(selected): state('mask_free_q',processed=number,total=len(selected))
        np.savez_compressed(output/'image_q.npz',sample_keys=np.asarray(sig['sample_keys']),q=np.stack(values))
        save(output/'prepass.json',dict(status='complete',processed_images=len(selected),total_images=len(selected),
            signature=sig,target_masks_loaded=False,archive_sha256=file_sha(output/'image_q.npz'),
            diagnostics={k:value/len(selected) for k,value in totals.items()},wall_seconds=time.perf_counter()-started,
            **check_frozen(frozen,g,v)))
        save(output/'worker_status.json',dict(status='complete',phase='mask_free_q',processed=len(selected),total=len(selected))); return

    mapping_path=root/'prepass'/args.dataset/'image_mapping.json'
    mapping=json.loads(mapping_path.read_text()); qp=root/'prepass'/args.dataset/'all_image_q.npz'
    if not mapping.get('mask_free_complete') or mapping['archive_sha256']!=file_sha(qp):
        raise RuntimeError('Frozen mask-free mapping/cache differs.')
    with np.load(qp,allow_pickle=False) as archive: qkeys=archive['sample_keys'].tolist(); bankq=archive['q']
    if qkeys!=sorted(keys) or mapping['recipient_keys']!=qkeys:
        raise RuntimeError('Complete inventory order differs.')
    indices=np.asarray(mapping['donor_indices']); own=np.arange(len(keys))
    if not np.array_equal(np.sort(indices),own) or np.any(indices==own): raise RuntimeError('Image null is not a derangement.')
    lookup={key:i for i,key in enumerate(qkeys)}; classes=banks['Current20'].class_names
    matrices={name:np.zeros((len(classes),len(classes)),np.int64) for name in FULL_METHODS}
    per_image={args.dataset+'__'+name:[] for name in FULL_METHODS}; totals={}; ignored=0
    changes={name:0 for name in (trial.BASE,trial.ALIAS_NULL,trial.IMAGE_NULL,trial.FAMILY)}
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); sig['image_mapping_sha256']=file_sha(mapping_path)
    for number,sample in enumerate(selected,1):
        image=loader(sample); ix=lookup[sample.key]
        peer=torch.from_numpy(bankq[indices[ix]]).to(device); expected=torch.from_numpy(bankq[ix]).to(device)
        predictions,diag=trial.predict(image,g,v,banks,queries,plans,args.dataset,peer_q=peer,expected_q=expected)
        predictions[trial.VIP]=v.predict(image,queries['Complete20'],upstream_settings(args.dataset))[0]
        target=masks(sample,args.dataset,tuple(image.shape[-2:])); valid=(target>=0)&(target<len(classes)); ignored+=int((~valid).sum())
        for name,pred in predictions.items():
            if pred.shape!=target.shape or np.any(pred[valid]>=len(classes)): raise RuntimeError('Invalid prediction shape/class.')
            cm=np.bincount(target[valid].astype(np.int64)*len(classes)+pred[valid],minlength=len(classes)**2).reshape(len(classes),len(classes))
            matrices[name]+=cm; per_image[args.dataset+'__'+name].append(cm)
        for name in changes: changes[name]+=int(((predictions[trial.PRIMARY]!=predictions[name])&valid).sum())
        if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards'] or diag['additional_visual_forwards'] or diag['additional_semantic_heads']:
            raise RuntimeError('Visual budget exceeded.')
        for k,val in diag.items(): totals[k]=totals.get(k,0.)+val
        if number==1 or number%20==0 or number==len(selected):
            row=dict(status='running',processed_images=number,total_images=len(selected),signature=sig,
                metrics={args.dataset:{name:summary(cm,classes,ignored) for name,cm in matrices.items()}},
                diagnostics={args.dataset:{k:value/number for k,value in totals.items()}},primary_pixel_changes=changes,
                target_masks_used_only_after_prediction=True,wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row); state('full',processed=number,total=len(selected))
            print(json.dumps(dict(dataset=args.dataset,shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(sig['sample_keys']),
                        **{name:np.stack(values) for name,values in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v)); save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('benchmark','prepass','full'),required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official'); p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
