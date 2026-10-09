"""Frozen existing-root trial: mask-free singleton gate, then full scoring."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.generation_root_alias import (IMPLEMENTATION, BASE, QUOTIENT,
    PRIMARY, SHUFFLE, CANONICAL, METHODS, prepare_plan, predict_image)
from dinotool.fixed20_task_readout import POLICIES, VIP, predict_image as task_predict
from dinotool.vip_official_adapter import upstream_settings
from eval_alias_full_comparison_r3 import setup as original_setup
from alias_curated_setup import setup as curated_setup
from eval_canonical_rival_trial import (task_text, official_speed_queries,
    official_natural_single, OFFICIAL_SPEED)
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def main(args):
    output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing trial output; preserve it.')
    device=torch.device(args.device);require_available_gpu(device)
    lease=torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**values):
        save(output/'worker_status.json',dict(status='running',phase=phase,**values))
    state('loading')
    setup=curated_setup if args.dataset=='ade150' else original_setup
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,identity,reference,_=setup(args)
    if reference is not None:raise RuntimeError('Unexpected extra reference model.')
    for relative,expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen trial source changed: '+relative)
    b,q,text_identity=task_text(Path(args.suite_root),args.dataset,g,v,old_b,old_q,identity)
    provenance=protocol['generation_provenance'] if args.dataset=='ade150' else None
    plans={p:prepare_plan(bank,provenance) for p,bank in b.items()}
    frozen=frozen_state(g,v)
    def call(name,image):
        if name==VIP:
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        if name==OFFICIAL_SPEED:
            return {p:{name:official_natural_single(image,v,query,official_settings) if args.dataset=='ade150'
                else v.predict(image,query,official_settings)[0]} for p,query in official_q.items()},{}
        return predict_image(image,g,b,v,q,plans,args.dataset,methods=(name,))
    if args.mode=='benchmark':
        official_q,official_settings,official_identity=official_speed_queries(args.dataset,v,b)
        names=(BASE,PRIMARY,VIP,OFFICIAL_SPEED);images=[];smoke=[]
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            image=loader(samples[index]);require_available_gpu(device)
            state('smoke',sample_key=samples[index].key)
            joint,diag=predict_image(image,g,b,v,q,plans,args.dataset)
            independent,_=task_predict(image,g,b,v,q,args.dataset)
            for p in b:
                if not np.array_equal(joint[p][BASE],independent[p][BASE]):
                    raise RuntimeError('Exact Base independent prediction replay differs.')
                if args.dataset=='ade150':
                    from dinotool.generation_family_alias import prepare_plan as old_plan, predict_image as old_predict
                    old,_=old_predict(image,g,{'C':b[p],'H':b[p]},v,{'C':q[p],'H':q[p]},
                        old_plan(b[p],provenance['family_ids']),None,methods=('CQ',))
                    if not np.array_equal(joint[p][QUOTIENT],old['CQ']):
                        raise RuntimeError('Existing CQ independent prediction replay differs.')
                elif any(not np.array_equal(joint[p][BASE],joint[p][name]) for name in (PRIMARY,QUOTIENT,SHUFFLE)):
                    raise RuntimeError('No-provenance fallback changed predictions.')
            for name in METHODS:
                single,_=call(name,image)
                if any(not np.array_equal(single[p][name],joint[p][name]) for p in b):
                    raise RuntimeError('Singleton/joint prediction mismatch: '+name)
            smoke.append(dict(sample_key=samples[index].key,exact_base_replayed=True,
                exact_cq_replayed=args.dataset=='ade150',exact_no_provenance_fallback=args.dataset=='vdd',
                all_singleton_joint_predictions_equal=True,diagnostics=diag))
            expected={name:call(name,image)[0] for name in names}
            row={name:dict(seconds=[],peaks=[]) for name in names}
            state('timing',sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in order:
                    require_available_gpu(device)
                    actual,timing=measure(call,(name,image),device,1)
                    require_available_gpu(device)
                    if any(not np.array_equal(actual[0][p][name],expected[name][p][name]) for p in b):
                        raise RuntimeError('Warmed singleton prediction changed.')
                    row[name]['seconds']+=timing['seconds'];row[name]['peaks'].append(timing['peak_allocated_mib'])
            for value in row.values():
                value.update(median_seconds=statistics.median(value['seconds']),
                    peak_allocated_mib=max(value.pop('peaks')))
            ratios={name:row[PRIMARY]['median_seconds']/row[name]['median_seconds'] for name in (VIP,OFFICIAL_SPEED)}
            images.append(dict(sample_key=samples[index].key,image_size=list(image.shape[-2:]),
                timings=row,ratios=ratios,below6x=max(ratios.values())<=6))
            save(output/'timing.json',dict(status='running',images=images,target_masks_loaded=False))
        passed=all(row['below6x'] for row in images)
        save(output/'smoke.json',dict(status='complete',images=smoke,target_masks_loaded=False,**check_frozen(frozen,g,v)))
        save(output/'timing.json',dict(status='complete',images=images,cost_gate_passed=passed,
            exclusive_gpu_verified=True,whole_image_scope=True,target_masks_loaded=False,
            official_speed_reference=official_identity,excludes='decode, model/text/plan setup',
            includes='resize, bounded encoders, original20 salience and root aggregation, H, restoration/argmax',
            memory_scope='both backbones and fixed20 text caches resident; not isolated VIP deployment memory'))
        save(output/'worker_status.json',dict(status='complete' if passed else 'cost_rejected',phase='complete'))
        return
    for d in protocol['datasets']:
        timing=json.loads((Path(args.suite_root)/'benchmark'/d/'timing.json').read_text())
        if not timing.get('cost_gate_passed'):raise RuntimeError('Full masks forbidden before both cost gates.')
    if not 0<=args.shard_index<args.num_shards:raise RuntimeError('Invalid shard.')
    keys=[sample.key for sample in samples];selected=samples[args.shard_index::args.num_shards]
    if len(keys)!=entry['total_images'] or len(keys)!=len(set(keys)) or not selected:
        raise RuntimeError('Full unique inventory changed.')
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),sources=protocol['method_sources'],
            task_text_identity=text_identity,generation_provenance=provenance),
        competitive=None,global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected],sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args))
    matrices={p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in METHODS} for p,bank in b.items()}
    per_image={p+'__'+name:[] for p in b for name in METHODS};ignored=dict.fromkeys(b,0);totals={p:{} for p in b}
    changed={p:dict.fromkeys((BASE,QUOTIENT,SHUFFLE,CANONICAL),0) for p in b}
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image=loader(sample);predictions,diag=predict_image(image,g,b,v,q,plans,args.dataset)
        for p,bank in b.items():
            target=masks(sample,p,tuple(image.shape[-2:]));valid=(target>=0)&(target<bank.class_count)
            ignored[p]+=int((~valid).sum())
            for name,pred in predictions[p].items():
                if pred.shape!=target.shape or np.any(pred[valid]>=bank.class_count):
                    raise RuntimeError('Invalid full-size prediction/class indices.')
                cm=np.bincount(target[valid].astype(np.int64)*bank.class_count+pred[valid],
                    minlength=bank.class_count**2).reshape(bank.class_count,bank.class_count)
                matrices[p][name]+=cm;per_image[p+'__'+name].append(cm)
            for name in changed[p]:changed[p][name]+=int(((predictions[p][PRIMARY]!=predictions[p][name])&valid).sum())
            if (diag[p]['geometry_encodings']>4 or diag[p]['wide_encodings']>4 or diag[p]['fine_forwards']
                    or diag[p]['additional_visual_forwards'] or diag[p]['additional_semantic_heads']):
                raise RuntimeError('Frozen visual budget exceeded.')
            for key,value in diag[p].items():totals[p][key]=totals[p].get(key,0.)+value
        if number==1 or number%20==0 or number==len(selected):
            row=dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={p:{name:summary(cm,b[p].class_names,ignored[p]) for name,cm in group.items()} for p,group in matrices.items()},
                diagnostics={p:{k:x/number for k,x in values.items()} for p,values in totals.items()},
                primary_pixel_changes=changed,target_masks_used_only_after_prediction=True,
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row);state('full',processed=number,total=len(selected))
            print(json.dumps(dict(dataset=args.dataset,shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{key:np.stack(value) for key,value in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v));save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('benchmark','full'),required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--sample-seed',type=int,default=20260923)
    parser.add_argument('--vdd-ontology',default='official');parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--num-shards',type=int,default=1);parser.add_argument('--shard-index',type=int,default=0)
    main(parser.parse_args())
