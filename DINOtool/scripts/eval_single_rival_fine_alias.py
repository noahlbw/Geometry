"""Frozen single-rival fine trial; singleton cost gates full-mask evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.single_rival_fine_alias import (IMPLEMENTATION,BASE,NOALIAS,PRIMARY,POOLED,
    SHUFFLED,METHODS,prepare_plan,predict_image)
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.fixed20_task_readout import POLICIES,VIP,predict_image as task_predict
from dinotool.vip_official_adapter import upstream_settings
from eval_alias_full_comparison_r3 import setup as original_setup
from alias_curated_setup import setup as curated_setup
from eval_canonical_rival_trial import (task_text,official_speed_queries,
    official_natural_single,OFFICIAL_SPEED)
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen,frozen_state,save
from merge_gear_ov_shards import merge


def verify_diagnostics(diag,classes):
    for p,row in diag.items():
        c = classes[p]
        if (row['geometry_encodings']>4 or row['wide_encodings']>4 or not 1<=row['fine_forwards']<=16
                or row['canonical_risk_max']!=0 or row['delta_positive_max']>1e-9
                or row['fine_coverage_max_error']>1e-6
                or max(row['pooled_mass_error'],row['shuffled_mass_error'])>1e-5
                or row['maximum_stencil_elements']>32*16*c*20 or row['maximum_risk_elements']>1024*c*20):
            raise RuntimeError('Single-rival rule/linear workspace/bounded observations violated: '+p)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    device = torch.device(args.device)
    require_available_gpu(device)
    lease = torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**fields):
        save(output/'worker_status.json',dict(status='running',phase=phase,**fields))
    state('loading')
    setup = curated_setup if args.dataset=='ade150' else original_setup
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,identity,reference,_ = setup(args)
    if reference is not None:
        raise RuntimeError('Unexpected historical model loading.')
    b,q,text_identity = task_text(Path(args.suite_root),args.dataset,g,v,old_b,old_q,identity)
    official_q,official_settings,official_identity = official_speed_queries(args.dataset,v,b)
    plans = {p:prepare_plan(bank) for p,bank in b.items()}
    classes = {p:bank.class_count for p,bank in b.items()}
    execution = FineCoverageExecution(cached=True,burst=True)
    frozen = frozen_state(g,v)
    names = (*METHODS,VIP)
    timing_names = (*names,OFFICIAL_SPEED)
    def call(name,image):
        if name == VIP:
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        if name == OFFICIAL_SPEED:
            return {p:{name:official_natural_single(image,v,query,official_settings) if args.dataset=='ade150'
                       else v.predict(image,query,official_settings)[0]} for p,query in official_q.items()},{}
        return predict_image(image,g,b,v,q,plans,args.dataset,methods=(name,),execution=execution)
    state('smoke')
    image = loader(samples[0])
    started = time.perf_counter()
    joint,diag = predict_image(image,g,b,v,q,plans,args.dataset,execution=execution)
    cold_seconds = time.perf_counter()-started
    independent,_ = task_predict(image,g,b,v,q,args.dataset)
    for p in b:
        if not np.array_equal(joint[p][BASE],independent[p][BASE]):
            raise RuntimeError('Base differs from existing same-input task profile.')
        for name in METHODS:
            single,_ = call(name,image)
            if not np.array_equal(single[p][name],joint[p][name]):
                raise RuntimeError('Singleton/joint prediction mismatch: '+name)
    verify_diagnostics(diag,classes)
    save(output/'smoke.json',dict(status='complete',target_masks_loaded=False,
        exact_existing_profile_replay=True,singleton_joint_predictions_equal=True,
        diagnostics=diag,cold_joint_seconds=cold_seconds,fine_execution=execution.report(),**check_frozen(frozen,g,v)))
    del image,joint,independent
    state('benchmark')
    timings = []
    for index in sorted({0,len(samples)//2,len(samples)-1}):
        sample = samples[index];image = loader(sample)
        require_available_gpu(device)
        expected = {name:call(name,image)[0] for name in timing_names}
        row = {name:dict(seconds=[],peaks=[]) for name in timing_names}
        for repeat in range(args.repetitions):
            order = timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]
            for name in order:
                require_available_gpu(device)
                actual,timing = measure(call,(name,image),device,1)
                require_available_gpu(device)
                if any(not np.array_equal(actual[0][p][name],expected[name][p][name]) for p in b):
                    raise RuntimeError('Warmed singleton prediction changed.')
                row[name]['seconds'] += timing['seconds']
                row[name]['peaks'].append(timing['peak_allocated_mib'])
        for value in row.values():
            value.update(median_seconds=statistics.median(value['seconds']),peak_allocated_mib=max(value.pop('peaks')))
        timings.append(dict(sample_key=sample.key,image_size=list(image.shape[-2:]),timings=row))
        save(output/'timing.json',dict(status='running',images=timings,target_masks_loaded=False))
        del expected,image,actual
    ratios = [{name:row['timings'][PRIMARY]['median_seconds']/row['timings'][name]['median_seconds']
               for name in (VIP,OFFICIAL_SPEED)} for row in timings]
    passed_cost = all(max(row.values())<=6 for row in ratios)
    save(output/'timing.json',dict(status='complete',images=timings,exclusive_gpu_verified=True,
        official_speed_reference=official_identity,benchmark_predictions_equal=True,target_masks_loaded=False,
        whole_image_scope=True,excludes='decode, model/text/plan setup and separately recorded CUDA-graph cold setup',
        includes='resize, all Geometry/wide/fine encoders, text projection, sparse alias processing, H, restoration and argmax',
        fine_execution=execution.report(),cold_joint_seconds=cold_seconds,cost_gate_passed=passed_cost,
        per_image_primary_ratios=ratios,
        memory_scope='shared Geometry/VIP models and capture pools resident; not standalone deployment memory'))
    if not passed_cost:
        save(output/'worker_status.json',dict(status='cost_rejected',phase='terminal',processed=0,total=len(samples),
            target_masks_loaded=False,reason='Primary exceeded6x on at least one fixed complete image/reference.',ratios=ratios))
        return
    state('full',processed=0,total=len(samples))
    keys = [sample.key for sample in samples]
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=names,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),method_sources=protocol['method_sources'],task_text_identity=text_identity),
        competitive=dict(rule=protocol['rule'],fixed_alias_slots=20,shuffle_seed=20261008,
            canonical_protection=True,online_rivals_per_class=1,maximum_fine_encodings=16),
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),sample_keys=keys,
        sample_keys_sha256=digest(keys),num_shards=1,shard_index=0,config=vars(args))
    matrices = {p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in names} for p,bank in b.items()}
    per_image = {p+'__'+name:[] for p in b for name in names}
    ignored = dict.fromkeys(b,0)
    totals = {p:dict.fromkeys(diag[p],0.) for p in b}
    changed = {p:dict.fromkeys((BASE,NOALIAS,POOLED,SHUFFLED),0) for p in b}
    shard = output/'s0';shard.mkdir()
    started = time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(samples,1):
        image = loader(sample)
        predictions,diag = predict_image(image,g,b,v,q,plans,args.dataset,execution=execution)
        vip_predictions,_ = call(VIP,image)
        verify_diagnostics(diag,classes)
        for p,bank in b.items():
            predictions[p].update(vip_predictions[p])
            target = masks(sample,p,tuple(image.shape[-2:]))
            valid = (target>=0)&(target<bank.class_count)
            ignored[p] += int((~valid).sum())
            for name,prediction in predictions[p].items():
                if prediction.shape!=target.shape or np.any(prediction[valid]>=bank.class_count):
                    raise RuntimeError('Invalid prediction shape/class indices.')
                cm = np.bincount(target[valid].astype(np.int64)*bank.class_count+prediction[valid],
                    minlength=bank.class_count**2).reshape(bank.class_count,bank.class_count)
                matrices[p][name] += cm;per_image[p+'__'+name].append(cm)
            for name in changed[p]:
                changed[p][name] += int(((predictions[p][PRIMARY]!=predictions[p][name])&valid).sum())
            for key,value in diag[p].items():totals[p][key] += float(value)
        if number==1 or number%20==0 or number==len(samples):
            row = dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={p:{name:summary(cm,bank.class_names,ignored[p]) for name,cm in matrices[p].items()} for p,bank in b.items()},
                diagnostics={p:{key:value/number for key,value in fields.items()} for p,fields in totals.items()},
                primary_pixel_changes=changed,target_masks_used_only_after_prediction=True,
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(shard/'results.json',row);state('full',processed=number,total=len(samples))
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples),wall_seconds=row['wall_seconds'])),flush=True)
        del predictions,vip_predictions,image,target
    np.savez_compressed(shard/'per_image_confusions.npz',sample_keys=np.asarray(keys),
        **{key:np.stack(value) for key,value in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v));save(shard/'results.json',row)
    merged = merge([str(shard)],str(output/'merged.json'),diagnostic_weight='images')
    previous = json.loads(Path(entry['exact_base_reference']).read_text())
    if (previous['signature']['sample_keys']!=keys or previous['signature']['checkpoints']!=identity
            or previous['signature']['classes']!=merged['signature']['classes']
            or previous['signature']['vocabulary']['sha256']!=words['sha256']):
        raise RuntimeError('Full same-input sample, vocabulary, class or checkpoint identity changed.')
    for p,group in merged['metrics'].items():
        if group[BASE]['confusion_matrix']!=previous['metrics'][p][BASE]['confusion_matrix']:
            raise RuntimeError('Full same-input Base confusion changed.')
        targets = [np.asarray(value['confusion_matrix']).sum(-1) for value in group.values()]
        if any(not np.array_equal(targets[0],value) for value in targets[1:]):
            raise RuntimeError('Paired scored target counts differ.')
    merged.update(exact_base_confusion_replayed=True,paired_scored_targets_equal=True,
        primary_pixel_changes=changed,**check_frozen(frozen,g,v))
    save(output/'merged.json',merged)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(samples),total=len(samples)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--mode',default='full')
    parser.add_argument('--sample-seed',type=int,default=20260923);parser.add_argument('--vdd-ontology',default='official')
    parser.add_argument('--repetitions',type=int,default=5)
    main(parser.parse_args())
