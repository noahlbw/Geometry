"""Frozen curated20 input factorial, unchanged mixture and task profiles."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.alias_logprior_mixture import IMPLEMENTATION, BASE, PRIMARY, UNIFORM, SHUFFLED, METHODS, prepare_plan, predict_image
from dinotool.fixed20_task_readout import POLICIES, VIP, predict_image as task_predict
from dinotool.vip_official_adapter import upstream_settings
from alias_curated_setup import setup
from eval_canonical_rival_trial import task_text, official_speed_queries, official_natural_single, OFFICIAL_SPEED
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from merge_gear_ov_shards import merge


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
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,identity,reference,_ = setup(args)
    if reference is not None:
        raise RuntimeError('Unexpected historical model loading.')
    b,q,text_identity = task_text(Path(args.suite_root),args.dataset,g,v,old_b,old_q,identity)
    official_q,official_settings,official_identity = official_speed_queries(args.dataset,v,b)
    plans = {p:prepare_plan(bank) for p,bank in b.items()}
    frozen = frozen_state(g,v)
    names = (*METHODS,VIP)
    timing_names = (*names,OFFICIAL_SPEED)
    def call(name,image):
        if name == VIP:
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        if name == OFFICIAL_SPEED:
            return {p:{name:official_natural_single(image,v,query,official_settings) if args.dataset=='ade150'
                       else v.predict(image,query,official_settings)[0]} for p,query in official_q.items()},{}
        return predict_image(image,g,b,v,q,plans,args.dataset,methods=(name,))
    state('smoke')
    image = loader(samples[0])
    joint,diag = predict_image(image,g,b,v,q,plans,args.dataset)
    independent,_ = task_predict(image,g,b,v,q,args.dataset)
    for p in b:
        if not np.array_equal(joint[p][BASE],independent[p][BASE]):
            raise RuntimeError('Base differs from existing task-profile implementation.')
        for name in METHODS:
            single,_ = call(name,image)
            if not np.array_equal(single[p][name],joint[p][name]):
                raise RuntimeError('Singleton/joint prediction mismatch: '+name)
    save(output/'smoke.json',dict(status='complete',target_masks_loaded=False,
        exact_existing_profile_replay=True,singleton_joint_predictions_equal=True,
        diagnostics=diag,**check_frozen(frozen,g,v)))
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
    save(output/'timing.json',dict(status='complete',images=timings,exclusive_gpu_verified=True,
        official_speed_reference=official_identity,benchmark_predictions_equal=True,target_masks_loaded=False,
        whole_image_scope=True,excludes='decode, model/text/plan setup',
        includes='resize, all encoders/heads, wide alias mixture, original H writer, restoration and argmax',
        memory_scope='shared Geometry/VIP models resident; not standalone VIP deployment memory'))
    state('full',processed=0,total=len(samples))
    keys = [sample.key for sample in samples]
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=names,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),method_sources=protocol['method_sources'],task_text_identity=text_identity),
        competitive=dict(rule='wide additive log(20*softmax(salience)); unchanged local and H writer',
            fixed_alias_slots=20,shuffle_seed=20261008,canonical_protection=False,no_extra_heads=True,no_extra_encodings=True),
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),
        sample_keys=keys,sample_keys_sha256=digest(keys),num_shards=1,shard_index=0,config=vars(args))
    matrices = {p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in names} for p,bank in b.items()}
    per_image = {p+'__'+name:[] for p in b for name in names}
    ignored = dict.fromkeys(b,0)
    totals = {p:dict.fromkeys(diag[p],0.) for p in b}
    changed = {p:dict.fromkeys((BASE,UNIFORM,SHUFFLED),0) for p in b}
    shard = output/'s0';shard.mkdir()
    started = time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(samples,1):
        image = loader(sample)
        predictions,diag = predict_image(image,g,b,v,q,plans,args.dataset)
        vip_predictions,_ = call(VIP,image)
        for p,bank in b.items():
            predictions[p].update(vip_predictions[p])
            target = masks(sample,p,tuple(image.shape[-2:]))
            valid = (target>=0)&(target<bank.class_count)
            ignored[p] += int((~valid).sum())
            for name,prediction in predictions[p].items():
                if prediction.shape != target.shape or np.any(prediction[valid]>=bank.class_count):
                    raise RuntimeError('Invalid prediction shape/class indices.')
                cm = np.bincount(target[valid].astype(np.int64)*bank.class_count+prediction[valid],
                    minlength=bank.class_count**2).reshape(bank.class_count,bank.class_count)
                matrices[p][name] += cm;per_image[p+'__'+name].append(cm)
            for name in changed[p]:
                changed[p][name] += int(((predictions[p][PRIMARY]!=predictions[p][name])&valid).sum())
            if (diag[p]['geometry_encodings']>4 or diag[p]['wide_encodings']>4
                    or diag[p]['fine_forwards'] or diag[p]['additional_visual_forwards']):
                raise RuntimeError('Frozen visual budget exceeded.')
            for key,value in diag[p].items():
                totals[p][key] += float(value)
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
    previous = json.loads(Path(entry['original_run_reference']).read_text())
    if (previous['signature']['sample_keys']!=keys or previous['signature']['checkpoints']!=identity
            or previous['signature']['classes']!=merged['signature']['classes']):
        raise RuntimeError('Full original/curated sample, class or checkpoint identity changed.')
    if previous['signature']['vocabulary']['sha256']==words['sha256']:
        raise RuntimeError('Input factorial accidentally reused the original vocabulary.')
    if words['sha256']!=protocol['input_sha256']['curated20']:
        raise RuntimeError('Curated vocabulary identity differs from the frozen input bank.')
    for p,group in merged['metrics'].items():
        original_targets=np.asarray(previous['metrics'][p][BASE]['confusion_matrix']).sum(-1)
        if not np.array_equal(np.asarray(group[BASE]['confusion_matrix']).sum(-1),original_targets):
            raise RuntimeError('Original/curated scored target counts differ.')
        targets = [np.asarray(value['confusion_matrix']).sum(-1) for value in group.values()]
        if any(not np.array_equal(targets[0],value) for value in targets[1:]):
            raise RuntimeError('Paired scored target counts differ.')
    merged.update(exact_base_confusion_replayed=False,paired_scored_targets_equal=True,
        curated20_identity_verified=True,original_curated_inputs_verified=True,
        primary_pixel_changes=changed,**check_frozen(frozen,g,v))
    save(output/'merged.json',merged)
    state('complete',processed=len(samples),total=len(samples))
    final_state = json.loads((output/'worker_status.json').read_text());final_state['status']='complete'
    save(output/'worker_status.json',final_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--mode',default='full')
    parser.add_argument('--sample-seed',type=int,default=20260923);parser.add_argument('--vdd-ontology',default='official')
    parser.add_argument('--repetitions',type=int,default=5)
    main(parser.parse_args())
