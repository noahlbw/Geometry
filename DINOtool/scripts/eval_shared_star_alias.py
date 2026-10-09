"""VDD/ADE fixed20 star experiment: mask-free checks, timing, then full masks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.shared_star_alias import (IMPLEMENTATION, PROTOCOL, BASELINE, MEAN,
    PRIMARY, BUDGET4, POOLED, SHUFFLED, predict_image)
from dinotool.shared_local_alias import predict_image as source_predict
from dinotool.vip_official_adapter import upstream_settings
from eval_alias_full_comparison_r3 import setup
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import frozen_state, check_frozen, save
from merge_gear_ov_shards import merge

METHODS = (BASELINE, MEAN, PRIMARY, BUDGET4, POOLED, SHUFFLED)
VIP = 'VIP_Matched20'


@torch.inference_mode()
def main(args):
    root = Path(args.output_dir)
    if root.exists():
        raise RuntimeError('Existing worker output; preserve it.')
    require_available_gpu(torch.device(args.device))
    # Acquire the device before loading checkpoints so independent dispatchers
    # observe this experiment's occupied GPU throughout all phases.
    lease = torch.empty(1, device=args.device)
    root.mkdir(parents=True)
    def state(phase, **fields):
        save(root/'worker_status.json', dict(status='running',phase=phase,**fields))
    state('loading')
    protocol,entry,samples,loader,masks,g,b,v,q,words,identity,reference,_ = setup(args)
    if reference is not None:
        raise RuntimeError('This same-word experiment must not load historical task model.')
    frozen = frozen_state(g,v)
    cache = CheckedGroupedCache();cache.verify = False
    def call(name,image):
        if name == VIP:
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        return predict_image(image,g,b,v,q,methods=(name,),cache=cache)
    def load(sample):
        return loader(sample.image_path if args.dataset == 'loveda' else sample)
    state('smoke')
    image = load(samples[0])
    joint,diag = predict_image(image,g,b,v,q,methods=METHODS,cache=cache)
    original,_ = source_predict(image,g,b,v,q,methods=(BASELINE,MEAN),cache=cache)
    for p in b:
        for name in (BASELINE,MEAN):
            if not np.array_equal(joint[p][name],original[p][name]):
                raise RuntimeError('Original same-source endpoint changed: '+name)
        for name in (PRIMARY,BUDGET4):
            single,_ = call(name,image)
            if not np.array_equal(single[p][name],joint[p][name]):
                raise RuntimeError('Singleton star differs from joint controls.')
    save(root/'smoke.json',dict(status='complete',implementation=IMPLEMENTATION,
        original_endpoints_exact=True,singleton_primary_exact=True,target_masks_loaded=False,
        diagnostics=diag,**check_frozen(frozen,g,v)))
    del image,joint,original
    state('benchmark')
    timing_names = (BASELINE,MEAN,PRIMARY,BUDGET4,VIP)
    timings = []
    for index in sorted({0,len(samples)//2,len(samples)-1}):
        sample = samples[index];image = load(sample)
        expected = {name:call(name,image)[0] for name in timing_names}
        row = {name:dict(seconds=[],peaks=[]) for name in timing_names}
        for repeat in range(args.repetitions):
            order = timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]
            for name in order:
                actual,t = measure(call,(name,image),torch.device(args.device),1)
                if any(not np.array_equal(actual[0][p][name],expected[name][p][name]) for p in b):
                    raise RuntimeError('Warmed singleton prediction changed.')
                row[name]['seconds'] += t['seconds']
                row[name]['peaks'].append(t['peak_allocated_mib'])
        for item in row.values():
            item.update(median_seconds=statistics.median(item['seconds']),peak_allocated_mib=max(item.pop('peaks')))
        timings.append(dict(sample_key=sample.key,image_size=list(image.shape[-2:]),timings=row))
        save(root/'timing.json',dict(status='running',images=timings,implementation=IMPLEMENTATION,
            target_masks_loaded=False,memory_scope='one model arm at a time; shared Geometry/VIP models and frozen head snapshots resident',
            scope='exclusive physical GPU; other GPU jobs allowed; full loaded image, resize/branches/alias/writeback/restoration/argmax; excludes decode and initialization'))
    timing = json.loads((root/'timing.json').read_text())
    timing.update(status='complete',benchmark_predictions_equal=True,**check_frozen(frozen,g,v))
    save(root/'timing.json',timing)
    del expected,image,actual
    state('full',processed=0,total=len(samples))
    output = root/'s0';output.mkdir()
    keys = [sample.key for sample in samples]
    methods = (*METHODS,VIP)
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=methods,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(geometry=asdict(g.config),sources=protocol['method_sources']),competitive=PROTOCOL,
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),num_shards=1,shard_index=0,
        sample_keys=keys,sample_keys_sha256=digest(keys),config=vars(args))
    matrices = {p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in methods} for p,bank in b.items()}
    per_image = {p+'__'+name:[] for p in b for name in methods}
    ignored = dict.fromkeys(b,0)
    flips = {name:dict(wrong_to_correct=0,correct_to_wrong=0,changed=0) for name in METHODS[2:]}
    totals = {p:dict(tiles=0) for p in b}
    started = time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(samples,1):
        image = load(sample)
        predictions,diagnostics = predict_image(image,g,b,v,q,methods=METHODS,cache=cache)
        vip_pred,_ = call(VIP,image)
        for p in b:
            predictions[p].update(vip_pred[p])
            diag = diagnostics[p]
            if (diag['fine_forwards'] or diag['additional_visual_forwards'] or diag['geometry_encodings']>4
                    or diag['wide_encodings']>4 or diag['canonical_risk_max'] or diag['maximum_directed_edges']>4
                    or diag['maximum_risk_elements']>1024*4*20):
                raise RuntimeError('Sparse observation/alias budget changed.')
            totals[p]['tiles'] += diag['tiles']
            for field,value in diag.items():
                if field != 'tiles':totals[p][field] = totals[p].get(field,0.)+value*diag['tiles']
            target = masks(sample,p,tuple(image.shape[-2:]))
            valid = (target>=0)&(target<b[p].class_count);ignored[p] += int((~valid).sum())
            anchor = predictions[p][MEAN]
            for name in methods:
                pred = predictions[p][name]
                if pred.shape != target.shape or np.any((pred[valid]<0)|(pred[valid]>=b[p].class_count)):
                    raise RuntimeError('Invalid class prediction.')
                encoded = target[valid].astype(np.int64)*b[p].class_count+pred[valid]
                cm = np.bincount(encoded,minlength=b[p].class_count**2).reshape(b[p].class_count,-1)
                matrices[p][name] += cm;per_image[p+'__'+name].append(cm)
                if name in flips:
                    flips[name]['changed'] += int(((pred!=anchor)&valid).sum())
                    flips[name]['wrong_to_correct'] += int(((anchor!=target)&(pred==target)&valid).sum())
                    flips[name]['correct_to_wrong'] += int(((anchor==target)&(pred!=target)&valid).sum())
        if number==1 or number%20==0 or number==len(samples):
            row = dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={p:{name:summary(cm,b[p].class_names,ignored[p]) for name,cm in group.items()} for p,group in matrices.items()},
                diagnostics={p:{field:value if field=='tiles' else value/totals[p]['tiles'] for field,value in diag.items()} for p,diag in totals.items()},
                flips_against_same_source_mean=flips,target_masks_used_only_after_prediction=True,
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row);state('full',processed=number,total=len(samples))
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples),seconds=row['wall_seconds'])),flush=True)
        del predictions,vip_pred,image,target
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(keys),
        **{key:np.stack(values) for key,values in per_image.items()})
    row.update(status='complete',all_visual_budgets_verified=True,**check_frozen(frozen,g,v))
    save(output/'results.json',row)
    merged = merge([str(output)],str(root/'merged.json'))
    previous = json.loads(Path(entry['matched_patch_result']).read_text())
    if merged['signature']['sample_keys']!=previous['signature']['sample_keys']:
        raise RuntimeError('Full sample sequence changed.')
    for p,group in merged['metrics'].items():
        old_p = p if entry['family']=='remote_sensing' else p+'__original20'
        if not np.array_equal(group[BASELINE]['confusion_matrix'],previous['metrics'][old_p][BASELINE]['confusion_matrix']):
            raise RuntimeError('Full matched Patch2 confusion changed.')
        targets = [np.asarray(value['confusion_matrix']).sum(-1) for value in group.values()]
        if any(not np.array_equal(targets[0],value) for value in targets[1:]):
            raise RuntimeError('Paired scored target counts differ.')
    merged.update(matched_patch_full_confusion_exact=True,paired_scored_targets_equal=True,
        **check_frozen(frozen,g,v))
    save(root/'merged.json',merged)
    save(root/'worker_status.json',dict(status='complete',phase='complete',processed=len(samples),total=len(samples),
        result=str(root/'merged.json'),timing=str(root/'timing.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='full')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
