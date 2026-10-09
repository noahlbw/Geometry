"""Frozen original20 task profiles and a single component-penalty hypothesis."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.canonical_rival_penalty import (IMPLEMENTATION, BASE, PRIMARY, POOLED,
    SHUFFLED, METHODS, prepare_plan, predict_image)
from dinotool.fixed20_task_readout import POLICIES, VIP, predict_image as task_predict
from dinotool.inference import tile_starts
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, _load_upstream, upstream_settings, upstream_aliases
from eval_alias_full_comparison_r3 import setup
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from merge_gear_ov_shards import merge


OFFICIAL_SPEED = 'VIP_OfficialQueries'


@torch.inference_mode()
def official_speed_queries(dataset, vip, banks):
    upstream = Path(vip.upstream_root)
    if dataset == 'vdd':
        path = upstream/'configs/cls_vdd.txt'
        settings, family = upstream_settings(dataset), 'openai_imagenet_template'
    else:
        from dinotool.natural_text_adaptation import CLASS_FILES
        from eval_natural_text_adaptation import official_options
        path = upstream/'configs'/('cls_'+CLASS_FILES[dataset]+'.txt')
        settings, family = official_options(upstream,dataset)
    groups = upstream_aliases(path)
    prompt = _load_upstream(upstream/'prompts/imagenet_template.py', 'canonical_rival_official_speed_prompts')
    previous = vip.templates;vip.templates = prompt.get_text_template(family)
    try:
        queries = {p:vip.encode_queries(bank.class_names,groups) for p,bank in banks.items()}
    finally:
        vip.templates = previous
    return queries,settings,dict(source=str(path),template=family,settings=asdict(settings),
        alias_counts=[len(group) for group in groups],
        resolution='upstream short336/max2048' if dataset=='ade150' else 'official long448',
        numerical_repair='self-Value on empty proxy rows only')


@torch.inference_mode()
def official_natural_single(image, vip, query, settings):
    """The existing official_prediction protocol, returning its default arm only."""
    h,w = image.shape[-2:]
    ratio = min(336/min(h,w),2048/max(h,w));nh,nw = int(h*ratio+.5),int(w*ratio+.5)
    array = (image.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
    resized = torch.from_numpy(np.ascontiguousarray(cv2.resize(array,(nw,nh)))).permute(2,0,1).float()/255
    total = torch.zeros(len(query.class_names),nh,nw,device=vip.device)
    count = torch.zeros(nh,nw,device=vip.device)
    for top in tile_starts(nh,336,224):
        for left in tile_starts(nw,336,224):
            ah,aw = min(336,nh-top),min(336,nw-left)
            crop = F.pad(resized[:,top:top+ah,left:left+aw],(0,336-aw,0,336-ah))
            total[:,top:top+ah,left:left+aw] += vip.crop_logits(crop,query,settings)[:,:ah,:aw]
            count[top:top+ah,left:left+aw] += 1
    probability = F.interpolate((total/count)[None],(h,w),mode='bilinear',align_corners=False)[0].softmax(0)
    prediction = probability.argmax(0)
    if settings.prob_thd > 0:
        prediction = prediction.masked_fill(probability.amax(0)<settings.prob_thd,
                                             settings.bg_idx if settings.background else 255)
    return prediction.to(torch.uint8).cpu().numpy()


@torch.inference_mode()
def task_text(root, dataset, geometry, vip, banks, queries, checkpoint_identity):
    policy = POLICIES[dataset]
    path = root/'text_cache'/(dataset+'.pt')
    identity = dict(template=policy.template, checkpoints=checkpoint_identity,
        classes={p:list(b.class_names) for p,b in banks.items()},
        aliases={p:list(b.alias_names) for p,b in banks.items()})
    stored = torch.load(path, map_location=geometry.device, weights_only=True) if path.exists() else None
    if stored and stored['identity'] != identity:
        raise RuntimeError('Frozen task text identity changed.')
    output_banks, output_queries, encoded = {}, {}, {}
    for p, bank in banks.items():
        original = queries[p]
        if stored:
            row = stored['encoded'][p]
            query = VIPQueries(row['wide'], original.parents, original.class_names, original.aliases)
            local = row['local']
        else:
            query = original
            if policy.template == 'seg_template':
                module = _load_upstream(Path(vip.upstream_root)/'prompts/imagenet_template.py', 'canonical_rival_templates')
                previous = vip.templates
                vip.templates = module.get_text_template(policy.template)
                groups = tuple(tuple(a for i,a in enumerate(original.aliases)
                                     if int(original.parents[i]) == c) for c in range(bank.class_count))
                try:
                    query = vip.encode_queries(original.class_names, groups)
                finally:
                    vip.templates = previous
            local = F.normalize(query.features.float().mean(1), dim=-1)
            encoded[p] = dict(local=local.cpu(), wide=query.features.cpu())
        new = TCPRTextBank(local, bank.parent_indices, bank.canonical_mask,
                          bank.class_names, bank.alias_names)
        new.validate()
        if query.aliases != original.aliases or not torch.equal(query.parents, original.parents):
            raise RuntimeError('Task policy changed original20 strings or parents.')
        output_banks[p], output_queries[p] = new, query
    if stored is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(identity=identity, encoded=encoded), temporary)
        temporary.rename(path)
    return output_banks, output_queries, identity


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    device = torch.device(args.device)
    require_available_gpu(device)
    lease = torch.empty(1, device=device)
    output.mkdir(parents=True)
    def state(phase, **fields):
        save(output/'worker_status.json', dict(status='running', phase=phase, **fields))
    state('loading')
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,identity,reference,_ = setup(args)
    if reference is not None:
        raise RuntimeError('Unexpected historical model loading.')
    b,q,text_identity = task_text(Path(args.suite_root), args.dataset, g,v,old_b,old_q,identity)
    official_q,official_settings,official_identity = official_speed_queries(args.dataset,v,b)
    plans = {p:prepare_plan(bank) for p,bank in b.items()}
    frozen = frozen_state(g,v)
    names = (*METHODS,VIP)
    timing_names = (*names,OFFICIAL_SPEED)
    def call(name,image):
        if name == OFFICIAL_SPEED:
            if args.dataset == 'ade150':
                return {p:{name:official_natural_single(image,v,query,official_settings)}
                        for p,query in official_q.items()},{}
            return {p:{name:v.predict(image,query,official_settings)[0]} for p,query in official_q.items()},{}
        if name == VIP:
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        return predict_image(image,g,b,v,q,plans,args.dataset,methods=(name,))
    state('smoke')
    image = loader(samples[0])
    joint,diag = predict_image(image,g,b,v,q,plans,args.dataset)
    independent,_ = task_predict(image,g,b,v,q,args.dataset)
    if args.dataset == 'ade150':
        from eval_natural_text_adaptation import official_prediction
        official_single,_ = call(OFFICIAL_SPEED,image)
        for p in b:
            old_official = official_prediction(image,v,official_q[p],official_settings)[0]
            if not np.array_equal(official_single[p][OFFICIAL_SPEED],old_official):
                raise RuntimeError('Singleton official VIP protocol differs from preserved official prediction.')
    for p in b:
        if not np.array_equal(joint[p][BASE],independent[p][BASE]):
            raise RuntimeError('Fixed task baseline differs from existing profile implementation.')
        for name in METHODS:
            single,_ = call(name,image)
            if not np.array_equal(joint[p][name],single[p][name]):
                raise RuntimeError('Singleton/joint mismatch: '+name)
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
            for name in timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]:
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
        del expected,image
    save(output/'timing.json',dict(status='complete',images=timings,exclusive_gpu_verified=True,
        official_speed_reference=official_identity,
        benchmark_predictions_equal=True,target_masks_loaded=False,whole_image_scope=True,
        excludes='decode, model/text/plan setup',includes='resize, all encoders/heads, local alias penalty, writer, full restoration and argmax',
        memory_scope='shared Geometry/VIP models resident; not standalone VIP deployment memory'))
    state('full',processed=0,total=len(samples))
    all_keys = [s.key for s in samples]
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=names,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),method_sources=protocol['method_sources'],
                  task_text_identity=text_identity),
        competitive=dict(fixed_alias_slots=20,rule='nearest foreign canonical orthogonal component, alpha1, local only, canonical protection',
            shuffle_seed=20261008,no_extra_heads=True,no_extra_encodings=True),
        global_sample_count=len(all_keys),global_sample_keys_sha256=digest(all_keys),
        sample_keys=all_keys,sample_keys_sha256=digest(all_keys),num_shards=1,shard_index=0,config=vars(args))
    matrices = {p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in names} for p,bank in b.items()}
    per_image = {p+'__'+name:[] for p in b for name in names}
    ignored = dict.fromkeys(b,0)
    totals = {p:dict.fromkeys(diag[p],0.) for p in b}
    started = time.perf_counter();torch.cuda.reset_peak_memory_stats()
    shard = output/'s0';shard.mkdir()
    for number,sample in enumerate(samples,1):
        image = loader(sample)
        predictions,diag = predict_image(image,g,b,v,q,plans,args.dataset)
        vip_predictions,_ = call(VIP,image)
        for p,bank in b.items():
            predictions[p].update(vip_predictions[p])
            # Masks are first loaded after all predictions for this image.
            target = masks(sample,p,tuple(image.shape[-2:]))
            valid = (target >= 0)&(target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for name,prediction in predictions[p].items():
                cm = np.bincount(target[valid].astype(np.int64)*bank.class_count+prediction[valid],
                    minlength=bank.class_count**2).reshape(bank.class_count,bank.class_count)
                matrices[p][name] += cm;per_image[p+'__'+name].append(cm)
            for key,value in diag[p].items():
                totals[p][key] += float(value)
        if number == 1 or number % 10 == 0 or number == len(samples):
            row = dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={p:{name:summary(cm,bank.class_names,ignored[p]) for name,cm in matrices[p].items()} for p,bank in b.items()},
                ignored_pixels=ignored,diagnostics={p:{key:value/number for key,value in fields.items()} for p,fields in totals.items()},
                target_masks_used_only_after_prediction=True,wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(shard/'results.json',row);state('full',processed=number,total=len(samples))
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples),wall_seconds=row['wall_seconds'])),flush=True)
        del predictions,vip_predictions,image,target
    np.savez_compressed(shard/'per_image_confusions.npz',sample_keys=np.asarray(all_keys),
        **{key:np.stack(values) for key,values in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v));save(shard/'results.json',row)
    merged = merge([str(shard)],str(output/'merged.json'))
    previous = json.loads(Path(entry['matched_patch_result']).read_text())
    if merged['signature']['sample_keys'] != previous['signature']['sample_keys']:
        raise RuntimeError('Full original20 sample sequence differs.')
    for p,group in merged['metrics'].items():
        target_counts = [np.asarray(value['confusion_matrix']).sum(-1) for value in group.values()]
        if any(not np.array_equal(target_counts[0],value) for value in target_counts[1:]):
            raise RuntimeError('Paired scored target counts differ.')
        if entry.get('exact_task_reference'):
            archived = json.loads(Path(entry['exact_task_reference']).read_text())
            if archived['signature']['sample_keys'] != merged['signature']['sample_keys']:
                raise RuntimeError('Task replay sample sequence differs.')
            if not np.array_equal(group[BASE]['confusion_matrix'],archived['metrics'][p]['DomainShared']['confusion_matrix']):
                raise RuntimeError('Full VDD prior-developed original20 profile confusion differs.')
    merged.update(paired_scored_targets_equal=True,original20_identity_verified=True,
        exact_task_reference_replayed=bool(entry.get('exact_task_reference')),**check_frozen(frozen,g,v))
    save(output/'merged.json',merged)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(samples),total=len(samples)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='full')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
