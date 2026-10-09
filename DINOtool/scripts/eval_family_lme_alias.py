"""One frozen wide-only family rule, paired source controls, whole-image cost."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.family_lme_alias import (IMPLEMENTATION, METHODS, CURRENT_BASE, COMPLETE_BASE,
                                      PRIMARY, SHUFFLE, VIP, prepare_plan, predict)
from dinotool.fixed20_task_readout import POLICIES, predict_image as task_predict
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, _load_upstream, upstream_settings
from eval_alias_full_comparison_r3 import setup as original_setup
from alias_curated_setup import setup as curated_setup
from eval_canonical_rival_trial import (task_text, official_speed_queries,
                                      official_natural_single, OFFICIAL_SPEED)
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest
from eval_rival_fine_full import check_frozen, frozen_state, save


FULL_METHODS = (*METHODS, VIP)


@torch.inference_mode()
def input_banks(root, dataset, geometry, vip, old_b, old_q, checkpoints, protocol):
    b, q, inherited = task_text(root, dataset, geometry, vip, old_b, old_q, checkpoints)
    bank, query = b[dataset], q[dataset]
    path = root/'vocabularies'/(dataset+'.json')
    if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['candidate_sha256'][dataset]:
        raise RuntimeError('Frozen bank/family metadata changed.')
    meta = json.loads(path.read_text())
    names = tuple(row['name'] for row in meta['Complete20']['classes'])
    if names != bank.class_names: raise RuntimeError('Scored class order changed.')
    original = tuple(bank.alias_names)
    declared_current = tuple(a for row in meta['Current20']['classes'] for a in row['synonyms'])
    if declared_current != original: raise RuntimeError('Current20 strings/order differ.')
    aliases = tuple(a for row in meta['Complete20']['classes'] for a in row['synonyms'])
    if any(len(row['synonyms']) != 20 for row in meta['Complete20']['classes']):
        raise RuntimeError('Complete20 slot counts differ.')
    for i, (word, old) in enumerate(zip(aliases, original)):
        if bool(bank.canonical_mask[i]) and word != old:
            raise RuntimeError('Canonical anchor changed.')
    identity = dict(checkpoints=checkpoints, template=POLICIES[dataset].template,
                    aliases=aliases, inherited=inherited, metadata_sha256=protocol['candidate_sha256'][dataset])
    cp = root/'family_text_cache'/(dataset+'.pt')
    cached = torch.load(cp, map_location=geometry.device, weights_only=True) if cp.exists() else None
    if cached is not None and cached['identity'] != identity: raise RuntimeError('Text cache identity changed.')
    old_lookup = {word: i for i, word in enumerate(original)}
    new_words = sorted(set(aliases)-set(original))
    if cached is None:
        encoded = {}; seconds = 0.
        if new_words:
            module = _load_upstream(Path(vip.upstream_root)/'prompts/imagenet_template.py', 'family_lme_prompts')
            previous = vip.templates; vip.templates = module.get_text_template(POLICIES[dataset].template)
            started = time.perf_counter()
            try: new = vip.encode_queries(('offline expression strings',), (tuple(new_words),))
            finally: vip.templates = previous
            seconds = time.perf_counter()-started
            encoded = {word: new.features[i].cpu() for i, word in enumerate(new_words)}
        cached = dict(identity=identity, encoded=encoded, unique_new_queries=len(new_words), encoding_seconds=seconds)
        cp.parent.mkdir(exist_ok=True); temporary=cp.with_suffix('.tmp'); torch.save(cached, temporary); temporary.rename(cp)
    wide, local = query.features.clone(), bank.features.clone()
    for i, word in enumerate(aliases):
        if word in old_lookup:
            wide[i] = query.features[old_lookup[word]]; local[i] = bank.features[old_lookup[word]]
        else:
            wide[i] = cached['encoded'][word].to(geometry.device)
            local[i] = F.normalize(wide[i].float().mean(0), dim=-1)
    new_bank = TCPRTextBank(local, bank.parent_indices, bank.canonical_mask, bank.class_names, aliases)
    new_bank.validate()
    banks={'Current20':bank, 'Complete20':new_bank}
    queries={'Current20':query, 'Complete20':VIPQueries(wide, query.parents, query.class_names, aliases)}
    plans={name:prepare_plan(banks[name], meta[name]['family_ids']) for name in banks}
    if not plans['Complete20'].changed_classes: raise RuntimeError('No positive expression treatment formed.')
    return banks, queries, plans, identity, dict(unique_new_queries=cached['unique_new_queries'],
                                               encoding_seconds=cached['encoding_seconds'])


@torch.inference_mode()
def main(args):
    root, output = Path(args.suite_root), Path(args.output_dir)
    if output.exists(): raise RuntimeError('Existing output; preserve it.')
    device=torch.device(args.device); require_available_gpu(device); lease=torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase, **kw): save(output/'worker_status.json',dict(status='running',phase=phase,**kw))
    state('loading')
    setup=curated_setup if args.dataset=='ade150' else original_setup
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,checkpoints,reference,_=setup(args)
    if reference is not None: raise RuntimeError('Unexpected reference model.')
    for name, expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen method source differs: '+name)
    banks,queries,plans,text_identity,text_cost=input_banks(root,args.dataset,g,v,old_b,old_q,checkpoints,protocol)
    frozen=frozen_state(g,v)
    def call(name,image):
        if name==VIP: return {name:v.predict(image,queries['Complete20'],upstream_settings(args.dataset))[0]},{}
        if name==OFFICIAL_SPEED:
            query=official_q[args.dataset]
            pred=official_natural_single(image,v,query,official_settings) if args.dataset=='ade150' else v.predict(image,query,official_settings)[0]
            return {name:pred},{}
        return predict(image,g,v,banks,queries,plans,args.dataset,(name,))
    if args.mode=='benchmark':
        official_q,official_settings,official_identity=official_speed_queries(args.dataset,v,{args.dataset:banks['Complete20']})
        timing_names=(COMPLETE_BASE,PRIMARY,VIP,OFFICIAL_SPEED); images=[]; smoke=[]
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            image=loader(samples[index]); require_available_gpu(device); state('smoke',sample_key=samples[index].key)
            joint,diag=predict(image,g,v,banks,queries,plans,args.dataset)
            independent,_=task_predict(image,g,{args.dataset:banks['Current20']},v,{args.dataset:queries['Current20']},args.dataset)
            if not np.array_equal(joint[CURRENT_BASE],independent[args.dataset]['Fixed20_TaskCoupled']):
                raise RuntimeError('Independent original Base replay differs.')
            for name in METHODS:
                if not np.array_equal(call(name,image)[0][name],joint[name]):
                    raise RuntimeError('Single/joint prediction differs: '+name)
            smoke.append(dict(sample_key=samples[index].key,independent_base_equal=True,
                              all_singleton_joint_predictions_equal=True,diagnostics=diag))
            expected={name:call(name,image)[0][name] for name in timing_names}
            row={name:dict(seconds=[],peaks=[]) for name in timing_names}
            state('timing',sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order=timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]
                for name in order:
                    require_available_gpu(device); actual,t=measure(call,(name,image),device,1); require_available_gpu(device)
                    if not np.array_equal(actual[0][name],expected[name]): raise RuntimeError('Warm prediction changed.')
                    row[name]['seconds']+=t['seconds']; row[name]['peaks'].append(t['peak_allocated_mib'])
            for val in row.values(): val.update(median_seconds=statistics.median(val['seconds']),peak_allocated_mib=max(val.pop('peaks')))
            ratios={name:row[PRIMARY]['median_seconds']/row[name]['median_seconds'] for name in (VIP,OFFICIAL_SPEED)}
            images.append(dict(sample_key=samples[index].key,timings=row,ratios=ratios,below6x=max(ratios.values())<=6))
            save(output/'timing.json',dict(status='running',images=images,target_masks_loaded=False))
        passed=all(row['below6x'] for row in images)
        save(output/'smoke.json',dict(status='complete',images=smoke,text_cost=text_cost,target_masks_loaded=False,**check_frozen(frozen,g,v)))
        save(output/'timing.json',dict(status='complete',images=images,cost_gate_passed=passed,
            target_masks_loaded=False,official_speed_reference=official_identity,whole_image_scope=True,
            exclusive_gpu_verified=True,includes='resize, encoders, alias, Geometry writeback, restoration and argmax',
            excludes='decode, frozen model/text setup',memory_scope='both backbones, both text banks resident'))
        save(output/'worker_status.json',dict(status='complete' if passed else 'cost_rejected',phase='complete')); return
    for d in protocol['datasets']:
        if not json.loads((root/'benchmark'/d/'timing.json').read_text())['cost_gate_passed']:
            raise RuntimeError('Both cost gates required before full masks.')
    keys=[s.key for s in samples]; selected=samples[args.shard_index::args.num_shards]
    if len(keys)!=entry['total_images'] or len(keys)!=len(set(keys)) or not selected:
        raise RuntimeError('Unique full inventory/shard differs.')
    classes=banks['Current20'].class_names
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=FULL_METHODS,classes={args.dataset:classes},
        checkpoints=checkpoints,vocabulary=words,gear=dict(profile=asdict(POLICIES[args.dataset]),
        task_text_identity=text_identity,candidate_sha256=protocol['candidate_sha256'][args.dataset]),competitive=None,
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),sample_keys=[s.key for s in selected],
        sample_keys_sha256=digest([s.key for s in selected]),num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args))
    matrices={name:np.zeros((len(classes),len(classes)),np.int64) for name in FULL_METHODS}
    per_image={args.dataset+'__'+name:[] for name in FULL_METHODS}; totals={}; ignored=0
    changes={name:0 for name in (COMPLETE_BASE,SHUFFLE)}; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image=loader(sample); predictions,diag=predict(image,g,v,banks,queries,plans,args.dataset)
        predictions[VIP]=call(VIP,image)[0][VIP]
        target=masks(sample,args.dataset,tuple(image.shape[-2:])); valid=(target>=0)&(target<len(classes)); ignored+=int((~valid).sum())
        for name,pred in predictions.items():
            if pred.shape!=target.shape or np.any(pred[valid]>=len(classes)): raise RuntimeError('Invalid prediction shape/class.')
            cm=np.bincount(target[valid].astype(np.int64)*len(classes)+pred[valid],minlength=len(classes)**2).reshape(len(classes),len(classes))
            matrices[name]+=cm; per_image[args.dataset+'__'+name].append(cm)
        for name in changes: changes[name]+=int(((predictions[PRIMARY]!=predictions[name])&valid).sum())
        if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards'] or diag['additional_visual_forwards'] or diag['additional_semantic_heads']:
            raise RuntimeError('Visual budget exceeded.')
        for k,val in diag.items(): totals[k]=totals.get(k,0.)+val
        if number==1 or number%20==0 or number==len(selected):
            row=dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={args.dataset:{name:summary(cm,classes,ignored) for name,cm in matrices.items()}},
                diagnostics={args.dataset:{k:val/number for k,val in totals.items()}},primary_pixel_changes=changes,
                target_masks_used_only_after_prediction=True,wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row); state('full',processed=number,total=len(selected))
            print(json.dumps(dict(dataset=args.dataset,shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
                        **{name:np.stack(values) for name,values in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v)); save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('benchmark','full'),required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--sample-seed',type=int,default=20260923); p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--num-shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--repetitions',type=int,default=5); main(p.parse_args())
