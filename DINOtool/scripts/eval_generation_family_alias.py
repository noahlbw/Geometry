"""Frozen ADE input-source/family diagnostic, with a pre-mask cost gate."""
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
from dinotool.generation_family_alias import (IMPLEMENTATION, CC, HH, PRIMARY, SHUFFLE,
    MEAN, METHODS, prepare_plan, predict_image)
from dinotool.fixed20_task_readout import POLICIES, VIP, predict_image as task_predict
from dinotool.packed_alias_readout import PackedAliasReadout
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, upstream_settings
from alias_curated_setup import setup
from eval_canonical_rival_trial import (task_text, official_speed_queries,
    official_natural_single, OFFICIAL_SPEED)
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save


def load_historical(protocol,identity,device,curated_bank):
    source = Path(protocol['historical_source'])
    for relative,expected in protocol['historical_files'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen historical source changed: '+relative)
    old = json.loads((source/'protocol.json').read_text())['datasets']['ade150']
    cache = torch.load(source/'text_cache/ade150.pt',map_location=device,weights_only=True)
    if cache['identity']['banks']!=old['banks'] or cache['identity']['checkpoints']!=identity:
        raise RuntimeError('Historical word/cache/checkpoint identity differs.')
    classes = old['banks']['semantic_segmentation']['classes']
    names = tuple(c['name'] for c in classes)
    aliases = tuple(a for c in classes for a in c['synonyms'])
    row = cache['encoded']['semantic_segmentation']
    bank = TCPRTextBank(row['local'],row['parents'],row['canonical'],names,aliases)
    bank.validate()
    if names!=curated_bank.class_names or len(aliases)!=433:
        raise RuntimeError('Historical scored classes or query count changed.')
    query = VIPQueries(row['wide'],row['parents'],names,aliases)
    return bank,query


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing diagnostic output; preserve it.')
    device = torch.device(args.device)
    if args.mode=='benchmark':require_available_gpu(device)
    lease = torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**values):save(output/'worker_status.json',dict(status='running',phase=phase,**values))
    state('loading')
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,identity,_,_ = setup(args)
    for relative,expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen trial source changed: '+relative)
    b,q,text_identity = task_text(Path(args.suite_root),args.dataset,g,v,old_b,old_q,identity)
    cb,cq = b[args.dataset],q[args.dataset]
    provenance = protocol['generation_provenance']
    if (list(cb.class_names)!=provenance['class_names'] or list(cb.alias_names)!=provenance['aliases']):
        raise RuntimeError('Generation provenance does not match exact curated strings/order.')
    hb,hq = load_historical(protocol,identity,device,cb)
    banks,queries = {'C':cb,'H':hb},{'C':cq,'H':hq}
    plan = prepare_plan(cb,provenance['family_ids'])
    reader = PackedAliasReadout(hb)
    historical = TaxonomyInference.for_task(g,v,{'semantic_segmentation':hb},
        {'semantic_segmentation':hq},family='natural',background=None,local_background='union_max')
    if (historical.soft or historical.profile.strength!='original' or historical.profile.coupling!=.5
            or historical.profile.temperature!=.07):
        raise RuntimeError('Historical executed profile changed.')
    frozen = frozen_state(g,v)
    def call(name,image):
        if name==VIP:return {VIP:v.predict(image,cq,upstream_settings(args.dataset))[0]},{}
        if name==OFFICIAL_SPEED:
            return {name:official_natural_single(image,v,official_q[args.dataset],official_settings)},{}
        return predict_image(image,g,banks,v,queries,plan,reader,methods=(name,))
    if args.mode=='benchmark':
        official_q,official_settings,official_identity = official_speed_queries(args.dataset,v,b)
        timing_names = (CC,PRIMARY,VIP,OFFICIAL_SPEED)
        images,smoke = [],[]
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            image = loader(samples[index]);require_available_gpu(device)
            joint,diag = predict_image(image,g,banks,v,queries,plan,reader)
            independent,_ = task_predict(image,g,b,v,q,args.dataset)
            hp,hd = historical.predict(image)
            if (not np.array_equal(joint[CC],independent[args.dataset]['Fixed20_TaskCoupled'])
                    or not np.array_equal(joint[HH],hp)):
                raise RuntimeError('CC or historical packed HH independent replay differs.')
            for name in METHODS:
                single,_ = call(name,image)
                if not np.array_equal(single[name],joint[name]):
                    raise RuntimeError('Singleton/joint mismatch: '+name)
            smoke.append(dict(sample_key=samples[index].key,exact_cc_replay=True,
                exact_historical_packed_hh_replay=True,all_singleton_joint_predictions_equal=True,
                historical_diagnostic=hd,diagnostics=diag))
            expected = {name:call(name,image)[0] for name in timing_names}
            row = {name:dict(seconds=[],peaks=[]) for name in timing_names}
            for repeat in range(args.repetitions):
                order = timing_names[repeat%len(timing_names):]+timing_names[:repeat%len(timing_names)]
                for name in order:
                    require_available_gpu(device)
                    actual,timing = measure(call,(name,image),device,1)
                    require_available_gpu(device)
                    if not np.array_equal(actual[0][name],expected[name][name]):
                        raise RuntimeError('Warmed singleton prediction changed.')
                    row[name]['seconds'] += timing['seconds'];row[name]['peaks'].append(timing['peak_allocated_mib'])
            for value in row.values():
                value.update(median_seconds=statistics.median(value['seconds']),
                             peak_allocated_mib=max(value.pop('peaks')))
            ratios = {name:row[PRIMARY]['median_seconds']/row[name]['median_seconds']
                      for name in (VIP,OFFICIAL_SPEED)}
            images.append(dict(sample_key=samples[index].key,image_size=list(image.shape[-2:]),
                timings=row,ratios=ratios,below6x=max(ratios.values())<=6))
            save(output/'timing.json',dict(status='running',images=images,target_masks_loaded=False))
        passed = all(row['below6x'] for row in images)
        save(output/'smoke.json',dict(status='complete',images=smoke,target_masks_loaded=False,
                                     **check_frozen(frozen,g,v)))
        save(output/'timing.json',dict(status='complete',images=images,cost_gate_passed=passed,
            exclusive_gpu_verified=True,whole_image_scope=True,target_masks_loaded=False,
            official_speed_reference=official_identity,excludes='decode, model/text/plan setup',
            includes='resize, bounded encoders, C-only alias projection/Q, original H, full-size restoration/argmax',
            memory_scope='C/H text caches and both backbones resident; not isolated deployment memory'))
        state('complete',status_override='cost gate passed' if passed else 'cost rejected')
        save(output/'worker_status.json',dict(status='complete' if passed else 'cost_rejected',phase='complete'))
        return
    timing = json.loads((Path(args.suite_root)/'benchmark/ade150/timing.json').read_text())
    if not timing.get('cost_gate_passed'):raise RuntimeError('Full masks forbidden before singleton cost gate.')
    if not 0<=args.shard_index<args.num_shards:raise RuntimeError('Invalid shard.')
    selected = samples[args.shard_index::args.num_shards]
    keys = [sample.key for sample in samples]
    if len(keys)!=2000 or len(set(keys))!=2000 or not selected:raise RuntimeError('Full unique inventory changed.')
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,
        classes={args.dataset:cb.class_names},checkpoints=identity,
        vocabulary=dict(C=words,H=dict(aliases=hb.alias_names,counts=[int((hb.parent_indices==c).sum()) for c in range(hb.class_count)])),
        gear=dict(profile=asdict(POLICIES['ade150']),sources=protocol['method_sources'],task_text_identity=text_identity,
                  historical_files=protocol['historical_files'],generation_provenance=provenance),
        competitive=None,global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected],sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args))
    matrices = {name:np.zeros((cb.class_count,)*2,np.int64) for name in METHODS}
    per_image = {args.dataset+'__'+name:[] for name in METHODS};ignored=0;totals={}
    changed = {name:0 for name in (CC,SHUFFLE,MEAN)}
    started = time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image = loader(sample)
        predictions,diag = predict_image(image,g,banks,v,queries,plan,reader)
        target = masks(sample,args.dataset,tuple(image.shape[-2:]))
        valid = (target>=0)&(target<cb.class_count);ignored += int((~valid).sum())
        for name,pred in predictions.items():
            if pred.shape!=target.shape or np.any(pred[valid]>=cb.class_count):
                raise RuntimeError('Invalid full-size prediction/class indices.')
            cm = np.bincount(target[valid].astype(np.int64)*cb.class_count+pred[valid],
                minlength=cb.class_count**2).reshape(cb.class_count,cb.class_count)
            matrices[name] += cm;per_image[args.dataset+'__'+name].append(cm)
        for name in changed:changed[name] += int(((predictions[PRIMARY]!=predictions[name])&valid).sum())
        if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards'] or diag['additional_visual_forwards']:
            raise RuntimeError('Frozen visual budget exceeded.')
        for key,value in diag.items():totals[key] = totals.get(key,0.)+value
        if number==1 or number%20==0 or number==len(selected):
            row = dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={args.dataset:{name:summary(cm,cb.class_names,ignored) for name,cm in matrices.items()}},
                diagnostics={args.dataset:{k:x/number for k,x in totals.items()}},primary_pixel_changes=changed,
                target_masks_used_only_after_prediction=True,wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row);state('full',processed=number,total=len(selected))
            print(json.dumps(dict(shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{key:np.stack(value) for key,value in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,g,v));save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('benchmark','full'),required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--sample-seed',type=int,default=20260923)
    parser.add_argument('--vdd-ontology',default='official');parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--num-shards',type=int,default=1);parser.add_argument('--shard-index',type=int,default=0)
    main(parser.parse_args())
