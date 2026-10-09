"""Three frozen input banks with identical Geometry and coupled readout."""
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

from benchmark_natural_sense_fast import measure,require_available_gpu
from dinotool.fixed20_task_readout import POLICIES,predict_image as task_predict
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries,_load_upstream,upstream_settings
from eval_alias_full_comparison_r3 import setup as original_setup
from alias_curated_setup import setup as curated_setup
from eval_canonical_rival_trial import task_text,official_speed_queries,official_natural_single,OFFICIAL_SPEED
from eval_development_readout import observations,probabilities,wide_scores
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest
from eval_rival_fine_full import check_frozen,frozen_state,save


IMPLEMENTATION='geometry-fixed20-single-generation-coverage-input-v1-20261009'
BASE,PRIMARY,CONTROL='Existing20','Coverage20','SynonymOnly20'
METHODS=(BASE,PRIMARY,CONTROL)
VIP_BASE,VIP_COVERAGE='VIP_Existing20','VIP_Coverage20'
FULL_METHODS=(*METHODS,VIP_BASE,VIP_COVERAGE)


@torch.inference_mode()
def input_banks(root,dataset,geometry,vip,old_banks,old_queries,checkpoints,protocol):
    old_b,old_q,old_identity=task_text(root,dataset,geometry,vip,old_banks,old_queries,checkpoints)
    bank,query=old_b[dataset],old_q[dataset]
    groups={}
    for arm in METHODS:
        path=root/'vocabularies'/(dataset+'_'+arm+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['candidate_input_sha256'][dataset][arm]:
            raise RuntimeError('Frozen candidate input changed: '+arm)
        rows=json.loads(path.read_text())['classes']
        if tuple(r['name'] for r in rows)!=bank.class_names or any(len(r['synonyms'])!=20 for r in rows):
            raise RuntimeError('Frozen class order/count differs: '+arm)
        groups[arm]=tuple(tuple(r['synonyms']) for r in rows)
    originals=tuple(tuple(bank.alias_names[i] for i in torch.nonzero(bank.parent_indices==c).flatten().cpu().tolist())
        for c in range(bank.class_count))
    if groups[BASE]!=originals:raise RuntimeError('Existing20 differs from exact inherited strings.')
    for arm in METHODS:
        for c,words in enumerate(groups[arm]):
            ids=torch.nonzero(bank.parent_indices==c).flatten()
            canonical=int(torch.nonzero(bank.canonical_mask[ids]).flatten()[0])
            if words[canonical]!=originals[c][canonical]:raise RuntimeError('Canonical slot changed.')
    identity=dict(checkpoints=checkpoints,template=POLICIES[dataset].template,
        class_names=list(bank.class_names),groups={name:[list(g) for g in value] for name,value in groups.items()},
        inherited_task_identity=old_identity)
    path=root/'coverage_text_cache'/(dataset+'.pt')
    cache=torch.load(path,map_location=geometry.device,weights_only=True) if path.exists() else None
    if cache and cache['identity']!=identity:raise RuntimeError('Frozen coverage text cache identity differs.')
    new_words=sorted({word for name in (PRIMARY,CONTROL) for group,old_group in zip(groups[name],originals)
        for word,old in zip(group,old_group) if word!=old})
    if cache is None:
        encoded={}
        if new_words:
            templates=_load_upstream(Path(vip.upstream_root)/'prompts/imagenet_template.py','coverage_input_templates')
            previous=vip.templates;vip.templates=templates.get_text_template(POLICIES[dataset].template)
            started=time.perf_counter()
            try:new=vip.encode_queries(('offline replacement phrases',),(tuple(new_words),))
            finally:vip.templates=previous
            encoded={word:new.features[i].cpu() for i,word in enumerate(new_words)}
            encoding_seconds=time.perf_counter()-started
        else:encoding_seconds=0.
        cache=dict(identity=identity,encoded=encoded,unique_new_queries=len(new_words),encoding_seconds=encoding_seconds)
        path.parent.mkdir(exist_ok=True);temporary=path.with_suffix('.tmp');torch.save(cache,temporary);temporary.rename(path)
    output_banks={BASE:bank};output_queries={BASE:query}
    for arm in (PRIMARY,CONTROL):
        aliases=tuple(word for group in groups[arm] for word in group)
        wide=query.features.clone();local=bank.features.clone()
        for i,(word,old_word) in enumerate(zip(aliases,bank.alias_names)):
            if word!=old_word:
                wide[i]=cache['encoded'][word].to(geometry.device)
                local[i]=F.normalize(wide[i].float().mean(0),dim=-1)
        new_bank=TCPRTextBank(local,bank.parent_indices,bank.canonical_mask,bank.class_names,aliases);new_bank.validate()
        output_banks[arm]=new_bank
        output_queries[arm]=VIPQueries(wide,query.parents,query.class_names,aliases)
    return output_banks,output_queries,identity,dict(unique_new_queries=cache['unique_new_queries'],
        encoding_seconds=cache['encoding_seconds'])


@torch.inference_mode()
def predict(image,geometry,vip,banks,queries,dataset,methods=METHODS):
    policy=POLICIES[dataset]
    source=observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    result={}
    for arm in methods:
        broad=wide_scores(source,queries[arm],policy.tau,policy.tem)
        probability=probabilities(source,policy.profile(),banks[arm],broad,{})
        if not bool(torch.isfinite(probability).all()):raise RuntimeError('Nonfinite coverage-input probability.')
        result[arm]=probability.argmax(0).cpu().numpy()
    diagnostics=dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
        fine_forwards=0,additional_visual_forwards=0,additional_semantic_heads=0,alias_slots_per_class=20)
    return result,diagnostics


@torch.inference_mode()
def main(args):
    output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing worker output; preserve it.')
    device=torch.device(args.device);require_available_gpu(device);lease=torch.empty(1,device=device)
    output.mkdir(parents=True)
    def state(phase,**fields):save(output/'worker_status.json',dict(status='running',phase=phase,**fields))
    state('loading')
    setup=curated_setup if args.dataset=='ade150' else original_setup
    protocol,entry,samples,loader,masks,geometry,old_b,vip,old_q,words,checkpoints,reference,_=setup(args)
    if reference is not None:raise RuntimeError('Unexpected reference model.')
    for relative,expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen trial source changed: '+relative)
    root=Path(args.suite_root)
    banks,queries,text_identity,text_cost=input_banks(root,args.dataset,geometry,vip,old_b,old_q,checkpoints,protocol)
    frozen=frozen_state(geometry,vip)
    def call(name,image):
        if name in (VIP_BASE,VIP_COVERAGE):
            arm=BASE if name==VIP_BASE else PRIMARY
            return {name:vip.predict(image,queries[arm],upstream_settings(args.dataset))[0]},{}
        if name==OFFICIAL_SPEED:
            query=official_q[args.dataset]
            pred=official_natural_single(image,vip,query,official_settings) if args.dataset=='ade150' else vip.predict(image,query,official_settings)[0]
            return {name:pred},{}
        return predict(image,geometry,vip,banks,queries,args.dataset,(name,))
    if args.mode=='benchmark':
        official_q,official_settings,official_identity=official_speed_queries(args.dataset,vip,{args.dataset:banks[BASE]})
        names=(BASE,PRIMARY,VIP_COVERAGE,OFFICIAL_SPEED);images=[];smoke=[]
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            image=loader(samples[index]);require_available_gpu(device);state('smoke',sample_key=samples[index].key)
            joint,diag=predict(image,geometry,vip,banks,queries,args.dataset)
            independent,_=task_predict(image,geometry,{args.dataset:banks[BASE]},vip,{args.dataset:queries[BASE]},args.dataset)
            if not np.array_equal(joint[BASE],independent[args.dataset]['Fixed20_TaskCoupled']):
                raise RuntimeError('Independent Existing20 baseline replay differs.')
            for name in METHODS:
                single,_=call(name,image)
                if not np.array_equal(single[name],joint[name]):raise RuntimeError('Single/joint prediction differs: '+name)
            smoke.append(dict(sample_key=samples[index].key,independent_base_equal=True,
                all_singleton_joint_predictions_equal=True,diagnostics=diag))
            expected={name:call(name,image)[0][name] for name in names};row={name:dict(seconds=[],peaks=[]) for name in names}
            state('timing',sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in order:
                    require_available_gpu(device);actual,timing=measure(call,(name,image),device,1);require_available_gpu(device)
                    if not np.array_equal(actual[0][name],expected[name]):raise RuntimeError('Warmed prediction changed.')
                    row[name]['seconds']+=timing['seconds'];row[name]['peaks'].append(timing['peak_allocated_mib'])
            for values in row.values():values.update(median_seconds=statistics.median(values['seconds']),peak_allocated_mib=max(values.pop('peaks')))
            ratios={name:row[PRIMARY]['median_seconds']/row[name]['median_seconds'] for name in (VIP_COVERAGE,OFFICIAL_SPEED)}
            images.append(dict(sample_key=samples[index].key,image_size=list(image.shape[-2:]),timings=row,ratios=ratios,below6x=max(ratios.values())<=6))
            save(output/'timing.json',dict(status='running',images=images,target_masks_loaded=False))
        passed=all(row['below6x'] for row in images)
        save(output/'smoke.json',dict(status='complete',images=smoke,text_encoding_cost=text_cost,target_masks_loaded=False,**check_frozen(frozen,geometry,vip)))
        save(output/'timing.json',dict(status='complete',images=images,cost_gate_passed=passed,
            target_masks_loaded=False,official_speed_reference=official_identity,exclusive_gpu_verified=True,
            whole_image_scope=True,excludes='decode, model/text setup and offline Qwen generation',
            includes='resize, bounded encoders, unchanged alias aggregation, H, restoration/argmax',
            memory_scope='both models and three text banks resident; not isolated deployment memory'))
        save(output/'worker_status.json',dict(status='complete' if passed else 'cost_rejected',phase='complete'));return
    for d in protocol['datasets']:
        timing=json.loads((root/'benchmark'/d/'timing.json').read_text())
        if not timing['cost_gate_passed']:raise RuntimeError('Full scoring forbidden before both cost gates.')
    keys=[s.key for s in samples];selected=samples[args.shard_index::args.num_shards]
    if len(keys)!=entry['total_images'] or len(keys)!=len(set(keys)) or not selected or not 0<=args.shard_index<args.num_shards:
        raise RuntimeError('Full unique inventory/shard differs.')
    classes=banks[BASE].class_names
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=FULL_METHODS,
        classes={args.dataset:classes},checkpoints=checkpoints,vocabulary=words,
        gear=dict(profile=asdict(POLICIES[args.dataset]),sources=protocol['method_sources'],
            task_text_identity=text_identity,candidate_inputs=protocol['candidate_input_sha256'][args.dataset]),
        competitive=None,global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected],sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args))
    matrices={name:np.zeros((len(classes),len(classes)),np.int64) for name in FULL_METHODS}
    per_image={args.dataset+'__'+name:[] for name in FULL_METHODS};ignored=0;totals={}
    changed={name:0 for name in (BASE,CONTROL)};started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image=loader(sample);predictions,diag=predict(image,geometry,vip,banks,queries,args.dataset)
        for name in (VIP_BASE,VIP_COVERAGE):predictions[name]=call(name,image)[0][name]
        target=masks(sample,args.dataset,tuple(image.shape[-2:]));valid=(target>=0)&(target<len(classes));ignored+=int((~valid).sum())
        for name,pred in predictions.items():
            if pred.shape!=target.shape or np.any(pred[valid]>=len(classes)):raise RuntimeError('Invalid prediction classes/shape.')
            cm=np.bincount(target[valid].astype(np.int64)*len(classes)+pred[valid],minlength=len(classes)**2).reshape(len(classes),len(classes))
            matrices[name]+=cm;per_image[args.dataset+'__'+name].append(cm)
        for name in changed:changed[name]+=int(((predictions[PRIMARY]!=predictions[name])&valid).sum())
        if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards'] or diag['additional_visual_forwards'] or diag['additional_semantic_heads']:
            raise RuntimeError('Frozen candidate visual budget exceeded.')
        for k,value in diag.items():totals[k]=totals.get(k,0.)+value
        if number==1 or number%20==0 or number==len(selected):
            row=dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={args.dataset:{name:summary(cm,classes,ignored) for name,cm in matrices.items()}},
                diagnostics={args.dataset:{k:value/number for k,value in totals.items()}},
                primary_pixel_changes=changed,target_masks_used_only_after_prediction=True,
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row);state('full',processed=number,total=len(selected))
            print(json.dumps(dict(dataset=args.dataset,shard=args.shard_index,processed=number,total=len(selected))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{name:np.stack(values) for name,values in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,geometry,vip));save(output/'results.json',row)
    save(output/'worker_status.json',dict(status='complete',phase='complete',processed=len(selected),total=len(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('benchmark','full'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--repetitions',type=int,default=5);main(p.parse_args())
