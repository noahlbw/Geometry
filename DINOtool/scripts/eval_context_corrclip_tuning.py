"""PC59/60: labeled finite alias/scalar development, frozen efficient full inference."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import eval_curated_variable_natural as core
from eval_alias_finalization import load_samples
from eval_natural_static_search import move
from eval_rival_fine_full import save, frozen_state, check_frozen
from eval_geometry_vip_reliability import summary
from dinotool.natural_static_search import StaticProfile, StaticReader
import dinotool.natural_static_search as static
from dinotool.efficient_deployment import EfficientGeometry, EfficientVIPObserver, GroupedWideScorer, GroupedLocalScorer, reconstruction, write_dense
from dinotool.compact_deployment import compact_observations, offload_text_towers
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries

TOOL=Path(__file__).resolve().parents[1]
BASE=TOOL.parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def options(stage,selected,entry,menu):
    candidates=[selected]
    if stage=='words':candidates += [replace(selected,bank=k) for k in entry['search_banks']]
    elif stage=='readout':candidates += [replace(selected,strength=s,coupling=g) for s in menu['strengths'] for g in menu['coupling']]
    elif stage=='wide':candidates += [replace(selected,tau=t,tem=m) for t in menu['tau'] for m in menu['tem']]
    elif stage=='temperature':candidates += [replace(selected,temperature=t) for t in menu['temperature']]
    return tuple(dict.fromkeys(candidates))


def search(args,root,entry,output):
    protocol=read(root/'protocol.json');menu=protocol['search'];bg=entry['background_index']
    needed=tuple(entry['search_banks'])+('candidate_pool_segmentation','candidate_pool_imagenet')
    geometry,vip,banks,queries,checkpoints,residual=core.previous.load_models(args,entry,needed)
    # Keep the current best's exact historical text vectors as the incumbent.
    packed=torch.load(TOOL/'results/efficient_geometry_20261012/text_cache'/f'{args.dataset}.pt',map_location='cpu',weights_only=True)
    incumbent_name=entry['incumbent_config']['profile']['bank'];row=packed['encoded'][incumbent_name]
    old=banks['incumbent'];parents=row['parents'].to(args.device)
    banks['incumbent']=TCPRTextBank(row['local'].to(args.device),parents,row['canonical'].to(args.device),old.class_names,old.alias_names)
    queries['incumbent']=VIPQueries(row['wide'].to(args.device),parents,old.class_names,old.alias_names)
    del packed,row,old
    frozen=frozen_state(geometry,vip);offload_text_towers(geometry,vip,release=True)
    geometry=EfficientGeometry(geometry);vip.__class__=EfficientVIPObserver
    validation,_,loader,target,names=load_samples(args,entry);names=names[args.dataset]
    keys=core.expanded_keys(validation,entry);lookup={s.key:s for s in validation}
    source_dir=output/'sources';source_dir.mkdir();memory={};records={};passes=[];counter=0;started=time.perf_counter()
    save(output/'development_keys.json',dict(keys=keys,previous_keys=entry['development_keys']))
    def observation(key):
        if key in memory:return move(memory[key],args.device)
        path=source_dir/(hashlib.sha256(key.encode()).hexdigest()+'.pt')
        if path.exists():source=torch.load(path,map_location='cpu',weights_only=True)
        else:
            source=compact_observations(loader(lookup[key]),geometry,vip,tuple(menu['strengths']),
                wide_policy=entry['incumbent_profile']['wide_policy'],reconstruction=reconstruction)
            source=move(source,'cpu');torch.save(source,path)
        if len(memory)<entry['search_count']:memory[key]=source
        return move(source,args.device)
    def register(groups,template):
        key=core.bank_key(groups,template)
        record=dict(template=template,classes=[dict(name=n,synonyms=g) for n,g in zip(names,groups)])
        records[key]=record;core.add_bank(key,record,entry,banks,queries)
        return key
    static.BIASES=tuple(menu['biases']);static.THRESHOLDS=tuple(menu['thresholds'])
    def evaluate(profiles,sample_keys,stage):
        nonlocal counter
        counter+=1;profiles=tuple(dict.fromkeys(profiles));chosen={p.bank for p in profiles}
        reader=StaticReader({k:banks[k] for k in chosen},{k:queries[k] for k in chosen},bg,residual)
        scorers={k:GroupedLocalScorer(banks[k]) for k in chosen};wide=GroupedWideScorer()
        biases=static.BIASES if bg is not None else (0.,)
        hist=np.zeros((len(profiles),len(biases),len(static.THRESHOLDS)+1,len(names),len(names)),np.int64)
        for number,key in enumerate(sample_keys,1):
            source=observation(key);label=torch.as_tensor(target(lookup[key],args.dataset,source['output_size']),device=args.device)
            cache={}
            for index,profile in enumerate(profiles):
                probability=reader.probabilities(source,profile,cache,wide_scorer=wide,dense_writer=write_dense,local_scorers=scorers)
                for j,bias in enumerate(biases):
                    hist[index,j]+=core.base.threshold_histogram(probability,label,bg,bias,static.THRESHOLDS)
            del source,label,probability,cache
            if number==1 or number%8==0 or number==len(sample_keys):
                progress=dict(stage=stage,processed_images=number,total_images=len(sample_keys),profiles=len(profiles),
                    passes=counter,wall_seconds=time.perf_counter()-started)
                save(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        ranked=static.rank(hist,profiles,bg);winners=[]
        for row in ranked:
            profile=StaticProfile(**row['profile'])
            if profile not in winners:winners.append(profile)
            if len(winners)==3:break
        record=dict(stage=stage,sample_keys=sample_keys,profiles=[p.record() for p in profiles],best=copy.deepcopy(ranked[0]),
            ranked=[{k:v for k,v in r.items() if k!='confusion_matrix'} for r in ranked])
        save(output/f'pass_{counter:03d}.json',record);passes.append(record)
        return ranked,winners
    selected=StaticProfile(**entry['incumbent_profile']);drafts=[selected]
    for stage in ('words','readout','wide','temperature','words'):
        _,winners=evaluate(options(stage,selected,entry,menu),keys[:entry['coarse_count']],stage)
        ranked,winners=evaluate([selected,*winners],keys[:entry['search_count']],stage+'_confirm')
        selected=winners[0];drafts.extend(winners)
    record=records.get(selected.bank,entry['banks'].get(selected.bank))
    groups=[list(c['synonyms']) for c in record['classes']];template=record['template']
    cm=np.asarray(ranked[0]['confusion_matrix']);union=cm.sum(0)+cm.sum(1)-cm.diagonal()
    order=np.argsort(np.divide(cm.diagonal(),union,out=np.zeros(len(names),float),where=union>0))
    accepted=0
    for index in order:
        index=int(index)
        if index==bg:continue
        proposals=[]
        for option in core.class_options(entry,groups,index)[:12]:
            changed=copy.deepcopy(groups);changed[index]=option
            proposals.append(replace(selected,bank=register(changed,template)))
        _,short=evaluate([selected,*proposals],keys[:entry['coarse_count']],'class_'+names[index])
        ranked,short=evaluate([selected,*short[:2]],keys[:entry['search_count']],'class_'+names[index]+'_confirm')
        baseline=max(r['development_miou'] for r in ranked if r['profile']==selected.record())
        if ranked[0]['development_miou']>baseline+1e-6:
            selected=short[0];groups=[list(c['synonyms']) for c in records.get(selected.bank,entry['banks'].get(selected.bank))['classes']]
            accepted+=1
            print(json.dumps(dict(accepted_class=names[index],miou=ranked[0]['development_miou'],aliases=groups[index])),flush=True)
            if accepted%10==0:drafts.append(selected)
        keep={p.bank for p in [*drafts,selected]}
        for name in list(banks):
            if name.startswith('mix_') and name not in keep:del banks[name],queries[name]
    drafts.append(selected)
    for stage in ('readout','wide','temperature'):
        _,winners=evaluate(options(stage,selected,entry,menu),keys[:entry['coarse_count']],'mixed_'+stage)
        ranked,winners=evaluate([selected,*winners],keys[:entry['search_count']],'mixed_'+stage+'_confirm')
        selected=winners[0];drafts.extend(winners)
    for profile in drafts:
        if profile.bank not in banks:core.add_bank(profile.bank,records[profile.bank],entry,banks,queries)
    ranked,winners=evaluate(drafts,keys,'final_confirmation');choice=copy.deepcopy(ranked[0]);profile=winners[0]
    record=records.get(profile.bank,entry['banks'].get(profile.bank))
    choice.update(status='complete',implementation=protocol['implementation'],development_keys=keys,
        development_confusion=choice['confusion_matrix'],vocabulary={profile.bank:record},
        alias_counts=[len(c['synonyms']) for c in record['classes']],target_label_development=True,
        inference_uses_target_labels=False,accepted_class_swaps=accepted,search_passes=counter)
    config=copy.deepcopy(entry['incumbent_config']);config.update(profile=profile.record(),
        background_bias=choice['background_bias'],background_threshold=choice['background_threshold'],
        vocabulary='../vocabularies/'+args.dataset+'.json',
        provenance=dict(source_trial=root.name,labeled_development=True,full_validation_viewed=True,allowed_changes='aliases and scalars'))
    vocabulary=dict(dataset=args.dataset,banks={profile.bank:record},residual=entry.get('residual_identity'))
    save(root/'configs'/f'{args.dataset}.yaml',config);save(root/'vocabularies'/f'{args.dataset}.json',vocabulary)
    bank,query=banks[profile.bank],queries[profile.bank]
    text=dict(identity=dict(vocabulary=vocabulary,checkpoints=core.base.checkpoint_manifest(checkpoints),
        vip_commit=config['upstream']['vip_commit']),encoded={profile.bank:dict(local=bank.features.cpu(),wide=query.features.cpu(),
        parents=bank.parent_indices.cpu(),canonical=bank.canonical_mask.cpu())})
    if residual is not None:text['residual']=residual.cpu()
    cache=root/'deployment_text_cache'/f'{args.dataset}.pt';cache.parent.mkdir(exist_ok=True);torch.save(text,cache)
    save(output/'selection.json',choice)
    return dict(status='complete',processed_images=len(keys),development_miou=choice['development_miou'],
        wall_seconds=time.perf_counter()-started,**check_frozen(frozen,geometry,vip))


def deployed(args,root,entry,output):
    model=FrozenGeometry(root/'configs'/f'{args.dataset}.yaml',checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo,upstream_root=args.upstream_root,
        text_cache=root/'deployment_text_cache'/f'{args.dataset}.pt',efficient=True)
    state=frozen_state(model.geometry,model.vip)
    validation,samples,loader,target,names=load_samples(args,entry);names=names[args.dataset]
    if len(validation)!=5105:raise RuntimeError('Unexpected Context coverage.')
    choice=read(root/'search'/args.dataset/'selection.json')
    if args.mode=='cost':
        lookup={s.key:s for s in validation};rows=[]
        for key in read(root/'cost_keys.json')[args.dataset]:
            image=loader(lookup[key]);model.predict(image);model.predict(image);torch.cuda.synchronize()
            times=[];peaks=[]
            for _ in range(7):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                actual=model.predict(image);torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-started));peaks.append(torch.cuda.max_memory_allocated()/1048576)
                if actual.shape!=tuple(image.shape[-2:]):raise RuntimeError('Invalid prediction shape.')
            rows.append(dict(key=key,shape=list(image.shape[-2:]),milliseconds=times,peak_allocated_mib=max(peaks)))
        times=[t for row in rows for t in row['milliseconds']]
        return dict(status='complete',dataset=args.dataset,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in rows),images=rows,choice=choice,
            target_masks_loaded=False,standalone_process=True,**check_frozen(state,model.geometry,model.vip))
    matrix=np.zeros((len(names),len(names)),np.int64);arrays=[];keys=[];started=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=loader(sample);actual=model.predict(image)
        cm=core.base.confusion(actual,target(sample,args.dataset,tuple(image.shape[-2:])),len(names))
        matrix+=cm;arrays.append(cm);keys.append(sample.key)
        if number%25==0:
            progress=dict(stage='full',processed_images=number,total_images=len(samples),wall_seconds=time.perf_counter()-started)
            save(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',keys=np.asarray(keys),confusions=np.stack(arrays))
    return dict(status='complete',dataset=args.dataset,processed_images=len(keys),total_images=5105,
        sample_keys=keys,global_sample_keys=[s.key for s in validation],num_shards=args.num_shards,shard_index=args.shard_index,
        metrics=summary(matrix,names,0),choice=choice,config=model.config,cache_identity=model.cache_identity,
        wall_seconds=time.perf_counter()-started,**check_frozen(state,model.geometry,model.vip))


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);entry=read(root/'protocol.json')['datasets'][args.dataset]
    output=root/args.mode/args.dataset
    if args.mode=='full':output=output/f's{args.shard_index}'
    if output.exists():raise RuntimeError('Existing worker output is preserved.')
    output.mkdir(parents=True);args.data_root=entry['data_root']
    result=search(args,root,entry,output) if args.mode=='search' else deployed(args,root,entry,output)
    save(output/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True);p.add_argument('--dataset',required=True,choices=('context59','context60'))
    p.add_argument('--mode',required=True,choices=('search','full','cost'))
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'));p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    p.add_argument('--device',default='cuda');p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    args=p.parse_args();args.data_root='';args.original_cache=str(TOOL/'results/curated20_patchonly2_full_20261006/text_cache'/f'{args.dataset}.pt')
    main(args)
