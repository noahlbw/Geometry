"""Direct variable-size class vocabularies, frozen visual model, finite development."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import eval_kev_alias_search as previous
import eval_development_readout as base
from eval_alias_finalization import load_samples
from eval_natural_static_search import move
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save, frozen_state, check_frozen
from dinotool.compact_deployment import CompactGeometry, compact_observations, offload_text_towers, CachedWideScorer
from dinotool.natural_wide_resolution import replace_wide
from dinotool.natural_static_search import StaticProfile, StaticReader, stage_profiles, rank, BIASES, THRESHOLDS
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries

IMPLEMENTATION='geometry-curated-variable-natural-v1-20261010'
METHODS=('CuratedTuned','PreviousFinal')


def bank_key(groups,template):
    raw=json.dumps([groups,template],ensure_ascii=False,separators=(',',':'))
    return 'mix_'+hashlib.sha256(raw.encode()).hexdigest()[:20]


def add_bank(name,record,entry,banks,queries):
    """Exact subsets of the encoded union; preserve each group's first anchor."""
    if name in banks:return
    template=record['template'];pool='candidate_pool_'+('imagenet' if template=='ImageNet80' else 'segmentation')
    b,q=banks[pool],queries[pool]
    lookup={(int(p),a):i for i,(p,a) in enumerate(zip(b.parent_indices.cpu().tolist(),b.alias_names))}
    groups=[c['synonyms'] for c in record['classes']]
    indices=torch.tensor([lookup[(c,a)] for c,g in enumerate(groups) for a in g],device=b.features.device)
    parents=torch.tensor([c for c,g in enumerate(groups) for _ in g],device=b.features.device)
    canonical=torch.zeros(len(indices),device=b.features.device,dtype=torch.bool)
    canonical[torch.tensor(np.cumsum([0]+[len(g) for g in groups[:-1]]),device=b.features.device)]=True
    aliases=tuple(a for g in groups for a in g)
    banks[name]=TCPRTextBank(b.features[indices],parents,canonical,b.class_names,aliases)
    queries[name]=VIPQueries(q.features[indices],parents,b.class_names,aliases)
    banks[name].validate()


def class_options(entry,groups,index):
    """Group swaps plus literal single additions/removals; no forced quota."""
    options=[groups[index]]
    for name in entry['coordinate_banks']:
        options.append(entry['banks'][name]['classes'][index]['synonyms'])
    literal=entry['banks']['literal_segmentation']['classes'][index]['synonyms']
    for word in literal:
        if word not in groups[index]:options.append(groups[index]+[word])
    if len(groups[index])<=12:
        # Keep the natural-language anchor, even when the annotation id is opaque.
        options.extend(groups[index][:j]+groups[index][j+1:] for j in range(1,len(groups[index])))
    unique=[]
    for group in options:
        group=list(dict.fromkeys(group))
        if group and group not in unique:unique.append(group)
    return unique


def expanded_keys(validation,entry):
    known=list(entry['development_keys'])
    rest=sorted((s.key for s in validation if s.key not in set(known)),
        key=lambda k:hashlib.sha256(('curated-variable-20261010:'+k).encode()).hexdigest())
    return known+rest[:entry['confirmation_count']-len(known)]


def search(args,protocol,entry,models,loaders,output):
    geometry,vip,banks,queries,checkpoints,residual=models
    validation,samples,loader,target,names=loaders;n=names[args.dataset];bg=entry['background_index']
    lookup={s.key:s for s in validation};all_keys=expanded_keys(validation,entry)
    save(output/'development_keys.json',dict(keys=all_keys,prior_keys=entry['development_keys']))
    source_folder=output/'sources';source_folder.mkdir()
    started=time.perf_counter();history=[];drafts=[];records={};evaluation=0
    # CPU cache is bounded; additional confirmation observations are read from disk.
    memory={}
    def observation(key):
        if key in memory:return move(memory[key],args.device)
        path=source_folder/(hashlib.sha256(key.encode()).hexdigest()+'.pt')
        if path.exists():source=torch.load(path,map_location='cpu',weights_only=True)
        else:
            image=loader(lookup[key])
            source=compact_observations(image,geometry,vip,('original',.5,1.,2.,3.,4.),wide_policy='long448')
            source={'long448':source,'natural_short336_cap672':replace_wide(source,image,vip)}
            source=move(source,'cpu');torch.save(source,path)
        if len(memory)<entry['search_count']:memory[key]=source
        return move(source,args.device)
    def register(groups,template):
        key=bank_key(groups,template)
        record=dict(classes=[dict(name=c,synonyms=list(g)) for c,g in zip(n,groups)],template=template)
        records[key]=record
        add_bank(key,record,entry,banks,queries)
        return key
    def evaluate(profiles,keys,stage):
        nonlocal evaluation
        profiles=tuple(dict.fromkeys(profiles));evaluation+=1
        selected_names=set(p.bank for p in profiles)
        reader=StaticReader({k:banks[k] for k in selected_names},{k:queries[k] for k in selected_names},bg,residual)
        scorer=CachedWideScorer();biases=BIASES if bg is not None else (0.,)
        hist=np.zeros((len(profiles),len(biases),len(THRESHOLDS)+1,len(n),len(n)),np.int64)
        for number,key in enumerate(keys,1):
            source=observation(key);label=torch.as_tensor(target(lookup[key],args.dataset,source['long448']['output_size']),device=args.device)
            cache={p:{} for p in source}
            for index,profile in enumerate(profiles):
                probability=reader.probabilities(source[profile.wide_policy],profile,cache[profile.wide_policy],wide_scorer=scorer)
                for j,bias in enumerate(biases):hist[index,j]+=base.threshold_histogram(probability,label,bg,bias,THRESHOLDS)
            del source,label,cache,probability
            if number==1 or number%8==0 or number==len(keys):
                save(output/'results.json',dict(status='running',stage=stage,processed_images=number,total_images=len(keys),
                    profiles=len(profiles),evaluations=evaluation,wall_seconds=time.perf_counter()-started))
                print(json.dumps(dict(stage=stage,images=number,total=len(keys),profiles=len(profiles))),flush=True)
        ranked=rank(hist,profiles,bg)
        record=dict(stage=stage,sample_keys=keys,profiles=[p.record() for p in profiles],best=copy.deepcopy(ranked[0]),
            ranked=[{k:v for k,v in r.items() if k!='confusion_matrix'} for r in ranked])
        save(output/('pass_'+str(evaluation).zfill(3)+'.json'),record);history.append(record)
        winners=[]
        for row in ranked:
            p=StaticProfile(**row['profile'])
            if p not in winners:winners.append(p)
            if len(winners)==3:break
        return ranked,winners
    selected=StaticProfile(**entry['incumbent_profile']);drafts.append(selected)
    retained=entry['retained_choice']['profile']
    if retained['bank']!=previous.PREVIOUS:drafts.append(StaticProfile(**retained))
    # Search from all strong historical seeds; failed Qwen20 is not the only pool.
    for stage in ('words','readout','calibration','revisit_words'):
        candidates=stage_profiles(stage,selected,entry['search_banks'],bg,residual is not None)
        ranked,winners=evaluate(candidates,all_keys[:entry['coarse_count']],stage)
        selected=winners[0];drafts.extend(winners)
    ranked,winners=evaluate(drafts,all_keys[:entry['search_count']],'seed_confirmation')
    selected=winners[0];confirmed=[selected,*winners]
    # Replace one semantic class at a time; score every competitor, not just its IoU.
    for sweep in range(2):
        current_record=records.get(selected.bank,entry['banks'].get(selected.bank))
        groups=[list(c['synonyms']) for c in current_record['classes']]
        template=current_record['template']
        cm=np.asarray(ranked[0]['confusion_matrix']);tp=cm.diagonal();union=cm.sum(0)+cm.sum(1)-tp
        order=np.argsort(np.divide(tp,union,out=np.zeros(len(n),float),where=union>0))
        accepted=0
        for index in order:
            index=int(index)
            if index==bg:continue
            candidates=[]
            for option in class_options(entry,groups,index):
                candidate=copy.deepcopy(groups);candidate[index]=option
                candidates.append(replace(selected,bank=register(candidate,template)))
            if len(candidates)==1:continue
            coarse,short=evaluate(candidates,all_keys[:entry['coarse_count']],f'class{sweep}_{n[index]}_coarse')
            refined,short=evaluate([selected,*short[:2]],all_keys[:entry['search_count']],f'class{sweep}_{n[index]}_confirm')
            winner=short[0]
            baseline=next(r['development_miou'] for r in refined if r['profile']==selected.record())
            if refined[0]['development_miou']>baseline+1e-6:
                selected=winner;groups=[list(c['synonyms']) for c in records[selected.bank]['classes']];accepted+=1
                print(json.dumps(dict(accepted_class=n[index],sweep=sweep,miou=refined[0]['development_miou'],
                    aliases=groups[index])),flush=True)
                if accepted%5==0:confirmed.append(selected)
            ranked=refined
            # Keep only the current mixed bank and saved drafts; avoid quadratic GPU residency.
            keep={p.bank for p in (*confirmed,selected)}
            for name in list(banks):
                if name.startswith('mix_') and name not in keep:
                    del banks[name],queries[name]
        confirmed.append(selected)
        if not accepted:break
    # Revisit scalar settings after class competition has changed.
    for stage in ('readout','calibration'):
        ranked,winners=evaluate(stage_profiles(stage,selected,entry['search_banks'],bg,residual is not None),
            all_keys[:entry['coarse_count']],'mixed_'+stage)
        _,winners=evaluate([selected,*winners],all_keys[:entry['search_count']],'mixed_'+stage+'_confirm')
        selected=winners[0];confirmed.extend(winners)
    # Larger development subset can reject an overfitted mixture and retain an old seed.
    candidates=list(dict.fromkeys([*confirmed,StaticProfile(**entry['incumbent_profile'])]))
    for p in candidates:
        if p.bank not in banks:add_bank(p.bank,records[p.bank],entry,banks,queries)
    ranked,winners=evaluate(candidates,all_keys,'final_confirmation');choice=copy.deepcopy(ranked[0])
    selected=winners[0];selected_record=records.get(selected.bank,entry['banks'].get(selected.bank))
    choice.update(status='complete',implementation=IMPLEMENTATION,development_keys=all_keys,
        development_source=entry['development_source'],development_confusions={args.dataset:choice['confusion_matrix']},
        vocabulary={selected.bank:selected_record},candidate_manifest=entry['banks'],
        target_label_development=True,inference_uses_target_labels=False,source_protocol=protocol['evaluation'],
        search_passes=evaluation,alias_counts=[len(c['synonyms']) for c in selected_record['classes']])
    save(output/'selection.json',choice)
    return dict(status='complete',processed_images=len(all_keys),total_images=len(all_keys),
        wall_seconds=time.perf_counter()-started,development_miou=choice['development_miou'])


def attach_choice(root,d,entry,models):
    geometry,vip,banks,queries,checkpoints,residual=models
    choice=json.loads((root/'search'/d/'selection.json').read_text());name=choice['profile']['bank']
    add_bank(name,choice['vocabulary'][name],entry,banks,queries)
    return choice


def full(args,entry,models,loaders,output):
    geometry,vip,banks,queries,checkpoints,residual=models
    validation,samples,loader,target,names=loaders;bg=entry['background_index'];n=names[args.dataset]
    choice=attach_choice(Path(args.suite_root),args.dataset,entry,models);profile=StaticProfile(**choice['profile'])
    reader=StaticReader({profile.bank:banks[profile.bank]},{profile.bank:queries[profile.bank]},bg,residual)
    _,old_models=previous.make_readers(geometry,vip,banks,queries,entry,names,residual);original=old_models[args.dataset]
    scorer=CachedWideScorer();strengths=tuple(dict.fromkeys((*original.strengths,profile.strength)))
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,classes=names,choice=choice,
        checkpoints=base.checkpoint_manifest(checkpoints),vocabulary=choice['vocabulary'],
        global_sample_count=len(validation),global_sample_keys_sha256=base.digest([s.key for s in validation]),
        sample_keys=[s.key for s in samples],num_shards=args.num_shards,shard_index=args.shard_index,
        geometry_cap=4,wide_cap=4,fine_forwards=0,prior_development=True)
    matrices={m:np.zeros((len(n),len(n)),np.int64) for m in METHODS};arrays={m:[] for m in METHODS}
    started=time.perf_counter();ignored=0
    for number,sample in enumerate(samples,1):
        image=loader(sample);source=compact_observations(image,geometry,vip,strengths,wide_policy=profile.wide_policy)
        sources={profile.wide_policy:source}
        if original.wide_policy not in sources:
            sources[original.wide_policy]=(replace_wide(source,image,vip) if original.wide_policy=='natural_short336_cap672'
                else compact_observations(image,geometry,vip,strengths,wide_policy=original.wide_policy))
        prob=reader.probabilities(source,profile,{},wide_scorer=scorer)
        predictions={'CuratedTuned':previous.predict(prob,choice,bg)}
        old=original.predict_observations(sources[original.wide_policy],return_probability=True,return_prediction=False,wide_scorer=scorer)[2]
        predictions['PreviousFinal']=old.argmax(0).cpu().numpy()
        label=target(sample,args.dataset,source['output_size']);ignored+=int(((label<0)|(label>=len(n))).sum())
        for method,prediction in predictions.items():
            cm=base.confusion(prediction,label,len(n));matrices[method]+=cm;arrays[method].append(cm)
        if number==1 or number%20==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,n,ignored) for m,cm in matrices.items()}},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row);print(json.dumps(dict(images=number,total=len(samples))),flush=True)
        del sources,source,prob,old
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(a) for m,a in arrays.items()})
    return dict(row,status='complete')


def cost(args,protocol,entry,models,loaders,output):
    import eval_qwen20_kev_natural as measured
    choice=attach_choice(Path(args.suite_root),args.dataset,entry,models);key=choice['profile']['bank']
    geometry,vip,banks,queries,checkpoints,residual=models
    for name in list(banks):
        if name!=key:del banks[name],queries[name]
    torch.cuda.empty_cache()
    return measured.cost(args,protocol,entry,(geometry,vip,banks,queries,checkpoints,residual),loaders,output)


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing worker output preserved')
    protocol=json.loads((root/'protocol.json').read_text());entry=protocol['datasets'][args.dataset]
    loaders=load_samples(args,entry)
    if len(loaders[0])!=entry['total_images']:raise RuntimeError('Global sample count changed')
    needed=None
    if args.mode=='cost':
        choice=json.loads((root/'search'/args.dataset/'selection.json').read_text());key=choice['profile']['bank']
        template=choice['vocabulary'][key]['template']
        needed=('candidate_pool_'+('imagenet' if template=='ImageNet80' else 'segmentation'),)
    models=previous.load_models(args,entry,needed)
    geometry,vip,banks,queries,checkpoints,residual=models
    frozen=frozen_state(geometry,vip);offload_text_towers(geometry,vip,release=True);geometry=CompactGeometry(geometry)
    models=(geometry,vip,banks,queries,checkpoints,residual);output.mkdir(parents=True)
    result=(search(args,protocol,entry,models,loaders,output) if args.mode=='search' else
        cost(args,protocol,entry,models,loaders,output) if args.mode=='cost' else full(args,entry,models,loaders,output))
    result.update(**check_frozen(frozen,geometry,vip));save(output/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('search','full','cost'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
