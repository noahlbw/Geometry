"""Fixed Geometry/wide/writeback, finite language-bank and readout development."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import eval_development_readout as base
from eval_alias_finalization import load_samples, foreground_banks
from eval_natural_static_search import move, observation_pair
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save, frozen_state, check_frozen
from eval_stride_ov_loveda_e1 import make_checkpoints
from dinotool.development_readout import threshold_histogram, biased_probabilities, score
from dinotool.natural_static_search import StaticProfile, StaticReader, stage_profiles, rank, BIASES, THRESHOLDS, projected
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.model import DINOTextSegmenter
from dinotool.geometry_execution import GeometryExecution
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.vip_official_adapter import VIPQueries, _load_upstream

IMPLEMENTATION='geometry-kev-variable-alias-static-search-v1-20261009'
PREVIOUS='__previous_final__'


def task_entry(args):
    root=Path(args.suite_root)
    protocol=json.loads((root/'protocol.json').read_text())
    e=dict(protocol['datasets'][args.dataset])
    judgment=json.loads((root/'language'/args.dataset/'selection.json').read_text())
    if judgment['status']!='complete' or not judgment['taxonomy_only'] or judgment['target_masks_loaded']:
        raise RuntimeError('Language selection not complete or has an invalid source.')
    e['banks']={**e['banks'],**judgment['banks']}
    return protocol,e,judgment


@torch.inference_mode()
def load_models(args,entry,needed=None):
    import subprocess
    from eval_vip_official_eight import PINNED_COMMIT
    if subprocess.check_output(['git','-C',args.upstream_root,'rev-parse','HEAD'],text=True).strip()!=PINNED_COMMIT:
        raise RuntimeError('Pinned VIP implementation changed.')
    root=Path(args.suite_root);path=root/'text_cache'/(args.dataset+'.pt')
    checkpoints=make_checkpoints(args)
    identity=dict(banks=entry['banks'],checkpoints=base.checkpoint_manifest(checkpoints))
    old_root=root.parent/'alias_finalization_20261009'
    if path.exists():
        packed=torch.load(path,map_location='cpu',weights_only=True)
        if packed['identity']!=identity:raise RuntimeError('Frozen encoded vocabulary changed.')
    else:
        packed=None
    geometry=GeometryExecution(TCPRSegmenter(DINOTextSegmenter(checkpoints,device=args.device),
        TCPRConfig(geometry_depth=2,maximum_aliases_per_class=20)))
    vip=FiniteVIPObserver(DINOTextSegmenter(checkpoints,device=args.device),Path(args.upstream_root))
    if packed is None:
        cached=torch.load(old_root/'text_cache'/(args.dataset+'.pt'),map_location='cpu',weights_only=True)
        if cached['identity']['banks']!=entry['baseline_banks']:raise RuntimeError('Incumbent text identity changed.')
        original=torch.load(args.original_cache,map_location='cpu',weights_only=True)
        key='D__original20' if args.dataset=='loveda' else args.dataset+'__original20'
        if original['identity']['aliases'][key]!=[c['synonyms'] for c in entry['banks']['original']['classes']]:
            raise RuntimeError('Original vocabulary differs.')
        rows={k:v for k,v in cached['encoded'].items() if k in entry['baseline_banks']}
        rows['original']=original['encoded'][key]
        lookup={}
        for name,row in rows.items():
            template=entry['banks'][name]['template']
            if template=='RS6':continue  # The local RS6 representation is kept exactly as cached.
            table=lookup.setdefault(template,{})
            aliases=[a for c in entry['banks'][name]['classes'] for a in c['synonyms']]
            for i,a in enumerate(aliases):table.setdefault(a,row['wide'][i])
        prompts=_load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py','kev_fixed_prompts')
        for template in ('ImageNet80','seg_template'):
            table=lookup.setdefault(template,{})
            words=list(dict.fromkeys(a for b in entry['banks'].values() if b['template']==template
                                      for c in b['classes'] for a in c['synonyms']))
            missing=[a for a in words if a not in table]
            vip.templates=prompts.get_text_template('openai_imagenet_template' if template=='ImageNet80' else template)
            # Encode every missing alias once per template, then take exact tensor subsets.
            for begin in range(0,len(missing),32):
                batch=missing[begin:begin+32]
                query=vip.encode_queries(('pending',),(tuple(batch),))
                for alias,feature in zip(batch,query.features):table[alias]=feature.cpu()
                print(json.dumps(dict(dataset=args.dataset,text_template=template,encoded=min(begin+32,len(missing)),total=len(missing))),flush=True)
        for name,record in entry['banks'].items():
            if name in rows:continue
            groups=[c['synonyms'] for c in record['classes']]
            features=torch.stack([lookup[record['template']][a] for g in groups for a in g])
            parents=torch.tensor([i for i,g in enumerate(groups) for _ in g],dtype=torch.long)
            canonical=torch.zeros(len(parents),dtype=torch.bool)
            canonical[torch.tensor(np.cumsum([0]+[len(g) for g in groups[:-1]]))]=True
            rows[name]=dict(local=F.normalize(features.float().mean(1),dim=-1),wide=features,
                            parents=parents,canonical=canonical)
        # Common per-template stores prevent serializing overlapping variable banks repeatedly.
        encoded={}
        stores={t:torch.stack(list(table.values())) for t,table in lookup.items()}
        indices={t:{a:i for i,a in enumerate(table)} for t,table in lookup.items()}
        for name,record in entry['banks'].items():
            if record['template']=='RS6' or name=='original':
                encoded[name]=rows[name]
            else:
                aliases=[a for c in record['classes'] for a in c['synonyms']]
                encoded[name]=dict(template=record['template'],indices=torch.tensor([indices[record['template']][a] for a in aliases]),
                    parents=rows[name]['parents'],canonical=rows[name]['canonical'])
        packed=dict(identity=identity,encoded=encoded,stores=stores)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(packed,path.with_suffix('.tmp'));path.with_suffix('.tmp').rename(path)
        del rows,lookup,cached,original
    banks,queries={},{}
    for name,record in entry['banks'].items():
        if needed is not None and name not in needed:continue
        row=packed['encoded'][name]
        wide=packed['stores'][row['template']][row['indices']] if 'indices' in row else row['wide']
        local=F.normalize(wide.float().mean(1),dim=-1) if 'indices' in row else row['local']
        names=tuple(c['name'] for c in record['classes']);aliases=tuple(a for c in record['classes'] for a in c['synonyms'])
        parents=row['parents'].to(args.device)
        banks[name]=TCPRTextBank(local.to(args.device),parents,row['canonical'].to(args.device),names,aliases)
        banks[name].validate();queries[name]=VIPQueries(wide.to(args.device),parents,names,aliases)
    residual=None
    if entry.get('residual_identity') is not None:
        item=torch.load(old_root/'residual_cache.pt',map_location=args.device,weights_only=True)
        if item['identity']!=entry['residual_identity']:raise RuntimeError('Residual ontology differs.')
        residual=item['wide']
    return geometry,vip,banks,queries,checkpoints,residual


def make_readers(geometry,vip,banks,queries,entry,names,residual):
    groups={p:(banks,queries) for p in names}
    if 'P' in names:groups['P']=foreground_banks(banks,queries)
    readers,previous={},{}
    for p,(bs,qs) in groups.items():
        bg=None if p=='P' else entry['background_index']
        readers[p]=StaticReader(bs,qs,bg,residual)
        previous[p]=TaxonomyInference.for_finalization(geometry,vip,bs,qs,
            ordinary_alias_policy='uniform',family=entry['family'],background=bg,residual_features=residual)
    return readers,previous


def predict_probability(source,profile,reader,previous,cache):
    if profile.bank==PREVIOUS:
        if PREVIOUS not in cache:cache[PREVIOUS]=previous.predict_observations(source,return_probability=True)[2]
        return cache[PREVIOUS]
    return reader.probabilities(source,profile,cache)


def source_pair(image,geometry,vip,strengths):
    return observation_pair(image,geometry,vip,strengths)


def background_for(entry,p):return None if p=='P' else entry['background_index']


def predict(probability,choice,bg):
    probability=biased_probabilities(probability,choice['background_bias'] if bg is not None else 0.,bg)
    confidence,prediction=probability.max(0)
    if bg is not None:prediction=prediction.masked_fill(confidence<choice['background_threshold'],bg)
    return prediction.cpu().numpy()


def rank_protocols(histograms,candidates,entry,names):
    primary='D' if 'D' in names else next(iter(names))
    rows=rank(histograms[primary],candidates,background_for(entry,primary))
    if 'P' in names:
        # P has no rejection/background: evaluate the actual six-class probabilities.
        supports=histograms['P'][0,0].sum(0).sum(1)>0
        p_scores=[score(h[0].sum(0),supports) for h in histograms['P']]
        for row in rows:
            row['development_D_miou']=row['development_miou']
            row['development_P_miou']=p_scores[row['profile_index']]
            row['development_miou']=(row['development_miou']+row['development_P_miou'])/2
        rows.sort(key=lambda v:-v['development_miou'])
    return rows


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing worker output is preserved.')
    protocol,entry,judgment=task_entry(args)
    validation,samples,load_image,load_mask,names=load_samples(args,entry)
    if len(validation)!=entry['total_images']:raise RuntimeError('Global sample count differs.')
    if args.dataset=='loveda' and args.mode=='search':
        lookup={s.key:s for s in validation};samples=[lookup[k] for k in entry['development_keys']]
    selection=json.loads((root/'search'/args.dataset/'selection.json').read_text()) if args.mode=='full' else None
    needed=None
    if selection is not None:
        chosen=selection['profile']['bank']
        needed=set(entry['baseline_banks'])
        if chosen!=PREVIOUS:
            needed.add(chosen)
            if chosen.startswith('kev'):
                needed.add('pool_'+('imagenet' if entry['banks'][chosen]['template']=='ImageNet80' else 'segmentation'))
    geometry,vip,banks,queries,checkpoints,residual=load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip)
    readers,previous=make_readers(geometry,vip,banks,queries,entry,names,residual)
    output.mkdir(parents=True);started=time.perf_counter()
    incumbent=StaticProfile(**entry['incumbent_profile'])
    baseline=replace(incumbent,bank=PREVIOUS)
    if args.mode=='search':
        selected=incumbent;pool=[baseline,incumbent];history=[];memory={}
        lookup={s.key:s for s in samples}
        coarse=entry['development_keys'][:entry['coarse_count']]
        # Full development observations stay in RAM and are reused across all coordinates.
        for stage in ('words','readout','calibration','revisit_words','refinement'):
            seed=incumbent if selected.bank==PREVIOUS else selected
            candidates=tuple(dict.fromkeys(pool)) if stage=='refinement' else tuple(dict.fromkeys(
                [baseline,selected,*stage_profiles(stage,seed,tuple(banks),entry['background_index'],residual is not None)]))
            keys=entry['development_keys'] if stage=='refinement' else coarse
            histograms={p:np.zeros((len(candidates),len(BIASES) if background_for(entry,p) is not None else 1,
                len(THRESHOLDS)+1,len(n),len(n)),np.int64) for p,n in names.items()}
            for number,key in enumerate(keys,1):
                sample=lookup[key]
                if key not in memory:
                    image=load_image(sample)
                    memory[key]=move(source_pair(image,geometry,vip,('original',.5,1.,2.,3.,4.)),'cpu')
                pair=move(memory[key],args.device)
                # Targets are scoring inputs only, loaded after the mask-free visual observation.
                targets={p:torch.as_tensor(load_mask(sample,p,pair['long448']['output_size']),device=args.device) for p in names}
                for p,reader in readers.items():
                    caches={policy:{} for policy in pair};bg=background_for(entry,p)
                    biases=BIASES if bg is not None else (0.,)
                    for index,profile in enumerate(candidates):
                        prob=predict_probability(pair[profile.wide_policy],profile,reader,previous[p],caches[profile.wide_policy])
                        for j,bias in enumerate(biases):
                            histograms[p][index,j]+=threshold_histogram(prob,targets[p],bg,bias,THRESHOLDS)
                del pair,targets,caches
                if number==1 or number%4==0 or number==len(keys):
                    save(output/'results.json',dict(status='running',stage=stage,processed_images=number,total_images=len(keys),
                        profiles=len(candidates),wall_seconds=time.perf_counter()-started))
                    print(json.dumps(dict(dataset=args.dataset,stage=stage,processed=number,total=len(keys),profiles=len(candidates))),flush=True)
            ranked=rank_protocols(histograms,candidates,entry,names)
            winners=[]
            for row in ranked:
                profile=StaticProfile(**row['profile'])
                if profile not in winners:winners.append(profile)
                if len(winners)==3:break
            selected=winners[0];pool.extend(winners)
            record=dict(stage=stage,sample_keys=keys,candidates=[c.record() for c in candidates],
                top_profiles=[c.record() for c in winners],ranked=[{k:v for k,v in r.items() if k!='confusion_matrix'} for r in ranked],
                best=ranked[0])
            history.append(record);save(output/('stage_'+stage+'.json'),record)
            np.savez_compressed(output/('stage_'+stage+'_histograms.npz'),**histograms)
        chosen=dict(ranked[0]);profile_index=chosen['profile_index'];bias_index=chosen['bias_index']
        from dinotool.development_readout import threshold_matrices
        chosen['development_confusions']={p:threshold_matrices(histograms[p][profile_index,bias_index if p!='P' else 0],
            background_for(entry,p),THRESHOLDS)[chosen['background_threshold'] if p!='P' else 0.].tolist() for p in names}
        chosen.update(implementation=IMPLEMENTATION,status='complete',development_keys=entry['development_keys'],
            development_source=entry['development_source'],vocabulary=entry['banks'],judgment=judgment['checkpoint'],
            inference_uses_target_labels=False,target_label_development=True,source_protocol=protocol['evaluation'])
        save(output/'selection.json',chosen)
        save(output/'results.json',dict(status='complete',processed_images=len(entry['development_keys']),
            total_images=len(entry['development_keys']),stages=len(history),wall_seconds=time.perf_counter()-started,
            **check_frozen(frozen,geometry,vip)))
        return
    profile=StaticProfile(**selection['profile'])
    primary=next(iter(previous.values()))
    required=tuple(dict.fromkeys((*primary.strengths,profile.strength if profile.bank!=PREVIOUS else 'original')))
    methods=('KevTuned','PreviousFinal','UnscreenedSameParameters')
    matrices={p:{m:np.zeros((len(n),len(n)),np.int64) for m in methods} for p,n in names.items()}
    arrays={p+'__'+m:[] for p in names for m in methods};ignored={p:0 for p in names};tiles=wide_calls=0
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=methods,classes=names,choice=selection,
        checkpoints=base.checkpoint_manifest(checkpoints),vocabulary=entry['banks'],
        global_sample_count=len(validation),global_sample_keys_sha256=base.digest([s.key for s in validation]),
        sample_keys=[s.key for s in samples],num_shards=args.num_shards,shard_index=args.shard_index,
        prior_development=True,geometry_cap=4,wide_cap=4,fine_forwards=0)
    baseline_policy=primary.wide_policy
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        old=base.projected;base.projected=projected
        try:source=base.observations(image,geometry,vip,required,wide_policy=profile.wide_policy)
        finally:base.projected=old
        pair={profile.wide_policy:source}
        if baseline_policy!=profile.wide_policy:
            # This is a paired accuracy control; standalone timing below excludes this extra view.
            if baseline_policy=='natural_short336_cap672':
                from dinotool.natural_wide_resolution import replace_wide
                pair[baseline_policy]=replace_wide(source,image,vip)
            else:raise RuntimeError('Unexpected baseline-wide policy; preserve outputs and inspect.')
        tiles+=len(source['local']);wide_calls+=len(source['wide'])
        for p,reader in readers.items():
            cache={};bg=background_for(entry,p)
            prev=previous[p].predict_observations(pair[baseline_policy],return_probability=True)[2]
            if profile.bank==PREVIOUS:cache[PREVIOUS]=prev
            tuned=predict_probability(source,profile,reader,previous[p],cache)
            unprofile=profile
            if profile.bank.startswith('kev'):
                unprofile=replace(profile,bank='pool_'+('imagenet' if entry['banks'][profile.bank]['template']=='ImageNet80' else 'segmentation'))
            unfiltered=predict_probability(source,unprofile,reader,previous[p],cache)
            predictions=dict(KevTuned=predict(tuned,selection,bg),PreviousFinal=prev.argmax(0).cpu().numpy(),
                             UnscreenedSameParameters=predict(unfiltered,selection,bg))
            target=load_mask(sample,p,source['output_size']);count=len(names[p])
            ignored[p]+=int(((target<0)|(target>=count)).sum())
            for m,prediction in predictions.items():
                cm=base.confusion(prediction,target,count);matrices[p][m]+=cm;arrays[p+'__'+m].append(cm)
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={p:{m:summary(cm,names[p],ignored[p]) for m,cm in group.items()} for p,group in matrices.items()},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                diagnostics=dict(geometry_encodings=tiles,primary_wide_encodings=wide_calls,fine_forwards=0))
            save(output/'results.json',row)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{k:np.stack(v) for k,v in arrays.items()})
    row.update(status='complete',**check_frozen(frozen,geometry,vip));save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('search','full'),required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
