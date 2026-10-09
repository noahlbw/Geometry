"""Frozen global alias simplification and same-observation coupling controls.

No search, threshold fitting, fine observations, or masks in inference. LoveDA
P is an independently read six-class bank, never a shifted D argmax.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
import eval_taxonomy_readout as scalar
from dinotool.coverage_soft_alias import coverage_scores
from dinotool.development_readout import Profile
from dinotool.prompts import ClassSpec
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_gear_ov import protocol as rs_protocol
from eval_geometry_vip_reliability import summary
from eval_residual_ontology import residual_wide
from eval_residual_rival_protection import protected_probability
from eval_rival_fine_full import save,frozen_state,check_frozen

IMPLEMENTATION='geometry-task-alias-global-finalization-v1-20261009'
METHODS=('Reference20SameViews','Retained','Uniform','UniformLocal','UniformWide','UniformMean','RetainedScalar')


def load_samples(args,entry):
    if args.dataset!='loveda':
        validation,samples,loader,target=base.load_samples(args,entry)
        names=tuple(c['name'] for c in entry['banks']['original']['classes'])
        return validation,samples,loader,lambda s,p,shape:target(s,shape),{args.dataset:names}
    specs=[ClassSpec(c['name'],tuple(c['synonyms'])) for c in entry['banks']['original']['classes']]
    validation,protocols,loader,target=rs_protocol(args,specs)
    selected=validation[:1] if args.mode=='smoke' else validation[args.shard_index::args.num_shards]
    if tuple(c.name for c in protocols['D'])!=tuple(c.name for c in specs):
        raise RuntimeError('LoveDA D order differs from frozen bank.')
    return validation,selected,lambda sample:loader(sample.image_path),target,{p:tuple(c.name for c in g) for p,g in protocols.items()}


def foreground_banks(banks,queries):
    """Exact P bank views of D text tensors; background is not predicted in P."""
    bs,qs={},{}
    for k,b in banks.items():
        if b.class_names[0]!='background':
            raise ValueError('LoveDA background must be slot zero.')
        mask=b.parent_indices!=0
        names=b.class_names[1:]
        aliases=tuple(a for a,p in zip(b.alias_names,b.parent_indices.tolist()) if p!=0)
        bs[k]=TCPRTextBank(b.features[mask],b.parent_indices[mask]-1,b.canonical_mask[mask],names,aliases)
        qs[k]=VIPQueries(queries[k].features[mask],bs[k].parent_indices,names,aliases)
        bs[k].validate()
    return bs,qs


def build_models(geometry,vip,banks,queries,entry,residual=None):
    kwargs=dict(family=entry['family'],background=entry['background_index'],residual_features=residual)
    models={m:TaxonomyInference.for_finalization(geometry,vip,banks,queries,
        ordinary_alias_policy='retained' if m=='Retained' else 'uniform',**kwargs)
        for m in ('Retained','Uniform','UniformLocal','UniformWide','UniformMean')}
    for m,correction in [('UniformLocal','local_endpoint'),('UniformWide','wide_endpoint'),('UniformMean','equal_mean')]:
        models[m].correction=correction
    return models


@torch.inference_mode()
def scalar_probability(source,model,banks,queries,diagnostic):
    p=Profile(**diagnostic['profile'])
    broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
    if model.residual_features is not None:
        changed=broad.clone()
        changed[model.background]=residual_wide(source,model.residual_features)['max'][0]
        return protected_probability(source,p,banks[p.bank],broad,changed,model.residual_text,model.residual_reader)
    if model.soft:
        scalar.competition_scores=coverage_scores
        return scalar.soft_probability(source,p,banks[p.bank],broad)
    return base.probabilities(source,p,banks[p.bank],broad,{})


def agreement(reference,actual):
    torch.testing.assert_close(reference,actual,rtol=1e-5,atol=1e-6)
    changed=reference.argmax(0)!=actual.argmax(0)
    top=reference.topk(2,dim=0).values
    gap=float((top[0]-top[1])[changed].max()) if bool(changed.any()) else 0.
    if gap>2e-6:
        raise RuntimeError('Deployed versus scalar difference exceeds a numerical tie.')
    return dict(changed_pixels=int(changed.sum()),maximum_changed_gap=gap,
                maximum_probability_error=float((reference-actual).abs().max()))


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing output; preserve it.')
    protocol=json.loads((root/'protocol.json').read_text());entry=protocol['datasets'][args.dataset]
    validation,samples,load_image,load_mask,names=load_samples(args,entry)
    if len(validation)!=entry['total_images'] or not samples:raise RuntimeError('Unexpected complete sample coverage.')
    if args.dataset=='loveda':args.original_cache_key='D__original20'
    needed=('original','original_imagenet','focused20') if entry['family']=='remote_sensing' else ('original','semantic_segmentation')
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip)
    residual=None
    if entry.get('residual_identity') is not None:
        encoded=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
        if encoded['identity']!=entry['residual_identity']:raise RuntimeError('Residual ontology identity differs.')
        residual=encoded['wide']
    protocol_banks={p:(banks,queries) for p in names}
    if args.dataset=='loveda':protocol_banks['P']=foreground_banks(banks,queries)
    models={}
    for p,(bs,qs) in protocol_banks.items():
        e=dict(entry,background_index=None if p=='P' else entry['background_index'])
        models[p]=build_models(geometry,vip,bs,qs,e,residual)
    primary=models['D' if args.dataset=='loveda' else args.dataset]['Retained']
    strengths=tuple(dict.fromkeys((2.,*primary.strengths)))
    matrices={p:{m:np.zeros((len(n),len(n)),np.int64) for m in METHODS} for p,n in names.items()}
    per_image={p+'__'+m:[] for p in names for m in METHODS}
    ignored={p:0 for p in names};checks=[];choices={};tiles=0;wide_calls=0
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,classes=names,
        gear=dict(core=protocol['retained_core'],cap=[4,4,0],wide_policy=primary.wide_policy),
        competitive=dict(ordinary_alias_policies=['retained','uniform'],no_new_selector=True),
        vocabulary=entry['banks'],residual_identity=entry.get('residual_identity'),
        checkpoints=base.checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args),provenance=protocol['source_label_development'])
    output.mkdir(parents=True);started=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,strengths,wide_policy=primary.wide_policy)
        tiles+=len(source['local']);wide_calls+=len(source['wide'])
        check=dict(key=sample.key,protocols={})
        for p,(bs,qs) in protocol_banks.items():
            retained,diagnostic,prob=models[p]['Retained'].predict_observations(source,return_probability=True)
            independent=scalar_probability(source,models[p]['Retained'],bs,qs,diagnostic)
            record=dict(retained=agreement(independent,prob),profile=diagnostic['profile'])
            results={'Retained':retained,'RetainedScalar':independent.argmax(0).cpu().numpy()}
            del independent,prob
            uniform,_,prob=models[p]['Uniform'].predict_observations(source,return_probability=True)
            independent=scalar_probability(source,models[p]['Uniform'],bs,qs,diagnostic)
            record['uniform']=agreement(independent,prob)
            if entry['family']=='natural' and not np.array_equal(retained,uniform):
                raise RuntimeError('Natural foreground already uniform; simplification changed predictions.')
            results['Uniform']=uniform
            del independent,prob
            for m in ('UniformLocal','UniformWide','UniformMean'):
                results[m]=models[p][m].predict_observations(source)[0]
            reference=base.probabilities(source,Profile(),bs['original'],base.wide_scores(source,qs['original'],1.,1.),{})
            results['Reference20SameViews']=reference.argmax(0).cpu().numpy();del reference
            if any(v.shape!=tuple(image.shape[-2:]) for v in results.values()):raise RuntimeError('Output dimensions differ.')
            check['protocols'][p]=record
            choice=p+'__'+diagnostic['profile']['bank'];choices[choice]=choices.get(choice,0)+1
            if args.mode!='smoke':
                # All model/scalar inference is frozen before any target loader.
                target=load_mask(sample,p,source['output_size']);count=len(names[p])
                ignored[p]+=int(((target<0)|(target>=count)).sum())
                for m,prediction in results.items():
                    cm=base.confusion(prediction,target,count);matrices[p][m]+=cm;per_image[p+'__'+m].append(cm)
            del results
        checks.append(check)
        if args.mode=='smoke':
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,scalar_agreement_verified=True,cap_verified=True,check=check,
                **check_frozen(frozen,geometry,vip)))
            return
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={p:{m:summary(cm,names[p],ignored[p]) for m,cm in group.items()} for p,group in matrices.items()},
                diagnostics={p:dict(tiles=tiles,wide_encodings=wide_calls,fine_forwards=0) for p in names},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                choices=choices.copy())
            save(output/'results.json',row)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{m:np.stack(v) for m,v in per_image.items()})
    save(output/'deployment_checks.json',dict(samples=checks))
    row.update(status='complete',scalar_agreement_verified=True,**check_frozen(frozen,geometry,vip))
    save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
