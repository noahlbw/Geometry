"""Paired PC60 residual-ontology experiment; no labels select residual queries."""
import argparse
import json
from pathlib import Path
import time
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
from dinotool.residual_ontology import residual_score,complement_queries
from dinotool.vip_official_adapter import VIPSettings,_load_upstream
from dinotool.taxonomy_readout import natural_profile
from dinotool.development_readout import coupled
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.inference import hann_blend_window
from eval_geometry_vip_reliability import sample_broad,summary
from eval_rival_fine_full import frozen_state,check_frozen,save

METHODS=('TaxonomyAdaptive','ResidualMean','ResidualMax')


@torch.inference_mode()
def residual_wide(source,text):
    h,w=source['wide_size']
    maps={m:torch.zeros(1,h,w,device=text.device) for m in ('mean','max')}
    count=torch.zeros(h,w,device=text.device)
    for crop in source['wide']:
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            raw=torch.einsum('bnd,mtd->bnmt',crop['features'],text.float()).mean(-1)[0]*VIPSettings().logit_scale
        top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
        for mode in maps:
            reduced=residual_score(raw,mode,1.).reshape(1,1,21,21)
            dense=F.interpolate(reduced,(336,336),mode='bilinear',align_corners=False)[0].float()
            maps[mode][:,top:top+ah,left:left+aw]+=dense[:,:ah,:aw]
        count[top:top+ah,left:left+aw]+=1
    return {mode:value/count[None] for mode,value in maps.items()}


@torch.inference_mode()
def residual_probability(source,p,bank,broad,background,text,mode):
    h,w=source['size']
    bank_text=F.normalize(bank.features.float(),dim=-1)
    blend=torch.from_numpy(hann_blend_window(512)).to(text.device)
    with DeviceProbabilityAccumulator(bank.class_count,h,w,text.device) as acc:
        for tile in source['local']:
            features=tile['features'][p.strength].float()
            local=base.alias_class_scores(features@bank_text.T,bank.parent_indices,bank.class_count)/p.temperature
            local[:,background]=residual_score(features@text.T,mode,.07)/p.temperature
            top,left=tile['top'],tile['left']
            wide=sample_broad(broad,top,left,h,w).reshape_as(local)
            logits=coupled(local,wide,tile['operator'],p.coupling)
            dense=F.interpolate(logits.T.reshape(1,bank.class_count,32,32),(512,512),mode='bilinear',align_corners=False)[0]
            ah,aw=min(512,h-top),min(512,w-left)
            acc.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
        return F.interpolate((acc.probabilities/acc.normalizer[None])[None],source['output_size'],mode='bilinear',align_corners=False)[0]


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing outputs.')
    manifest=json.loads((root/'protocol.json').read_text())
    entry=manifest['datasets'][args.dataset]
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,('semantic_segmentation',))
    frozen=frozen_state(geometry,vip)
    if args.mode=='prepare':
        if (root/'residual_cache.pt').exists():
            raise RuntimeError('Existing frozen residual cache.')
        with tarfile.open(manifest['metadata_archive'],'r:gz') as archive:
            names,rows=complement_queries(archive.extractfile('labels.txt').read().decode(),manifest['scored_raw_ids'])
        module=_load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py','residual_prompts')
        vip.templates=module.get_text_template('seg_template')
        encoded=vip.encode_queries(names,tuple((name,) for name in names))
        identity=dict(names=names,raw_residual_categories=rows,template='seg_template',
            selection='Official category-name complement of scored raw IDs; no image/mask access.',
            provenance='PC60-specific supplied full ontology, not generated unknown objects and not a synonym pool.')
        identity=json.loads(json.dumps(identity))
        torch.save(dict(identity=identity,wide=encoded.features.cpu()),root/'residual_cache.pt')
        manifest['residual_identity']=identity
        save(root/'protocol.json',manifest)
        output.mkdir(parents=True)
        save(output/'results.json',dict(status='complete',processed_images=0,total_images=0,
            target_masks_loaded=False,residual_queries=len(names),**check_frozen(frozen,geometry,vip)))
        return
    bank=banks['semantic_segmentation']
    p,decision=natural_profile(bank)
    residual=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
    if residual['identity']!=manifest['residual_identity']:
        raise RuntimeError('Frozen residual words differ.')
    text=F.normalize(residual['wide'].float().mean(1),dim=-1)
    matrices={m:np.zeros((bank.class_count,bank.class_count),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    signature=dict(implementation='geometry-taxonomy-residual-ontology-v1-20261007',dataset=args.dataset,
        methods=METHODS,classes={args.dataset:bank.class_names},gear=dict(core=manifest['retained_core'],cap=[4,4,0]),
        competitive=None,vocabulary=dict(foreground=entry['banks'],residual=manifest['residual_identity']),
        checkpoints=base.checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args),profile=p.record(),residual_uses_category_metadata_only=True)
    output.mkdir(parents=True)
    started=time.perf_counter()
    costs=[]
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,(p.strength,))
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        results={'TaxonomyAdaptive':base.probabilities(source,p,bank,broad,{})}
        bgmaps=residual_wide(source,residual['wide'])
        for method,mode in (('ResidualMean','mean'),('ResidualMax','max')):
            changed=broad.clone()
            changed[entry['background_index']]=bgmaps[mode][0]
            results[method]=residual_probability(source,p,bank,changed,entry['background_index'],text,mode)
        if not all(bool(torch.isfinite(v).all()) for v in results.values()):
            raise RuntimeError('Nonfinite prediction.')
        if args.mode=='smoke':
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,profile=decision,residual_queries=len(text),
                geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),fine_forwards=0,
                **check_frozen(frozen,geometry,vip)))
            return
        target=load_mask(sample,source['output_size'])
        for method,probability in results.items():
            cm=base.confusion(probability.argmax(0).cpu().numpy(),target,bank.class_count)
            matrices[method]+=cm
            per_image[method].append(cm)
        costs.append(len(source['local']))
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,bank.class_names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=sum(costs),fine_forwards=0)},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row)
            print(json.dumps(dict(processed=number,total=len(samples))),flush=True)
        del source,broad,results,bgmaps,changed,target
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,geometry,vip))
    save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--dataset',default='context60')
    p.add_argument('--mode',choices=('prepare','smoke','full'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
