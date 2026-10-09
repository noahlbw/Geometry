"""Same PC60 observations: automatic, residual max, bounded rival protection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
import eval_residual_ontology as prior
from dinotool.residual_rival_weight import ResidualRivalWeight
from dinotool.taxonomy_readout import natural_profile,canonical_features
from dinotool.development_readout import coupled
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.inference import hann_blend_window
from eval_geometry_vip_reliability import sample_broad,summary
from eval_rival_fine_full import frozen_state,check_frozen,save

METHODS=('TaxonomyAdaptive','ResidualMax','ResidualProtected')


@torch.inference_mode()
def protected_probability(source,p,bank,old_broad,new_broad,text,reader):
    h,w=source['size']
    bank_text=F.normalize(bank.features.float(),dim=-1)
    blend=torch.from_numpy(hann_blend_window(512)).to(text.device)
    with DeviceProbabilityAccumulator(bank.class_count,h,w,text.device) as acc:
        for tile in source['local']:
            features=tile['features'][p.strength].float()
            local=base.alias_class_scores(features@bank_text.T,bank.parent_indices,60)/p.temperature
            top,left=tile['top'],tile['left']
            previous=sample_broad(old_broad,top,left,h,w).reshape_as(local)
            foreground=coupled(local,previous,tile['operator'],p.coupling)[:,1:]
            rival=foreground.argmax(-1)+1
            local[:,0]=reader.score(features@text.T,rival)/p.temperature
            wide=sample_broad(new_broad,top,left,h,w).reshape_as(local)
            logits=coupled(local,wide,tile['operator'],p.coupling)
            dense=F.interpolate(logits.T.reshape(1,60,32,32),(512,512),mode='bilinear',align_corners=False)[0]
            ah,aw=min(512,h-top),min(512,w-left)
            acc.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
        return F.interpolate((acc.probabilities/acc.normalizer[None])[None],source['output_size'],mode='bilinear',align_corners=False)[0]


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output.')
    manifest=json.loads((root/'protocol.json').read_text())
    entry=manifest['datasets']['context60']
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,('semantic_segmentation',))
    frozen=frozen_state(geometry,vip)
    bank=banks['semantic_segmentation']
    p,_=natural_profile(bank)
    residual=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
    if residual['identity']!=manifest['residual_identity']:
        raise RuntimeError('Frozen residual source changed.')
    text=F.normalize(residual['wide'].float().mean(1),dim=-1)
    reader=ResidualRivalWeight(text,canonical_features(bank))
    matrices={m:np.zeros((60,60),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    signature=dict(implementation='geometry-residual-local-rival-protection-v1-20261007',dataset='context60',
        methods=METHODS,classes=dict(context60=bank.class_names),gear=dict(core=manifest['retained_core'],cap=[4,4,0]),
        competitive=None,vocabulary=dict(foreground=entry['banks'],residual=manifest['residual_identity']),
        checkpoints=base.checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args),rule=manifest['protection_rule'])
    output.mkdir(parents=True)
    start=time.perf_counter();tiles=0
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,(p.strength,))
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        modified=broad.clone()
        modified[0]=prior.residual_wide(source,residual['wide'])['max'][0]
        results=dict(TaxonomyAdaptive=base.probabilities(source,p,bank,broad,{}),
            ResidualMax=prior.residual_probability(source,p,bank,modified,0,text,'max'),
            ResidualProtected=protected_probability(source,p,bank,broad,modified,text,reader))
        if not all(bool(torch.isfinite(v).all()) for v in results.values()):
            raise RuntimeError('Nonfinite output.')
        if args.mode=='smoke':
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
                fine_forwards=0,**check_frozen(frozen,geometry,vip)))
            return
        target=load_mask(sample,source['output_size'])
        for method,probability in results.items():
            cm=base.confusion(probability.argmax(0).cpu().numpy(),target,60)
            matrices[method]+=cm;per_image[method].append(cm)
        tiles+=len(source['local'])
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics=dict(context60={m:summary(cm,bank.class_names,0) for m,cm in matrices.items()}),
                diagnostics=dict(context60=dict(tiles=tiles,fine_forwards=0)),wall_seconds=time.perf_counter()-start,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row)
            print(json.dumps(dict(processed=number,total=len(samples))),flush=True)
        del source,broad,modified,results
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{'context60__'+m:np.stack(v) for m,v in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,geometry,vip));save(output/'results.json',row)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--dataset',default='context60')
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--sample-seed',type=int,default=20260923)
    parser.add_argument('--vdd-ontology',default='official')
    main(parser.parse_args())
