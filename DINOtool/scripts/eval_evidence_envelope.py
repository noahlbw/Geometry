"""Paired frozen/envelope evaluation on identical cached visual observations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import eval_development_readout as base
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.evidence_envelope import IMPLEMENTATION
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save,frozen_state,check_frozen

METHODS=('Frozen','Envelope')
ARM_OPTIONS=({'correction':'frozen'},{'correction':'envelope'})
OBSERVATION_OPTIONS={}
SMOKE_VALIDATOR=None
RESULT_AUDITOR=None


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output.')
    protocol=json.loads((root/'protocol.json').read_text())
    entry=protocol['datasets'][args.dataset]
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip)
    residual=None
    if entry.get('residual_identity') is not None:
        encoded=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
        if encoded['identity']!=entry['residual_identity']:
            raise RuntimeError('Residual identity differs.')
        residual=encoded['wide']
    common=dict(family=entry['family'],background=entry['background_index'],
        soft=entry['family']=='remote_sensing',residual_features=residual,
        residual_mode='protected' if residual is not None else 'max')
    models={m:TaxonomyInference(geometry,vip,banks,queries,**options,**common)
        for m,options in zip(METHODS,ARM_OPTIONS)}
    names=next(iter(banks.values())).class_names
    count=len(names)
    matrices={m:np.zeros((count,count),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,classes={args.dataset:names},
        gear=dict(core=protocol['retained_core'],cap=[4,4,0]),competitive=None,vocabulary=entry['banks'],
        residual_identity=entry.get('residual_identity'),checkpoints=base.checkpoint_manifest(checkpoints),
        global_sample_count=len(validation),global_sample_keys_sha256=base.digest([s.key for s in validation]),
        sample_keys=[s.key for s in samples],sample_keys_sha256=base.digest([s.key for s in samples]),
        num_shards=args.num_shards,shard_index=args.shard_index,config=vars(args),provenance=protocol['source_label_development'])
    output.mkdir(parents=True)
    started=time.perf_counter();tiles=0;audit={}
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        strengths=tuple(dict.fromkeys(s for model in models.values() for s in model.strengths))
        source=base.observations(image,geometry,vip,strengths,**OBSERVATION_OPTIONS)
        results={}
        for m,model in models.items():
            prediction,diagnostic=model.predict_observations(source)
            if diagnostic['geometry_encodings']>4 or diagnostic['wide_encodings']>4 or diagnostic['fine_forwards']:
                raise RuntimeError('Visual budget changed.')
            results[m]=prediction
        if args.mode=='smoke':
            checks={} if SMOKE_VALIDATOR is None else SMOKE_VALIDATOR(models,source,results)
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,shared_visual_observations=True,**checks,**check_frozen(frozen,geometry,vip)))
            return
        target=load_mask(sample,source['output_size'])
        for m,prediction in results.items():
            cm=base.confusion(prediction,target,count)
            matrices[m]+=cm;per_image[m].append(cm)
        if RESULT_AUDITOR is not None:
            audit=RESULT_AUDITOR(audit,target,results,count)
        tiles+=len(source['local'])
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=tiles,fine_forwards=0)},wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            if RESULT_AUDITOR is not None:row['readout_transitions']=audit
            save(output/'results.json',row)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source,results
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    row.update(status='complete',**check_frozen(frozen,geometry,vip));save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
