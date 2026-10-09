"""Full paired scalar/deployed verification of the frozen five-task candidate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
import eval_taxonomy_readout as scalar
from eval_residual_ontology import residual_wide
from eval_residual_rival_protection import protected_probability
from dinotool.coverage_soft_alias import coverage_scores
from dinotool.development_readout import Profile
from dinotool.taxonomy_inference import TaxonomyInference
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save,frozen_state,check_frozen

METHODS=('Reference','ScalarFrozen','Deployed')


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output.')
    protocol=json.loads((root/'protocol.json').read_text())
    entry=protocol['datasets'][args.dataset]
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    needed=('original','original_imagenet','focused20') if entry['family']=='remote_sensing' else ('original','semantic_segmentation')
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip)
    residual=None
    if entry.get('residual_identity') is not None:
        encoded=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
        if encoded['identity']!=entry['residual_identity']:
            raise RuntimeError('Residual identity differs.')
        residual=encoded['wide']
    model=TaxonomyInference(geometry,vip,banks,queries,family=entry['family'],background=entry['background_index'],
        soft=entry['family']=='remote_sensing',residual_features=residual,
        residual_mode='protected' if residual is not None else 'max')
    strengths=tuple(dict.fromkeys((2.,*model.strengths)))
    names=banks['original'].class_names
    count=len(names)
    matrices={m:np.zeros((count,count),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    checks=[]
    scalar.competition_scores=coverage_scores
    signature=dict(implementation='geometry-task-adaptive-deployed-v1-20261007',dataset=args.dataset,
        methods=METHODS,classes={args.dataset:names},gear=dict(core=protocol['retained_core'],cap=[4,4,0]),
        competitive=None,vocabulary=entry['banks'],residual_identity=entry.get('residual_identity'),
        checkpoints=base.checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args),provenance=protocol['source_label_development'])
    output.mkdir(parents=True)
    started=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,strengths)
        prediction,diagnostic,deployed=model.predict_observations(source,return_probability=True)
        p=Profile(**diagnostic['profile'])
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        if residual is not None:
            modified=broad.clone()
            modified[entry['background_index']]=residual_wide(source,residual)['max'][0]
            text=F.normalize(residual.float().mean(1),dim=-1)
            selected=protected_probability(source,p,banks[p.bank],broad,modified,text,model.residual_reader)
        elif model.soft:
            selected=scalar.soft_probability(source,p,banks[p.bank],broad)
        else:
            selected=base.probabilities(source,p,banks[p.bank],broad,{})
        reference=base.probabilities(source,Profile(),banks['original'],base.wide_scores(source,queries['original'],1.,1.),{})
        torch.testing.assert_close(selected,deployed,rtol=1e-5,atol=1e-6)
        changed=selected.argmax(0)!=deployed.argmax(0)
        top=selected.topk(2,dim=0).values
        gap=float((top[0]-top[1])[changed].max()) if bool(changed.any()) else 0.
        if gap>2e-6:
            raise RuntimeError('Non-tie deployed prediction difference.')
        checks.append(dict(key=sample.key,changed_pixels=int(changed.sum()),maximum_changed_gap=gap,
            maximum_probability_error=float((selected-deployed).abs().max()),diagnostic=diagnostic))
        if args.mode=='smoke':
            retained,_=base.retained_predict(image,geometry,{args.dataset:banks['original']},vip,
                {args.dataset:queries['original']},methods=(base.REFERENCE,))
            if not np.array_equal(reference.argmax(0).cpu().numpy(),retained[args.dataset][base.REFERENCE]):
                raise RuntimeError('Reference smoke mismatch.')
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,exact_retained_prediction=True,check=checks[-1],**check_frozen(frozen,geometry,vip)))
            return
        target=load_mask(sample,source['output_size'])
        for method,probability in dict(Reference=reference,ScalarFrozen=selected,Deployed=deployed).items():
            cm=base.confusion(probability.argmax(0).cpu().numpy(),target,count)
            matrices[method]+=cm
            per_image[method].append(cm)
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=sum(v['diagnostic']['geometry_encodings'] for v in checks),
                    changed_pixels=sum(v['changed_pixels'] for v in checks),fine_forwards=0)},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source,reference,selected,deployed,broad,changed,top,target
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    save(output/'deployment_checks.json',dict(samples=checks))
    row.update(status='complete',**check_frozen(frozen,geometry,vip))
    save(output/'results.json',row)


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
