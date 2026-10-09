"""Frozen rank-driven/task-view calibration, no online label access or rejection."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
from dinotool.taxonomy_readout import IMPLEMENTATION,natural_profile,rs_profile
from dinotool.development_readout import Profile,coupled
from dinotool.evidence_adaptive_readout import competition_scores
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.inference import hann_blend_window
from eval_geometry_vip_reliability import sample_broad,summary
from eval_rival_fine_full import save,frozen_state,check_frozen

METHODS=('Reference','TaxonomyAdaptive','TaxonomySoft')


@torch.inference_mode()
def soft_probability(source,p,bank,broad):
    h,w=source['size']
    text=F.normalize(bank.features.float(),dim=-1)
    blend=torch.from_numpy(hann_blend_window(512)).to(text.device)
    with DeviceProbabilityAccumulator(bank.class_count,h,w,text.device) as acc:
        for tile in source['local']:
            alias=tile['features'][p.strength].float()@text.T
            local=competition_scores(alias,bank)/p.temperature
            top,left=tile['top'],tile['left']
            wide=sample_broad(broad,top,left,h,w).reshape_as(local)
            logits=coupled(local,wide,tile['operator'],p.coupling)
            dense=F.interpolate(logits.T.reshape(1,bank.class_count,32,32),(512,512),mode='bilinear',align_corners=False)[0]
            ah,aw=min(512,h-top),min(512,w-left)
            acc.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
        result=F.interpolate((acc.probabilities/acc.normalizer[None])[None],source['output_size'],mode='bilinear',align_corners=False)[0]
    return result


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing outputs.')
    protocol=json.loads((root/'protocol.json').read_text())
    entry=protocol['datasets'][args.dataset]
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    needed=('original','original_imagenet','focused20') if entry['family']=='remote_sensing' else ('original','semantic_segmentation')
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip)
    names,classes=banks['original'].class_names,banks['original'].class_count
    static,static_diagnostic=natural_profile(banks['semantic_segmentation']) if entry['family']=='natural' else (None,None)
    strengths=(2.,1.,3.,'original') if static is None else tuple(dict.fromkeys((2.,static.strength)))
    matrices={m:np.zeros((classes,classes),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    costs,choices=[],[]
    signature=dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=METHODS,classes={args.dataset:names},
        gear=dict(core=protocol['retained_core'],cap=[4,4,0]),competitive=None,vocabulary=entry['banks'],
        checkpoints=base.checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,shard_index=args.shard_index,
        config=vars(args),static_natural_selection=static_diagnostic,
        provenance=protocol['source_label_development'])
    output.mkdir(parents=True)
    started=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,strengths)
        chosen,diagnostic=rs_profile(source,banks,entry['background_index']) if static is None else (static,static_diagnostic)
        cache={}
        broad=base.wide_scores(source,queries[chosen.bank],chosen.tau,chosen.tem)
        reference=base.probabilities(source,Profile(),banks['original'],base.wide_scores(source,queries['original'],1.,1.),cache)
        selected=base.probabilities(source,chosen,banks[chosen.bank],broad,cache)
        soft=soft_probability(source,chosen,banks[chosen.bank],broad)
        probabilities=dict(Reference=reference,TaxonomyAdaptive=selected,TaxonomySoft=soft)
        if not all(bool(torch.isfinite(v).all()) for v in probabilities.values()):
            raise RuntimeError('Nonfinite output.')
        if args.mode=='smoke':
            retained,_=base.retained_predict(image,geometry,{args.dataset:banks['original']},vip,
                {args.dataset:queries['original']},methods=(base.REFERENCE,))
            if not np.array_equal(reference.argmax(0).cpu().numpy(),retained[args.dataset][base.REFERENCE]):
                raise RuntimeError('Reference replay mismatch.')
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,exact_retained_prediction=True,selected=diagnostic,**check_frozen(frozen,geometry,vip)))
            return
        target=load_mask(sample,source['output_size'])
        for m,p in probabilities.items():
            cm=base.confusion(p.argmax(0).cpu().numpy(),target,classes)
            matrices[m]+=cm
            per_image[m].append(cm)
        costs.append((len(source['local']),len(source['wide'])))
        choices.append(dict(key=sample.key,profile=chosen.record(),diagnostic=diagnostic))
        if number==1 or number%10==0 or number==len(samples):
            result=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=sum(c[0] for c in costs),geometry_encodings=float(np.mean([c[0] for c in costs])),
                    wide_encodings=float(np.mean([c[1] for c in costs])),fine_forwards=0)},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_labels_used_for_online_adaptation=False)
            save(output/'results.json',result)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source,cache,broad,reference,selected,soft,probabilities
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    save(output/'selection_diagnostics.json',dict(samples=choices))
    result.update(status='complete',**check_frozen(frozen,geometry,vip))
    save(output/'results.json',result)


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
