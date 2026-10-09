"""Mask-free real-image equivalence and synchronized deployment latency."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
import eval_taxonomy_readout as scalar
from dinotool.coverage_soft_alias import coverage_scores
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.taxonomy_readout import natural_profile,rs_profile
from eval_rival_fine_full import frozen_state,check_frozen,save
from run_evidence_adaptive_readout import BASE,TOOL,OLD,idle


@torch.inference_mode()
def scalar_predict(image,model,banks,queries,*,return_probability=False):
    source=base.observations(image,model.geometry,model.vip,model.strengths)
    p,_=rs_profile(source,banks,model.background) if model.profile is None else (model.profile,model.decision)
    broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
    probability=scalar.soft_probability(source,p,banks[p.bank],broad)
    prediction=probability.argmax(0).cpu().numpy()
    return (prediction,probability) if return_probability else prediction


def timed(fn,repeats):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        start=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    return dict(median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),
        repeats=repeats,peak_allocated_mb=torch.cuda.max_memory_allocated()/1048576)


@torch.inference_mode()
def main(args):
    if not idle(args.physical_gpu):
        raise RuntimeError('Benchmark GPU occupied; no sharing with existing workers.')
    output=Path(args.output)
    if output.exists():
        raise RuntimeError('Existing benchmark output.')
    protocol=json.loads((Path(args.suite_root)/'protocol.json').read_text())
    scalar.competition_scores=coverage_scores
    results=[]
    for dataset in ('vdd','potsdam','voc21','context60','ade150'):
        args.dataset=dataset
        entry=protocol['datasets'][dataset]
        args.data_root=entry['data_root']
        args.original_cache=str(OLD/'text_cache'/(dataset+'.pt'))
        _,samples,load_image,_=base.load_samples(args,entry)
        needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
        geometry,vip,banks,queries,_=base.load_models(args,entry,needed)
        frozen=frozen_state(geometry,vip)
        model=TaxonomyInference(geometry,vip,banks,queries,family=entry['family'],background=entry['background_index'])
        selected=[samples[i] for i in np.linspace(0,len(samples)-1,args.samples,dtype=int)]
        rows=[]
        for sample in selected:
            image=load_image(sample)
            source=base.observations(image,geometry,vip,model.strengths)
            p,_=rs_profile(source,banks,model.background) if model.profile is None else (model.profile,model.decision)
            bank=banks[p.bank]
            maximum_difference=0.
            head_scalar,head_packed=[],[]
            for tile in source['local']:
                alias=tile['features'][p.strength].float()@F.normalize(bank.features.float(),dim=-1).T
                reference=coverage_scores(alias,bank)
                packed=model.scorers[p.bank].coverage(alias)
                maximum_difference=max(maximum_difference,float((reference-packed).abs().max()))
                torch.testing.assert_close(reference,packed,rtol=1e-5,atol=1e-6)
                head_scalar.append(timed(lambda:coverage_scores(alias,bank),20))
                head_packed.append(timed(lambda:model.scorers[p.bank].coverage(alias),20))
            del source
            reference,reference_probability=scalar_predict(image,model,banks,queries,return_probability=True)
            prediction,diagnostic,packed_probability=model.predict(image,return_probability=True)
            torch.testing.assert_close(reference_probability,packed_probability,rtol=1e-5,atol=1e-6)
            unequal=int(np.count_nonzero(reference!=prediction))
            gap=reference_probability.topk(2,dim=0).values
            changed=torch.from_numpy(reference!=prediction).to(gap.device)
            maximum_changed_gap=float((gap[0]-gap[1])[changed].max()) if unequal else 0.
            probability_error=float((reference_probability-packed_probability).abs().max())
            if unequal:
                if maximum_changed_gap>2e-6:
                    raise RuntimeError(f'Non-tie prediction mismatch: {dataset}/{sample.key}: {unequal}, gap={maximum_changed_gap}')
            del reference_probability,packed_probability,gap,changed
            # The preceding calls warm all paths. Disk loading/text encoding excluded.
            soft=timed(lambda:model.predict(image),args.repeats)
            model.soft=False
            uniform=timed(lambda:model.predict(image),args.repeats)
            model.soft=True
            original=timed(lambda:scalar_predict(image,model,banks,queries),args.repeats)
            row=dict(key=sample.key,shape=list(image.shape[-2:]),exact_prediction=not unequal,
                changed_pixels=unequal,total_pixels=int(reference.size),
                maximum_changed_probability_gap=maximum_changed_gap,maximum_probability_error=probability_error,
                maximum_alias_score_difference=maximum_difference,diagnostic=diagnostic,
                packed_soft=soft,packed_uniform=uniform,scalar_soft=original,
                scalar_alias_ms=float(np.mean([v['median_ms'] for v in head_scalar])),
                packed_alias_ms=float(np.mean([v['median_ms'] for v in head_packed])))
            rows.append(row)
            print(json.dumps(dict(dataset=dataset,key=sample.key,packed_soft_ms=soft['median_ms'],
                packed_uniform_ms=uniform['median_ms'],scalar_soft_ms=original['median_ms'])),flush=True)
        results.append(dict(dataset=dataset,samples=rows,**check_frozen(frozen,geometry,vip)))
        save(output,dict(status='running',target_masks_loaded=False,results=results))
        del model,geometry,vip,banks,queries
        torch.cuda.empty_cache()
    save(output,dict(status='complete',target_masks_loaded=False,results=results,
        gpu=torch.cuda.get_device_name(),physical_gpu=args.physical_gpu,
        timing='Synchronized warmed deployment including visual encodings, readout, stitching, restoration and CPU argmax transfer. Excludes image disk I/O, model loading and text encoding.',
        scope='Three deterministic spread images per domain; numerical agreement allows near-tie argmax differences (gap<=2e-6). Not bitwise/full mIoU equivalence. Shared GPU power/host contention may affect time.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mode',default='full')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    main(p.parse_args())
