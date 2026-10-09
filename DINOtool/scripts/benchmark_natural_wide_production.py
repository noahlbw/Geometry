"""Single-view deployment timing and mask-free replay of the frozen scale prior."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import eval_development_readout as base
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.natural_wide_resolution import replace_wide
from eval_rival_fine_full import save,frozen_state,check_frozen
from run_region_semantic_suite_a800 import idle


@torch.inference_mode()
def main(args):
    if not idle(args.physical_gpu):raise RuntimeError('GPU occupied; do not interfere.')
    root,output=Path(args.suite_root),Path(args.output)
    if output.exists():raise RuntimeError('Existing timing output.')
    protocol=json.loads((root/'protocol.json').read_text());result={}
    for d in ('voc21','context60','ade150'):
        args.dataset=d;entry=protocol['datasets'][d];args.data_root=entry['data_root'];args.mode='full'
        validation,_,load_image,_=base.load_samples(args,entry)
        args.original_cache=str(root.parent/'curated20_patchonly2_full_20261006'/'text_cache'/(d+'.pt'))
        geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,('semantic_segmentation',))
        frozen=frozen_state(geometry,vip);residual=None
        if entry.get('residual_identity') is not None:
            row=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
            if row['identity']!=entry['residual_identity']:raise RuntimeError('Residual identity differs.')
            residual=row['wide']
        common=dict(family='natural',background=entry['background_index'],soft=False,residual_features=residual,
            residual_mode='protected' if residual is not None else 'max')
        models={p:TaxonomyInference(geometry,vip,banks,queries,wide_policy=p,**common) for p in ('long448','natural_short336_cap672')}
        records=[]
        for index in (0,len(validation)//2,len(validation)-1):
            sample=validation[index];image=load_image(sample)
            # Source replay is outside measured timing. No masks are loaded.
            source=base.observations(image,geometry,vip,models['long448'].strengths)
            changed=replace_wide(source,image,vip)
            checks={}
            for p,reference_source in zip(models,(source,changed)):
                model=models[p]
                _,_,reference=model.predict_observations(reference_source,return_probability=True)
                _,diagnostic,actual=model.predict(image,return_probability=True)
                torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                if diagnostic['wide_encodings']>4 or diagnostic['geometry_encodings']>4 or diagnostic['fine_forwards']:
                    raise RuntimeError('Single-view production budget exceeded.')
                checks[p]=dict(probability_max_difference=float((actual-reference).abs().max()),diagnostic=diagnostic)
                del reference,actual
            del source,changed,reference_source
            timings={};peaks={}
            for p,model in models.items():
                model.predict(image);torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();elapsed=[]
                for _ in range(3):
                    torch.cuda.synchronize();started=time.perf_counter();model.predict(image);torch.cuda.synchronize()
                    elapsed.append(1000*(time.perf_counter()-started))
                timings[p]=dict(repeats_ms=elapsed,median_ms=statistics.median(elapsed))
                peaks[p]=torch.cuda.max_memory_allocated()/1048576
            records.append(dict(key=sample.key,checks=checks,timings=timings,peak_memory_mib=peaks))
        result[d]=dict(images=records,mean_image_median_ms={p:statistics.mean(r['timings'][p]['median_ms'] for r in records) for p in models},
            **check_frozen(frozen,geometry,vip))
        del models,model,geometry,vip,banks,queries,residual,frozen;torch.cuda.empty_cache()
    save(output,dict(target_masks_loaded=False,physical_gpu=args.physical_gpu,results=result,
        protocol='Three fixed-spread images/domain; one warm pass then three synchronized repeats. Includes encoding/readout/stitch/restoration/CPU argmax, excludes image loading/text/cache/disk. Not full-data latency or matched VIP timing. No count correction or envelope.'))
    print(json.dumps({d:r['mean_image_median_ms'] for d,r in result.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('suite-root','output','dinov3-repo','checkpoint-dir','upstream-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    main(p.parse_args())
