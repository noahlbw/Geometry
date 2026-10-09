"""Matched mask-free production latency on three fixed images per target."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
import eval_development_readout as base
import run_evidence_adaptive_readout as paths
from dinotool.taxonomy_inference import TaxonomyInference
from eval_rival_fine_full import save,frozen_state,check_frozen


@torch.inference_mode()
def main(root,output):
    if output.exists():raise RuntimeError('Existing timing output.')
    output.mkdir(parents=True)
    protocol=paths.read_json(root/'protocol.json');rows={}
    for d in paths.ORDER:
        entry=protocol['datasets'][d]
        args=SimpleNamespace(dataset=d,data_root=entry['data_root'],suite_root=str(root),mode='full',device='cuda',
            original_cache=str(paths.OLD/'text_cache'/(d+'.pt')),dinov3_repo=str(paths.TOOL/'dinov3_hub'),
            checkpoint_dir=str(paths.BASE/'ckpt/DINO'),upstream_root=str(paths.BASE/'third_party/VIP_official_5bd25ee'),
            num_shards=1,shard_index=0,sample_seed=20260923,vdd_ontology='official')
        _,samples,load_image,_=base.load_samples(args,entry)
        needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
        geometry,vip,banks,queries,_=base.load_models(args,entry,needed)
        frozen=frozen_state(geometry,vip);residual=None
        if entry.get('residual_identity') is not None:
            cache=torch.load(root/'residual_cache.pt',map_location='cuda',weights_only=True)
            if cache['identity']!=entry['residual_identity']:raise RuntimeError('Residual cache mismatch.')
            residual=cache['wide']
        model=TaxonomyInference.for_task(geometry,vip,banks,queries,family=entry['family'],
            background=entry['background_index'],residual_features=residual,local_background='union_max')
        values={'Frozen':[],'TwoWitness':[]};memory={k:[] for k in values};keys=[]
        for i in (0,(len(samples)-1)//2,len(samples)-1):
            sample=samples[i];image=load_image(sample);keys.append(sample.key)
            for name,mode in (('Frozen','frozen'),('TwoWitness','two_witness')):
                model.correction=mode;model.predict(image);torch.cuda.synchronize()
            # Alternate order; synchronized batch1 total includes encoding/readout/stitch/CPUargmax.
            for repeat in range(3):
                order=(('Frozen','frozen'),('TwoWitness','two_witness'))
                if repeat%2:order=tuple(reversed(order))
                for name,mode in order:
                    model.correction=mode;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    start=time.perf_counter();_,diag=model.predict(image);torch.cuda.synchronize()
                    values[name].append(1000*(time.perf_counter()-start));memory[name].append(torch.cuda.max_memory_allocated()/1048576)
                    if diag['geometry_encodings']>4 or diag['wide_encodings']>4 or diag['fine_forwards']:raise RuntimeError('Changed encoding budget.')
            del image
        rows[d]=dict(sample_keys=keys,repeats=3,milliseconds=values,
            mean_ms={k:float(np.mean(v)) for k,v in values.items()},peak_allocated_mb={k:max(v) for k,v in memory.items()},
            **check_frozen(frozen,geometry,vip))
        save(output/'results.json',dict(status='running',datasets=rows,target_masks_loaded=False))
        print(d,rows[d]['mean_ms'],flush=True)
        del model,geometry,vip,banks,queries,residual;torch.cuda.empty_cache()
    save(output/'results.json',dict(status='complete',datasets=rows,target_masks_loaded=False,
        scope='Three fixed images per task; per-image warmup and3 alternating synchronized repeats. Batch1 includes encodings,readout,stitch,restoration,CPUargmax; excludes checkpoint/text loading and file IO. Not full-data latency or VIP speed comparison.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
