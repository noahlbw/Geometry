"""Standalone warmed whole-input cost of the frozen chosen model, no masks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import eval_development_readout as base
from eval_alias_finalization import load_samples
from eval_kev_alias_search import task_entry,load_models,PREVIOUS,predict
from dinotool.natural_static_search import StaticReader,StaticProfile,projected
from dinotool.taxonomy_inference import TaxonomyInference
from eval_rival_fine_full import save


@torch.inference_mode()
def main(args):
    path=Path(args.output)
    if path.exists():raise RuntimeError('Existing cost output is preserved.')
    path.parent.mkdir(parents=True,exist_ok=True)
    protocol,entry,_=task_entry(args)
    validation,_,loader,_,_=load_samples(args,entry)
    choice=json.loads((Path(args.suite_root)/'search'/args.dataset/'selection.json').read_text())
    profile=StaticProfile(**choice['profile'])
    needed=(('semantic_segmentation',) if entry['family']=='natural' else ('original_imagenet','focused20')) if profile.bank==PREVIOUS else (profile.bank,)
    started=time.perf_counter()
    geometry,vip,banks,queries,checkpoints,residual=load_models(args,entry,needed)
    if profile.bank==PREVIOUS:
        model=TaxonomyInference.for_finalization(geometry,vip,banks,queries,ordinary_alias_policy='uniform',
            family=entry['family'],background=entry['background_index'],residual_features=residual)
        def forward(image):
            probability=model.predict(image,return_probability=True)[2]
            return predict(probability,choice,entry['background_index'])
    else:
        reader=StaticReader(banks,queries,entry['background_index'],residual)
        def forward(image):
            old=base.projected;base.projected=projected
            try:source=base.observations(image,geometry,vip,(profile.strength,),wide_policy=profile.wide_policy)
            finally:base.projected=old
            return predict(reader.probabilities(source,profile,{}),choice,entry['background_index'])
    initialization=time.perf_counter()-started
    counts=dict(geometry=0,wide=0)
    original_geometry=geometry.prepare_image;original_wide=vip.crop_patch_features
    def local(*a,**k):counts['geometry']+=1;return original_geometry(*a,**k)
    def wide(*a,**k):counts['wide']+=1;return original_wide(*a,**k)
    geometry.prepare_image=local;vip.crop_patch_features=wide
    rows=[]
    for i in np.unique(np.linspace(0,len(validation)-1,protocol['cost']['samples'],dtype=int)):
        sample=validation[int(i)];start=time.perf_counter();image=loader(sample);decode=time.perf_counter()-start
        forward(image);forward(image);torch.cuda.synchronize()
        times=[];peaks=[];calls=[]
        for _ in range(protocol['cost']['repeats']):
            counts.update(geometry=0,wide=0);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter();prediction=forward(image);torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000);peaks.append(torch.cuda.max_memory_allocated()/1048576);calls.append(counts.copy())
            if counts['geometry']>4 or counts['wide']>4 or prediction.shape!=tuple(image.shape[-2:]):
                raise RuntimeError('Chosen model violates visual budget or restoration.')
        rows.append(dict(key=sample.key,shape=list(image.shape[-2:]),milliseconds=times,median_ms=float(np.median(times)),
                         peak_allocated_mib=max(peaks),calls=calls,decode_seconds=decode))
    flattened=[t for row in rows for t in row['milliseconds']]
    save(path,dict(status='complete',dataset=args.dataset,images=rows,choice=choice,
        median_ms=float(np.median(flattened)),p95_ms=float(np.quantile(flattened,.95)),
        peak_allocated_mib=max(row['peak_allocated_mib'] for row in rows),initialization_and_text_seconds=initialization,
        target_masks_loaded=False,standalone_process=True,offline_language_cost_excluded_and_reported_separately=True,
        boundary='RGB resize, all encodings, alias aggregation, coupled writeback, original-size restoration and CPU prediction. '
                 'Decode/init/offline language and development search excluded. Seven deterministic images and seven repeats; not full-dataset mean.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','data-root','suite-root','original-cache','dinov3-repo','checkpoint-dir','upstream-root','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',default='full');p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
