"""Fresh vocabulary encoding versus the historical frozen text cache, one RGB."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch

from eval_kev_alias_search import task_entry,load_models,predict
from eval_alias_finalization import load_samples,foreground_banks
from dinotool.natural_static_search import StaticReader,StaticProfile
from dinotool.compact_deployment import CompactGeometry,compact_observations,CachedWideScorer,cached_wide
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.taxonomy_inference import TaxonomyInference
from eval_rival_fine_full import save


@torch.inference_mode()
def main(args):
    output=Path(args.output)
    if output.exists():raise RuntimeError('Preserve existing smoke output.')
    _,entry,_=task_entry(args)
    validation,_,loader,_,_=load_samples(args,entry)
    choice=json.loads((Path(args.suite_root)/'search'/args.dataset/'selection.json').read_text())
    profile=StaticProfile(**choice['profile'])
    previous=profile.bank=='__previous_final__'
    bankname='semantic_segmentation' if previous else profile.bank
    geometry,vip,banks,queries,_,residual=load_models(args,entry,(bankname,))
    image=loader(validation[0])
    retained=TaxonomyInference.for_finalization(geometry,vip,banks,queries,ordinary_alias_policy='uniform',
        family=entry['family'],background=entry['background_index'],residual_features=residual) if previous else None
    source=compact_observations(image,CompactGeometry(geometry),vip,retained.strengths if previous else (profile.strength,),wide_policy=profile.wide_policy)
    reader=StaticReader(banks,queries,entry['background_index'],residual)
    with cached_wide(CachedWideScorer()):
        probability=retained.predict_observations(source,return_probability=True)[2] if previous else reader.probabilities(source,profile,{})
        reference=predict(probability,choice,entry['background_index'])
        reference_p=None
        if args.dataset=='loveda':
            bs,qs=foreground_banks(banks,queries)
            p_reader=StaticReader(bs,qs,None,None)
            reference_p=predict(p_reader.probabilities(source,profile,{}),choice,None)
            del bs,qs,p_reader
    expected_local=banks[bankname].features.cpu();expected_wide=queries[bankname].features.cpu()
    del geometry,vip,banks,queries,residual,reader,source,retained,probability
    gc.collect();torch.cuda.empty_cache()
    fresh=not Path(args.text_cache).exists()
    model=FrozenGeometry(args.config,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,text_cache=args.text_cache)
    local_error=float((model.banks[bankname].features.cpu()-expected_local).abs().max())
    wide_error=float((model.queries[bankname].features.cpu()-expected_wide).abs().max())
    actual=model.predict(image)
    p_check={}
    if reference_p is not None:
        actual_p=model.predict(image,protocol='P')
        p_check=dict(P_changed_pixels=int(np.count_nonzero(reference_p!=actual_p)),P_exact_prediction=np.array_equal(reference_p,actual_p))
    output.parent.mkdir(parents=True,exist_ok=True)
    save(output,dict(status='complete',dataset=args.dataset,key=validation[0].key,
        fresh_text_encoding=fresh,local_vector_max_error=local_error,wide_vector_max_error=wide_error,
        changed_pixels=int(np.count_nonzero(actual!=reference)),pixels=int(actual.size),
        exact_prediction=np.array_equal(actual,reference),target_masks_loaded=False,**p_check))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','data-root','suite-root','original-cache','dinov3-repo','checkpoint-dir','upstream-root','output','config','text-cache'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',default='full');p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
