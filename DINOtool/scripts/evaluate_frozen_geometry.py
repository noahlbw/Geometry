"""Full evaluation from public YAML/alias files, with no historical result cache."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from dinotool.frozen_deployment import FrozenGeometry
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save


def main(args):
    output=Path(args.output)
    if output.exists():raise RuntimeError('Use a new output file.')
    model=FrozenGeometry(args.config,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,text_cache=args.text_cache,compact=not args.reference,device=args.device)
    config=model.config
    args.dataset=config['dataset'];args.mode='full'
    entry=dict(config,banks={'original':next(iter(model.vocabulary['banks'].values()))})
    validation,samples,images,masks,names=load_samples(args,entry)
    if len(validation)!=config['total_images']:raise RuntimeError('Dataset split/image count differs from YAML.')
    matrices={p:np.zeros((len(n),len(n)),dtype=np.int64) for p,n in names.items()}
    ignored={p:0 for p in names};started=time.perf_counter()
    for i,sample in enumerate(samples,1):
        image=images(sample)
        for p,n in names.items():
            prediction=model.predict(image,protocol=p if args.dataset=='loveda' else None)
            target=masks(sample,p,tuple(image.shape[-2:]))
            matrices[p]+=confusion(prediction,target,len(n))
            ignored[p]+=int(((target<0)|(target>=len(n))).sum())
        if i%50==0:print(json.dumps(dict(processed=i,total=len(samples))),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    save(output,dict(status='complete',dataset=args.dataset,processed_images=len(samples),
        total_images=len(validation),sample_keys=[s.key for s in samples],
        full_coverage=args.num_shards==1 and len(samples)==len(validation),
        config=config,text_cache_identity=model.cache_identity,compact=not args.reference,
        metrics={p:summary(cm,names[p],ignored[p]) for p,cm in matrices.items()},wall_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data-root','checkpoint-dir','dinov3-repo','upstream-root','text-cache','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--reference',action='store_true')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
