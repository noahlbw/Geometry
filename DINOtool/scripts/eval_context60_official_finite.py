"""Frozen official-query PC60 comparator; no Geometry or alias search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import eval_development_readout as base
from eval_natural_text_adaptation import official_options, official_prediction
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT
from eval_rival_fine_full import save
from eval_geometry_vip_reliability import summary
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.vip_official_adapter import VIPQueries, upstream_aliases, _load_upstream
from dinotool.inference import tile_starts

METHODS=('VIP_Official_Finite','VIP_Official_NoThreshold')


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; inspect instead of relaunching.')
    entry=json.loads((root/'protocol.json').read_text())['datasets'][args.dataset]
    validation,samples,load_image,load_mask=base.load_samples(args,entry)
    if len(validation)!=5105:
        raise RuntimeError('Incomplete PC60 cohort.')
    upstream=Path(args.upstream_root)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit!=PINNED_COMMIT:
        raise RuntimeError('Upstream commit differs.')
    names=tuple(c['name'] for c in entry['banks']['semantic_segmentation']['classes'])
    groups=upstream_aliases(upstream/'configs/cls_context60.txt')
    # The prepared natural loader uses the official contiguous PC60 order.
    if len(groups)!=60 or len(names)!=60 or names[0]!='background':
        raise RuntimeError('Official class order/count invalid.')
    settings,family=official_options(upstream,args.dataset)
    checkpoints=make_checkpoints(args)
    vip=FiniteVIPObserver(DINOTextSegmenter(checkpoints,device=args.device),upstream)
    identity=json.loads(json.dumps(dict(checkpoints=checkpoint_manifest(checkpoints),groups=groups,
        classes=names,upstream_commit=commit,template=family,settings=asdict(settings))))
    cache=root/'official_text.pt'
    if args.mode=='smoke':
        if cache.exists():
            raise RuntimeError('Existing smoke cache.')
        prompt=_load_upstream(upstream/'prompts/imagenet_template.py','pc60_official_prompts')
        vip.templates=prompt.get_text_template(family)
        query=vip.encode_queries(names,groups)
        torch.save(dict(identity=identity,features=query.features.cpu(),parents=query.parents.cpu()),cache)
    else:
        encoded=torch.load(cache,map_location=args.device,weights_only=True)
        if encoded['identity']!=identity:
            raise RuntimeError('Frozen official text differs.')
        query=VIPQueries(encoded['features'],encoded['parents'],names,tuple(a for g in groups for a in g))
    head={n:t.clone() for n,t in vip.backbone.model.visual_model.head.state_dict().items()}
    def frozen():
        return dict(weights_frozen=all(not p.requires_grad for p in vip.backbone.model.parameters()),
            head_weights_unchanged=all(torch.equal(t,vip.backbone.model.visual_model.head.state_dict()[n]) for n,t in head.items()))
    signature=dict(implementation='context60-official-finite-comparator-v1-20261007',dataset=args.dataset,
        methods=METHODS,classes={args.dataset:names},gear=dict(preprocessing='short-edge336/max-long2048/stride224',
            numerical_repair='Self-Value only for all-masked proxy rows'),competitive=None,vocabulary=identity,
        checkpoints=checkpoint_manifest(checkpoints),global_sample_count=len(validation),
        global_sample_keys_sha256=base.digest([s.key for s in validation]),sample_keys=[s.key for s in samples],
        sample_keys_sha256=base.digest([s.key for s in samples]),num_shards=args.num_shards,
        shard_index=args.shard_index,config=vars(args))
    output.mkdir(parents=True)
    matrices={m:np.zeros((60,60),np.int64) for m in METHODS}
    per_image={m:[] for m in METHODS}
    started=time.perf_counter()
    tiles=0
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        calibrated,plain=official_prediction(image,vip,query,settings)
        if calibrated.shape!=tuple(image.shape[-2:]) or plain.shape!=calibrated.shape:
            raise RuntimeError('Prediction not restored.')
        if args.mode=='smoke':
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,official_query_identity=identity,**frozen()))
            return
        target=load_mask(sample,image.shape[-2:])
        for method,prediction in zip(METHODS,(calibrated,plain)):
            cm=base.confusion(prediction,target,60)
            matrices[method]+=cm
            per_image[method].append(cm)
        h,w=image.shape[-2:]
        ratio=min(336/min(h,w),2048/max(h,w))
        tiles+=len(tile_starts(int(h*ratio+.5),336,224))*len(tile_starts(int(w*ratio+.5),336,224))
        if number==1 or number%10==0 or number==len(samples):
            row=dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=tiles)},wall_seconds=time.perf_counter()-started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row)
            print(json.dumps(dict(processed=number,total=len(samples))),flush=True)
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    row.update(status='complete',**frozen())
    save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--dataset',default='context60',choices=('context60',))
    p.add_argument('--mode',required=True,choices=('smoke','full'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    main(p.parse_args())
