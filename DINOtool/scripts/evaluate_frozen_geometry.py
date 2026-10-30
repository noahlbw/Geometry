"""Full evaluation from public YAML/alias files, with no historical result cache."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from dinotool.frozen_deployment import FrozenGeometry
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save


def main(args):
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        import cv2
        cv2.setNumThreads(args.cpu_threads)
    # The pinned VIP checkout exposes a zero-argument tokenizer helper whose
    # default path is empty. Bind it to the frozen DINO BPE file before any
    # DINO text model is constructed. This is part of the reproducible
    # evaluator contract and does not alter model weights or text prompts.
    from dinov3.eval.text import tokenizer as dino_tokenizer
    dino_tokenizer.get_tokenizer = lambda *_a, **_k: dino_tokenizer.Tokenizer(
        vocab_path=str(Path(args.checkpoint_dir) / 'bpe_simple_vocab_16e6.txt.gz'))
    optimized=args.wide_readout=='optimized48'
    candidate=args.wide_readout in ('shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48')
    output=Path(args.output)
    if output.exists():raise RuntimeError('Use a new output file.')
    model=FrozenGeometry(args.config,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,text_cache=args.text_cache,compact=not args.reference,device=args.device,efficient=args.efficient,
        wide_branch=args.wide_branch,query_slot_capacity=48 if optimized else args.query_slot_capacity,writeback=args.writeback,
        shared_visual=optimized,wide_alias_aggregation=args.wide_alias_aggregation)
    forward=model.predict
    if candidate:
        if model.config['dataset']=='loveda':
            raise ValueError('LoveDA P requires its independent foreground reader and is not included in this optimization.')
        from dinotool.geov2_wide_optimized import attach_wide_readout
        forward=attach_wide_readout(model,args.wide_readout)
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
            prediction=forward(image,protocol=p) if args.dataset=='loveda' else forward(image)
            target=masks(sample,p,tuple(image.shape[-2:]))
            matrices[p]+=confusion(prediction,target,len(n))
            ignored[p]+=int(((target<0)|(target>=len(n))).sum())
        if i%50==0:print(json.dumps(dict(processed=i,total=len(samples))),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    save(output,dict(status='complete',dataset=args.dataset,processed_images=len(samples),
        total_images=len(validation),sample_keys=[s.key for s in samples],
        full_coverage=args.num_shards==1 and len(samples)==len(validation),
        config=config,text_cache_identity=model.cache_identity,compact=not args.reference,efficient=args.efficient,
        wide_branch=model.wide_branch,local_branch=model.local_branch,
        query_slot_capacity=getattr(model, 'query_slot_capacity', None),writeback=model.writeback,
        wide_alias_aggregation=model.wide_alias_aggregation,
        wide_readout=args.wide_readout,shared_visual_storage=model.shared_visual,
        shared_backbone_storage=candidate,
        context_memory=model.wide_branch=='raw32',
        context_memory_donors=args.context_memory_donors if model.wide_branch=='raw32' else None,
        context_memory_depth=args.context_memory_depth if model.wide_branch=='raw32' else None,
        metrics={p:summary(cm,names[p],ignored[p]) for p,cm in matrices.items()},wall_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data-root','checkpoint-dir','dinov3-repo','upstream-root','text-cache','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--reference',action='store_true')
    p.add_argument('--efficient',action='store_true',help='Opt in to the paired-verified efficient execution backend.')
    p.add_argument('--wide-branch',choices=('vip','raw32','query_slot'),default=None,
        help='Frozen wide donor branch. Defaults to the incumbent VIP reader; raw32/query_slot are explicit experiments.')
    p.add_argument('--query-slot-capacity',type=int,default=24,
        help='QuerySlot codebook size (screened default: 24); supports compact or efficient execution.')
    p.add_argument('--wide-alias-aggregation',choices=('logsumexp','logmeanexp'),default='logsumexp',
        help='Alias reduction in the QuerySlot/Raw32 wide branch; logmeanexp removes alias-count bias.')
    p.add_argument('--wide-readout',choices=('incumbent','shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48'),default='incumbent',
        help='incumbent keeps the frozen path; shared_canvas shares backbone storage; anchored_infra is the execution-only 24-slot Anchored path; anchored_infra_shared_visual additionally shares the frozen visual head; optimized48 is a 48-slot memory candidate.')
    p.add_argument('--cpu-threads',type=int)
    p.add_argument('--writeback',choices=('coupled','adaptive','equal_mean','local','wide','fixed_g2'),default='coupled',
        help='Explicit writer selection; preserves coupled as the frozen default.')
    p.add_argument('--context-memory',action='store_true',help='Deprecated alias for --wide-branch raw32.')
    p.add_argument('--context-memory-donors',choices=('all','outside'),default='all',
        help='With --context-memory, use all donors or physical local-view exclusion with removed mass returned to self.')
    p.add_argument('--context-memory-depth',choices=('both','final'),default='both',
        help='Apply outside-donor exclusion in both semantic blocks or only the final block.')
    p.add_argument('--context-memory-mode',choices=('memory','response_before'),default='memory',
        help='Use Value memory or transport native attention increments before the frozen MLP.')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    args=p.parse_args()
    if args.wide_readout in ('shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48') and (args.wide_branch!='query_slot' or not args.efficient or args.reference or args.writeback!='coupled'):
        p.error('--wide-readout shared_canvas/anchored_infra/anchored_infra_shared_visual/optimized48 requires --wide-branch query_slot --efficient and coupled writeback.')
    if args.context_memory:
        if args.wide_branch is not None:
            p.error('--context-memory cannot be combined with an explicit --wide-branch.')
        args.wide_branch = 'raw32'
    elif args.wide_branch is None:
        # Keep the frozen incumbent Geometry/VIP path as the default for both
        # compact and efficient execution. Experimental wide branches must be
        # selected explicitly so a deployment cannot silently change accuracy.
        args.wide_branch = 'vip'
    if args.wide_branch != 'raw32' and (args.context_memory_donors!='all' or args.context_memory_depth!='both'):
        p.error('Donor/depth options require --wide-branch raw32.')
    if args.context_memory_donors=='all' and args.context_memory_depth!='both':
        p.error('Final-block exclusion requires --context-memory-donors outside.')
    if args.context_memory_mode=='response_before' and (args.wide_branch!='raw32' or args.context_memory_donors!='all' or args.context_memory_depth!='both'):
        p.error('response_before requires all donors and both blocks.')
    main(args)
