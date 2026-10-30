"""Run a frozen YAML model on an RGB image without historical experiment files."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from dinotool.frozen_deployment import FrozenGeometry


def main(args):
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        import cv2
        cv2.setNumThreads(args.cpu_threads)
    optimized=args.wide_readout=='optimized48'
    candidate=args.wide_readout in ('shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48')
    model=FrozenGeometry(args.config,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,text_cache=args.text_cache,compact=not args.reference,device=args.device,efficient=args.efficient,
        wide_branch=args.wide_branch,query_slot_capacity=48 if optimized else args.query_slot_capacity,writeback=args.writeback,
        shared_visual=optimized,wide_alias_aggregation=args.wide_alias_aggregation)
    forward=model.predict
    if candidate:
        from dinotool.geov2_wide_optimized import attach_wide_readout
        forward=attach_wide_readout(model,args.wide_readout)
    array=np.asarray(Image.open(args.input).convert('RGB')).copy()
    image=torch.from_numpy(array).permute(2,0,1).float()/255
    prediction=forward(image)
    output=Path(args.output)
    if output.exists():raise RuntimeError('Preserve existing prediction.')
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(prediction).save(output)
    print(json.dumps(dict(output=str(output),shape=list(prediction.shape),cache_identity=model.cache_identity,
        compact=not args.reference,efficient=args.efficient,wide_branch=model.wide_branch,local_branch=model.local_branch,
        query_slot_capacity=getattr(model, 'query_slot_capacity', None),writeback=model.writeback,
        wide_alias_aggregation=model.wide_alias_aggregation,
        wide_readout=args.wide_readout,shared_visual_storage=model.shared_visual,
        shared_backbone_storage=candidate,
        context_memory=model.wide_branch=='raw32',
        context_memory_donors=args.context_memory_donors if model.wide_branch=='raw32' else None,
        context_memory_depth=args.context_memory_depth if model.wide_branch=='raw32' else None,target_masks_loaded=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint-dir','dinov3-repo','upstream-root','text-cache','input','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--reference',action='store_true')
    parser.add_argument('--efficient',action='store_true',help='Opt in to the paired-verified efficient execution backend.')
    parser.add_argument('--wide-branch',choices=('vip','raw32','query_slot'),default=None,
        help='Frozen wide donor branch. Defaults to the incumbent VIP reader; raw32/query_slot are explicit experiments.')
    parser.add_argument('--query-slot-capacity',type=int,default=24,
        help='QuerySlot codebook size (screened default: 24); supports compact or efficient execution.')
    parser.add_argument('--wide-alias-aggregation',choices=('logsumexp','logmeanexp'),default='logsumexp',
        help='Alias reduction in the QuerySlot/Raw32 wide branch; logmeanexp removes alias-count bias.')
    parser.add_argument('--wide-readout',choices=('incumbent','shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48'),default='incumbent',
        help='incumbent keeps the frozen path; shared_canvas shares backbone storage; anchored_infra is the execution-only 24-slot Anchored path; anchored_infra_shared_visual additionally shares the frozen visual head; optimized48 is a 48-slot memory candidate.')
    parser.add_argument('--cpu-threads',type=int,help='Bound CPU preprocessing threads for reproducible latency.')
    parser.add_argument('--writeback',choices=('coupled','adaptive','equal_mean','local','wide','fixed_g2'),default='coupled',
        help='Explicit writer selection; adaptive is an evaluated candidate, not an automatic replacement of the frozen writer.')
    parser.add_argument('--context-memory',action='store_true',help='Deprecated alias for --wide-branch raw32.')
    parser.add_argument('--context-memory-donors',choices=('all','outside'),default='all',
        help='With --context-memory, use all donors or physical local-view exclusion with removed mass returned to self.')
    parser.add_argument('--context-memory-depth',choices=('both','final'),default='both',
        help='Apply outside-donor exclusion in both semantic blocks or only the final block.')
    parser.add_argument('--context-memory-mode',choices=('memory','response_before'),default='memory',
        help='Use Value memory or transport native attention increments before the frozen MLP.')
    args=parser.parse_args()
    if args.wide_readout in ('shared_canvas','anchored_infra','anchored_infra_shared_visual','optimized48') and (args.wide_branch!='query_slot' or not args.efficient or args.reference or args.writeback!='coupled'):
        parser.error('--wide-readout shared_canvas/anchored_infra/anchored_infra_shared_visual/optimized48 requires --wide-branch query_slot --efficient and coupled writeback.')
    if args.context_memory:
        if args.wide_branch is not None:
            parser.error('--context-memory cannot be combined with an explicit --wide-branch.')
        args.wide_branch = 'raw32'
    elif args.wide_branch is None:
        # Keep the frozen incumbent Geometry/VIP path as the default for both
        # compact and efficient execution. Experimental wide branches must be
        # selected explicitly so a deployment cannot silently change accuracy.
        args.wide_branch = 'vip'
    if args.wide_branch != 'raw32' and (args.context_memory_donors!='all' or args.context_memory_depth!='both'):
        parser.error('Donor/depth options require --wide-branch raw32.')
    if args.context_memory_donors=='all' and args.context_memory_depth!='both':
        parser.error('Final-block exclusion requires --context-memory-donors outside.')
    if args.context_memory_mode=='response_before' and (args.wide_branch!='raw32' or args.context_memory_donors!='all' or args.context_memory_depth!='both'):
        parser.error('response_before requires all donors and both blocks.')
    main(args)
