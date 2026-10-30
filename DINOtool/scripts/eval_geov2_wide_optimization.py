"""Evaluate opt-in wide/readout changes with the local DonorBefore frozen."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.frozen_deployment import FrozenGeometry
from dinotool.geov2_wide_optimized import (
    OptimizedWideReadout, WideReadoutConfig, factor_reconstruction, factor_correction,
)
from dinotool.anchored_infra import configure_fast_math, cast_shared_visual_weights
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import metric, write


VARIANTS = {
    'baseline': None,
    'shared_canvas': WideReadoutConfig(direct=False),
    'dual_direct': WideReadoutConfig(shared_backbone=False),
    'shared_direct': WideReadoutConfig(),
    'shared_native': WideReadoutConfig(head='native'),
    'shared_regression': WideReadoutConfig(head='regression'),
    'shared_short336': WideReadoutConfig(wide_long_edge=336),
    # Resolution and execution-only controls are combined here to isolate
    # the VDD latency bottleneck.  DonorBefore/local geometry is untouched.
    'shared_short336_fast': WideReadoutConfig(
        wide_long_edge=336, fast_assignment=True, solve_dtype='float32'),
    'shared_short224_fast': WideReadoutConfig(
        wide_long_edge=224, fast_assignment=True, solve_dtype='float32'),
    'shared_batch2': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=2),
    'shared_batch2_short336': WideReadoutConfig(
        wide_long_edge=336, fast_assignment=True, solve_dtype='float32',
        local_batch_size=2),
    'shared_batch4': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=4),
    'shared_batch2_tf32': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=2),
    'shared_batch2_compile': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=2),
    'shared_batch4_compile': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=4),
    'shared_batch4_compile_all': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=4),
    'shared_native_batch4_compile': WideReadoutConfig(
        head='native', fast_assignment=True, solve_dtype='float32', local_batch_size=4),
    'shared_regression_batch4_compile': WideReadoutConfig(
        head='regression', fast_assignment=True, solve_dtype='float32', local_batch_size=4),
    'shared_local_memory_batch4_compile': WideReadoutConfig(
        fast_assignment=True, solve_dtype='float32', local_batch_size=4,
        local_memory=True),
    'shared_fp32': WideReadoutConfig(solve_dtype='float32'),
    'shared_slots48': WideReadoutConfig(capacity=48, solve_dtype='float32'),
    'shared_visual': WideReadoutConfig(shared_visual=True, solve_dtype='float32'),
    'optimized48': WideReadoutConfig(capacity=48, shared_visual=True, fast_assignment=True),
    'optimized24': WideReadoutConfig(capacity=24, shared_visual=True, fast_assignment=True),
    'shared_visual_canvas': WideReadoutConfig(shared_visual=True, direct=False),
    # Frozen Anchored deployment: 24 slots, independent wide head, shared
    # backbone storage, fast assignment, and factorized FP32 H action.
    'anchored_infra': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored'),
    # Batch only the independent fixed-size local crops and the Anchored
    # readout.  This keeps the visual/project kernels eager, so the local
    # DonorBefore path remains bitwise stable while reducing launch overhead.
    'anchored_infra_batched': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=4),
    # Reuse the already-computed local DonorBefore tiles to build the compact
    # global memory.  This removes the second full-image visual forward while
    # leaving the local path and the Anchored writer unchanged.  It is kept
    # as a separately audited candidate because the wide observation changes.
    'anchored_infra_local_memory': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=4,
        local_memory=True),
    # Opt-in BF16 frozen-weight storage.  The default Anchored infra arm
    # remains unchanged; this candidate is screened for memory/latency and
    # must pass the smoke numerical tolerance before full evaluation.
    'anchored_infra_bf16': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, weight_dtype='bfloat16'),
    # BF16 frozen storage plus shape-static compiler kernels.  Local crops
    # remain batch-one because the pinned upstream attention implementation
    # assumes B=1; this arm isolates compile/TF32 gains safely.
    'anchored_infra_bf16_compile': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=1,
        compile_visual=True, compile_project=True, weight_dtype='bfloat16'),
    # Execution-only candidate: same Anchored/DonorBefore equations as the
    # frozen deployment, but batch the fixed 512px local crops and compile
    # the shape-static visual/project kernels.  This is intentionally a
    # separate variant so its cost and parity can be audited independently.
    'anchored_infra_fast': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=4,
        compile_visual=True, compile_project=True),
    'anchored_infra_fast_bf16': WideReadoutConfig(
        shared_backbone=True, shared_visual=False, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=4,
        compile_visual=True, compile_project=True, weight_dtype='bfloat16'),
    'anchored_infra_fast_shared_bf16': WideReadoutConfig(
        shared_backbone=True, shared_visual=True, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored', batch_anchored=True, local_batch_size=4,
        compile_visual=True, compile_project=True, weight_dtype='bfloat16'),
    'anchored_infra_shared_visual': WideReadoutConfig(
        shared_backbone=True, shared_visual=True, direct=True,
        capacity=24, fast_assignment=True, solve_dtype='float32',
        readout='anchored'),
    'ridge01': WideReadoutConfig(head='regression', regression_ridge=.1, shared_visual=True, fast_assignment=True),
    'ridge03': WideReadoutConfig(head='regression', regression_ridge=.3, shared_visual=True, fast_assignment=True),
    'ridge1': WideReadoutConfig(head='regression', regression_ridge=1., shared_visual=True, fast_assignment=True),
}


@torch.inference_mode()
def main(args):
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        import cv2
        cv2.setNumThreads(args.cpu_threads)
    root = Path(args.experiment)
    protocol = json.loads((root/'optimization_protocol.json').read_text())
    record = protocol['datasets'][args.dataset]
    if args.gain is not None:
        record=dict(record,gain=args.gain)
    if not record.get('cost_keys'):
        # Older protocols kept the fixed cost mapping at the root.  Accept
        # both layouts while keeping cost selection explicit and reproducible.
        record = dict(record, cost_keys=protocol.get('cost_keys', {}).get(args.dataset, []))
    out = (root if args.output_root is None else Path(args.output_root))/args.mode/args.variant/(args.dataset+f'_s{args.shard_index}')
    if out.exists():
        raise RuntimeError('Preserve existing output: '+str(out))
    out.mkdir(parents=True)
    # The protocol keeps dataset loader fields under ``loader`` (the same
    # layout used by the frozen evaluation scripts).  Materialize them on the
    # argparse namespace before delegating to the canonical sample loader.
    for key, value in record.get('loader', {}).items():
        setattr(args, key, value)
    from dinov3.eval.text import tokenizer
    tokenizer.get_tokenizer = lambda *_a,**_k: tokenizer.Tokenizer(
        vocab_path=str(Path(protocol['checkpoint_dir'])/'bpe_simple_vocab_16e6.txt.gz'))
    config=VARIANTS[args.variant]
    if args.variant.endswith('_tf32') or config.weight_dtype == 'bfloat16':
        configure_fast_math(True)
    torch.cuda.reset_peak_memory_stats()
    m = FrozenGeometry(record['config_path'], checkpoint_dir=protocol['checkpoint_dir'],
        dinov3_repo=protocol['dinov3_repo'],upstream_root=protocol['upstream_root'],
        text_cache=record['text_cache'],efficient=True,wide_branch='query_slot',query_slot_capacity=24,
        shared_visual=config is not None and config.shared_visual,
        wide_alias_aggregation=args.wide_alias_aggregation)
    if config is not None and config.local_batch_size > 1:
        from dinotool.batched_attention import enable_batched_attention
        enable_batched_attention()
    if args.variant.endswith('_compile') or args.variant.endswith('_compile_all'):
        # Compile only the frozen visual feature extractor. Geometry relation,
        # DonorBefore, slot routing and writeback remain eager and unchanged.
        for branch in (m.geometry.backbone, m.vip.backbone):
            visual = branch.model.visual_model
            visual.get_backbone_features = torch.compile(
                visual.get_backbone_features, mode='reduce-overhead',
                fullgraph=False, dynamic=False)
    if args.variant.endswith('_compile_all'):
        # The projection head is also shape-static for the locked 512 crop.
        # Compile it separately; all tensor equations and the local path stay
        # identical, while Python/module dispatch is removed from each tile.
        m.compact_geometry.project = torch.compile(
            m.compact_geometry.project, mode='max-autotune-no-cudagraphs',
            fullgraph=False, dynamic=False)
    initialization_peak_mib=torch.cuda.max_memory_allocated()/1048576
    if m.strengths != ('original',):
        raise ValueError('Local DonorBefore must stay active.')
    m.profile = replace(m.profile,coupling=record['gain'])
    original_local_backbone = m.geometry.backbone.model.visual_model.backbone
    original_local_geometry = m.compact_geometry
    args.data_root = record['loader']['data_root']
    args.sample_seed,args.vdd_ontology = record['loader'].get('sample_seed', 20260923), record['loader'].get('vdd_ontology', 'official')
    loader_args = argparse.Namespace(**vars(args))
    loader_args.mode = 'full'
    loader_args.num_shards,loader_args.shard_index = 1,0
    entry=dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation,_,images,masks,names_map=load_samples(loader_args,entry)
    if len(validation) != m.config['total_images']:
        raise ValueError('Validation coverage changed.')
    if args.mode=='cost':
        lookup={s.key:s for s in validation}
        if not record['cost_keys']:
            raise ValueError(f'No fixed cost keys for {args.dataset}')
        samples=[lookup[k] for k in record['cost_keys']]
    elif args.mode in ('screen','smoke'):
        count=1 if args.mode=='smoke' else min(record.get('screen_images',protocol['screen_images']),len(validation))
        lookup={s.key:s for s in validation}
        if record.get('sample_keys'):
            samples=[lookup[k] for k in record['sample_keys'][:count]]
        else:
            samples=[validation[i] for i in np.linspace(0,len(validation)-1,count,dtype=int)]
    else:
        samples=validation[args.shard_index::args.num_shards]
    invariant={}
    if args.mode=='smoke':
        from dinotool.bounded_physical_coupling import resize_geometry
        from dinotool.gear_ov import _crop_at
        rgb=_crop_at(resize_geometry(images(samples[0])),0,0,512).to(m.geometry.device)
        prepared=m.compact_geometry.prepare_image(rgb)
        from dinotool.context_memory_observer import context_assignments
        from dinotool.geov2_wide_optimized import wide_assignments
        raw=prepared.backbone_tokens[:,prepared.prefix_tokens:]
        assignment_config=replace(m.query_slot_observer.config,capacity=48)
        a_old=context_assignments(raw,assignment_config)
        a_fast=wide_assignments(raw,assignment_config)
        invariant['assignment_max_abs']=float((a_old-a_fast).abs().max())
        invariant['assignment_argmax_mismatch_fraction']=float(
            (a_old.argmax(-1) != a_fast.argmax(-1)).float().mean())
        # ``wide_assignments`` is an execution kernel.  BF16/autocast inputs
        # can reassociate the spherical Lloyd scores even when the selected
        # slots are unchanged.  Record both diagnostics and let screen/full
        # mIoU decide the opt-in arm; the frozen default remains audited by
        # the exact unit test and the FP64 compatibility path.
        del raw,a_old,a_fast
        with m.geometry.backbone._autocast():
            old_local=m.compact_geometry.project(prepared,'original')
        valid=torch.ones(len(prepared.geometry_patch_conditional[0]),device=rgb.device,dtype=torch.bool)
        from dinotool.efficient_deployment import reconstruction
        relation=prepared.geometry_patch_conditional[0]
        operator,_=reconstruction(relation,valid)
        factors,_=factor_reconstruction(relation,valid)
        difference=torch.randn((len(valid),7),device=rgb.device,dtype=torch.float64)
        error=(operator@difference-factor_correction(factors,difference)).abs().max().item()
        if error>1e-10:
            raise RuntimeError('Quadratic factor action does not match H: '+str(error))
        invariant['factor_action_max_abs']=error
    optimized=None if config is None else OptimizedWideReadout(m,config)
    if m.geometry.backbone.model.visual_model.backbone is not original_local_backbone or m.compact_geometry is not original_local_geometry:
        raise RuntimeError('Local path object changed.')
    if args.mode=='smoke':
        if config is not None and config.solve_dtype == 'float32':
            f32,_=factor_reconstruction(relation,valid,torch.float32)
            error32=(operator@difference-factor_correction(f32,difference).double()).abs().max().item()
            invariant['fp32_factor_action_max_abs']=error32
            if error32>5e-4:
                raise RuntimeError('FP32 factor action exceeds tolerance: '+str(error32))
            del f32
        if optimized is not None:
            optimized.wide_memory(images(samples[0]))
        prepared_after=m.compact_geometry.prepare_image(rgb)
        with m.geometry.backbone._autocast():
            after=m.compact_geometry.project(prepared_after,'original')
        invariant['local_features_equal']=torch.equal(old_local,after)
        # Compiled kernels can reassociate floating point reductions while
        # preserving the same DonorBefore computation.  Keep the strict
        # bitwise gate for eager/default arms, and use a bounded numerical
        # parity gate for the opt-in compiled arm.
        if config is None or (config.weight_dtype == 'none' and
                              not (config.compile_visual or config.compile_project)):
            if not invariant['local_features_equal']:
                raise RuntimeError('Local DonorBefore changed.')
        elif config.weight_dtype == 'none':
            delta=(old_local.float()-after.float()).abs()
            old_norm=old_local.float().norm(dim=-1).clamp_min(1e-12)
            cosine=(old_local.float()*after.float()).sum(-1)/(old_norm*after.float().norm(dim=-1).clamp_min(1e-12))
            invariant['local_features_max_abs']=float(delta.max())
            invariant['local_features_mean_cosine']=float(cosine.mean())
            invariant['local_features_close']=bool(
                torch.isfinite(delta).all() and torch.isfinite(cosine).all()
                and delta.max() <= 2e-4 and cosine.mean() >= .9999)
            if not invariant['local_features_close']:
                raise RuntimeError('Compiled local DonorBefore deviation exceeds smoke tolerance: '
                                   +str(invariant['local_features_max_abs']))
        else:
            # BF16 is an explicitly approximate storage arm.  Check that the
            # local path remains finite and close to the FP32 reference while
            # keeping the exact equality gate for the default arm.
            delta=(old_local.float()-after.float()).abs()
            old_norm=old_local.float().norm(dim=-1).clamp_min(1e-12)
            cosine=(old_local.float()*after.float()).sum(-1)/(old_norm*after.float().norm(dim=-1).clamp_min(1e-12))
            invariant['local_features_max_abs']=float(delta.max())
            invariant['local_features_mean_cosine']=float(cosine.mean())
            invariant['local_features_close']=bool(
                torch.isfinite(delta).all() and torch.isfinite(cosine).all()
                and delta.max() <= 5e-2 and cosine.mean() >= .995)
            if not invariant['local_features_close']:
                raise RuntimeError('BF16 local DonorBefore deviation exceeds smoke tolerance.')
        del prepared,prepared_after,rgb,old_local,after,relation,operator,factors,difference
    forward=m.predict if optimized is None else optimized.predict
    common=dict(dataset=args.dataset,variant=args.variant,wide_config=None if config is None else asdict(config),
                wide_alias_aggregation=args.wide_alias_aggregation,
                gain=record['gain'],profile=m.profile.record(),config=m.config,cache_identity=m.cache_identity,
                local_path_unchanged=True,invariants=invariant,cpu_threads=torch.get_num_threads(),
                initialization_peak_allocated_mib=initialization_peak_mib,shared_visual_storage=m.shared_visual,
                weight_storage=None if optimized is None else optimized.weight_storage)
    if args.mode=='cost':
        gc.collect()
        torch.cuda.empty_cache()
        rows=[]
        for sample in samples:
            image=images(sample)
            for _ in range(2):
                forward(image)
            times,allocated,reserved=[],[],[]
            for _ in range(7):
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start=time.perf_counter()
                result=forward(image)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                allocated.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            if result.dtype!=np.uint8 or result.shape!=tuple(image.shape[-2:]):
                raise RuntimeError('Cost output shape/dtype differs.')
            rows.append(dict(key=sample.key,milliseconds=times,peak_allocated_mib=max(allocated),peak_reserved_mib=max(reserved)))
        write(out/'results.json',dict(status='complete',**common,images=rows,
            median_ms=float(np.median([t for r in rows for t in r['milliseconds']])),
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows),gpu=torch.cuda.get_device_name(),
            target_masks_loaded=False,boundary='CPU RGB to original-size CPU uint8 labels; excludes decode/init/text. Seven fixed images, two warmups, seven synchronized repeats. Fresh process for each variant.'))
        return
    names=names_map[args.dataset]
    matrix=np.zeros((len(names),len(names)),np.int64)
    keys,tps,predicted,targets=[],[],[],[]
    start=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=images(sample)
        prediction=forward(image)
        target=masks(sample,args.dataset,tuple(image.shape[-2:]))
        cm=confusion(prediction,target,len(names))
        matrix+=cm
        keys.append(sample.key)
        tps.append(cm.diagonal().copy())
        predicted.append(cm.sum(0))
        targets.append(cm.sum(1))
        if number==1 or number%25==0 or number==len(samples):
            progress=dict(dataset=args.dataset,variant=args.variant,processed=number,total=len(samples),wall_seconds=time.perf_counter()-start)
            write(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    np.savez_compressed(out/'statistics.npz',sample_keys=np.asarray(keys),tp=np.stack(tps),predicted=np.stack(predicted),target=np.stack(targets))
    write(out/'results.json',dict(status='complete',**common,processed_images=len(keys),global_sample_count=len(validation),
        sample_keys=keys,expected_full_keys=[s.key for s in validation],shard_index=args.shard_index,num_shards=args.num_shards,
        class_names=names,metrics=dict(mean_iou_percent=metric(matrix),confusion_matrix=matrix.tolist()),
        masks_used_only_for_scoring=True,wall_seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS),required=True)
    p.add_argument('--mode',choices=('smoke','screen','full','cost'),required=True)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--output-root')
    p.add_argument('--gain',type=float)
    p.add_argument('--wide-alias-aggregation',choices=('logsumexp','logmeanexp'),default='logsumexp',
        help='Alias reduction in the QuerySlot wide branch; logmeanexp removes alias-count bias.')
    main(p.parse_args())
