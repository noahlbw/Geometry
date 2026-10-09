"""Frozen full-domain accuracy and paired singleton cost; no model-rule edits."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.bounded_patch_only import PRIMARY as BASELINE, predict_image as baseline_predict
from dinotool.alias_comparison_memory import execution as memory_execution, cached_observations
from dinotool.geometry_execution import GeometryExecution
from dinotool.model import checkpoint_manifest
from dinotool.one_sided_alias_stream import PRIMARY as RETAINED, predict_image as retained_predict
from dinotool.shared_local_alias import PRIMARY as LIGHT, predict_image as light_predict
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.vip_official_adapter import VIPQueries, upstream_settings
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import frozen_state, check_frozen, save
import eval_curated20_patchonly2 as natural
import eval_development_readout as task
from eval_shared_rival_soft_full import prepare as rs_prepare

IMPLEMENTATION = 'frozen-sharedlocal-vs-oneside-vs-task-patch-full-v1-20261008'
HISTORICAL = 'PatchOnly2_SpecifiedHistorical'
PRIMARY_METHODS = (BASELINE, LIGHT, RETAINED)


@torch.inference_mode()
def setup(args):
    protocol = json.loads((Path(args.suite_root)/'protocol.json').read_text())
    entry = protocol['datasets'][args.dataset]
    if entry['family'] == 'remote_sensing':
        samples, loader, masks, models, old = rs_prepare(args)
        geometry, banks, vip, queries, checkpoints = models
        geometry = GeometryExecution(geometry)
        words = old['signature']['vocabulary']
    else:
        original = Path(protocol['word_source'])/'vocabularies'/(args.dataset+'_original20.json')
        curated = original.with_name(args.dataset+'_curated20.json')
        adapted = SimpleNamespace(**vars(args), original_vocabulary=str(original),
            curated_vocabulary=str(curated), text_cache=str(Path(protocol['word_source'])/'text_cache'/(args.dataset+'.pt')),
            family='natural')
        samples, _, loader, _ = natural.inputs(adapted)
        specs = {args.dataset+'__'+arm: group for arm, group in
                 zip(natural.ARMS, (natural.exact_specs(original), natural.exact_specs(curated)))}
        segmenter, all_banks, vip, all_queries, identity = natural.models(adapted, specs)
        key = args.dataset+'__original20'
        banks, queries = {args.dataset: all_banks[key]}, {args.dataset: all_queries[key]}
        geometry = GeometryExecution(segmenter)
        # Preserve exact cached strings/features. Only resolve whitespace in the
        # display-name -> canonical-string lookup (ADE's original "bed ").
        query = queries[args.dataset]
        parent_ids = query.parents.cpu().tolist()
        canonical_names = []
        for c,name in enumerate(query.class_names):
            group = [a for i,a in enumerate(query.aliases) if parent_ids[i]==c]
            if name in group:
                canonical_names.append(name)
            else:
                matches = [a for a in group if a.strip()==name.strip()]
                if len(matches)!=1:
                    raise RuntimeError('Canonical identity is ambiguous beyond whitespace: '+name)
                canonical_names.append(matches[0])
        queries[args.dataset] = VIPQueries(query.features,query.parents,tuple(canonical_names),query.aliases)
        checkpoints = segmenter.backbone.checkpoints if hasattr(segmenter.backbone, 'checkpoints') else None
        words = dict(sha256=identity['vocabulary_sha256']['original20'],
            aliases={p: b.alias_names for p,b in banks.items()},
            counts={p: [20]*b.class_count for p,b in banks.items()})
        masks = lambda sample,p,shape: natural.load_target(sample,args.dataset,shape)
    if len(samples) != entry['total_images']:
        raise RuntimeError('Incorrect full inventory count.')
    for p, q in queries.items():
        from dinotool.rival_alias_count import canonical_indices
        canonical_indices(q.class_names,q.aliases,q.parents)
        if any(int((q.parents == c).sum()) != 20 for c in range(len(q.class_names))):
            raise RuntimeError('Frozen20 groups changed.')
    reference = None
    reference_identity = None
    if entry.get('historical_result'):
        ref_root = Path(protocol['historical_source'])
        ref_entry = json.loads((ref_root/'protocol.json').read_text())['datasets'][args.dataset]
        ref_args = SimpleNamespace(**{**vars(args),'suite_root':str(ref_root),
            'original_cache':str(Path(protocol['word_source'])/'text_cache'/(args.dataset+'.pt')),'mode':'full'})
        needed = ('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
        rg,rv,rb,rq,rc = task.load_models(ref_args,ref_entry,needed)
        residual = None
        if ref_entry.get('residual_identity'):
            stored = torch.load(ref_root/'residual_cache.pt',map_location=args.device,weights_only=True)
            if stored['identity'] != ref_entry['residual_identity']:
                raise RuntimeError('Historical residual source changed.')
            residual = stored['wide']
        reference = TaxonomyInference.for_task(rg,rv,rb,rq,family=entry['family'],
            background=ref_entry['background_index'],residual_features=residual,local_background='union_max')
        if tuple(next(iter(rb.values())).class_names) != tuple(next(iter(banks.values())).class_names):
            raise RuntimeError('Historical scored class mapping differs.')
        reference_identity = dict(banks=ref_entry['banks'],residual_identity=ref_entry.get('residual_identity'),
            checkpoint=checkpoint_manifest(rc),source=str(ref_root),local_background='union_max')
    identity = checkpoint_manifest(checkpoints) if checkpoints is not None else identity['checkpoints']
    return protocol,entry,samples,loader,masks,geometry,banks,vip,queries,words,identity,reference,reference_identity


def raw_call(name,image,g,b,v,q,cache,execution):
    if name == LIGHT:
        with memory_execution():
            return light_predict(image,g,b,v,q,methods=(LIGHT,),cache=cache)
    if name == RETAINED:
        from dinotool import one_sided_alias_stream as retained
        from dinotool.supported_positive_alias import predict_image as source_predict
        with memory_execution():
            return source_predict(image,g,b,v,q,methods=(RETAINED,),cache=cache,execution=execution,
                tile_scorer=retained.scores,allowed_methods=retained.METHODS,
                observation_method=retained.OBSERVATION_MEAN,diagnostic_fields=retained.DIAGNOSTICS,
                observation_cacher=cached_observations)
    return baseline_predict(image,g,b,v,q,methods=(BASELINE,),cache=cache)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    if args.mode == 'benchmark':
        require_available_gpu(torch.device(args.device))
    protocol,entry,samples,loader,masks,g,b,v,q,words,identity,reference,ref_identity = setup(args)
    frozen = frozen_state(g,v)
    ref_frozen = None if reference is None else frozen_state(reference.geometry,reference.vip)
    cache,execution = CheckedGroupedCache(),FineCoverageExecution(cached=True,burst=True)
    methods = (*PRIMARY_METHODS,HISTORICAL) if reference else PRIMARY_METHODS
    def call(name,image):
        if name == HISTORICAL:
            prediction,diag = reference.predict(image)
            return {args.dataset:{name:prediction}},{args.dataset:diag}
        if name == 'VIP_Matched20':
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        return raw_call(name,image,g,b,v,q,cache,execution)
    output.mkdir(parents=True)
    if args.mode == 'benchmark':
        rows = []
        names = (*methods,'VIP_Matched20')
        for index in sorted({0,len(samples)//2,len(samples)-1}):
            sample = samples[index]
            image = loader(sample.image_path if args.dataset == 'loveda' else sample)
            expected = {name:call(name,image)[0] for name in names}
            times = {name:dict(seconds=[],peaks=[]) for name in names}
            for repeat in range(args.repetitions):
                for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                    actual,timing = measure(call,(name,image),torch.device(args.device),1)
                    if any(not np.array_equal(actual[0][p][name],expected[name][p][name]) for p in b):
                        raise RuntimeError('Warmed prediction changed.')
                    times[name]['seconds'] += timing['seconds']
                    times[name]['peaks'].append(timing['peak_allocated_mib'])
            for row in times.values():
                row.update(median_seconds=statistics.median(row['seconds']),peak_allocated_mib=max(row.pop('peaks')))
            rows.append(dict(sample_key=sample.key,image_size=list(image.shape[-2:]),timings=times))
        checks = check_frozen(frozen,g,v)
        if reference:
            check_frozen(ref_frozen,reference.geometry,reference.vip)
        save(output/'results.json',dict(status='complete',implementation=IMPLEMENTATION,images=rows,
            target_masks_loaded=False,source_vocabulary=words,historical_identity=ref_identity,
            benchmark_predictions_equal=True,retained_execution=execution.report(),
            memory_scope='all compared models/text/graph pools resident; not isolated deployment memory',**checks))
        return
    selected = samples[:1] if args.mode=='smoke' else samples[args.shard_index::args.num_shards]
    if not selected or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError('Invalid shard.')
    keys = [s.key for s in samples]
    signature = dict(implementation=IMPLEMENTATION,dataset=args.dataset,methods=methods,
        classes={p:bank.class_names for p,bank in b.items()},checkpoints=identity,vocabulary=words,
        gear=dict(geometry=asdict(g.config),frozen_method_sources=protocol['method_sources'],historical=ref_identity),
        competitive=dict(fixed_aliases_per_class=20,rule='unchanged OneSide_Stream and SharedLocal_Soft'),
        global_sample_count=len(keys),global_sample_keys_sha256=digest(keys),num_shards=args.num_shards,
        shard_index=args.shard_index,sample_keys=[s.key for s in selected],
        sample_keys_sha256=digest([s.key for s in selected]),config=vars(args))
    matrices = {p:{name:np.zeros((bank.class_count,)*2,np.int64) for name in methods} for p,bank in b.items()}
    per_image = {p+'__'+name:[] for p in b for name in methods}
    ignored = dict.fromkeys(b,0)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number,sample in enumerate(selected,1):
        image = loader(sample.image_path if args.dataset=='loveda' else sample)
        predictions,diagnostics = {},{}
        for name in methods:
            pred,diag = call(name,image)
            for p in b:
                predictions.setdefault(p,{}).update(pred[p])
            diagnostics[name] = diag
        for name in PRIMARY_METHODS:
            for diag in diagnostics[name].values():
                if diag['geometry_encodings']>4 or diag['wide_encodings']>4:
                    raise RuntimeError('Bounded observation cap changed.')
                if name==LIGHT and (diag['fine_forwards'] or diag['additional_visual_forwards']):
                    raise RuntimeError('Light path acquired extra RGB evidence.')
                if name==RETAINED and not 0<diag['fine_forwards']<=16:
                    raise RuntimeError('Retained fine observation cap changed.')
        if args.mode=='smoke':
            # Whole-image original-path replay where it fits; high-cardinality
            # exact operators additionally have focused execution tests.
            if max(bank.class_count for bank in b.values())<=81:
                joint,_ = light_predict(image,g,b,v,q,methods=(BASELINE,LIGHT),cache=cache)
                retained_joint,_ = retained_predict(image,g,b,v,q,methods=(BASELINE,RETAINED),cache=cache,execution=execution)
            else:
                from dinotool import one_sided_alias_stream as retained
                from dinotool.supported_positive_alias import predict_image as source_predict
                with memory_execution():
                    joint,_ = light_predict(image,g,b,v,q,methods=(BASELINE,LIGHT),cache=cache)
                    retained_joint,_ = source_predict(image,g,b,v,q,methods=(BASELINE,RETAINED),cache=cache,execution=execution,
                        tile_scorer=retained.scores,allowed_methods=retained.METHODS,
                        observation_method=retained.OBSERVATION_MEAN,diagnostic_fields=retained.DIAGNOSTICS,
                        observation_cacher=cached_observations)
            for p in b:
                if (not np.array_equal(joint[p][BASELINE],predictions[p][BASELINE])
                        or not np.array_equal(joint[p][LIGHT],predictions[p][LIGHT])
                        or not np.array_equal(retained_joint[p][BASELINE],predictions[p][BASELINE])
                        or not np.array_equal(retained_joint[p][RETAINED],predictions[p][RETAINED])):
                    raise RuntimeError('Singleton and combined inference differ.')
            checks = check_frozen(frozen,g,v)
            if reference:check_frozen(ref_frozen,reference.geometry,reference.vip)
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                implementation=IMPLEMENTATION,target_masks_loaded=False,singleton_exact=True,
                original_whole_image_replay=max(bank.class_count for bank in b.values())<=81,
                diagnostics=diagnostics,source_vocabulary=words,**checks))
            return
        for p,bank in b.items():
            target = masks(sample,p,tuple(image.shape[-2:]))
            valid = (target>=0)&(target<bank.class_count)
            ignored[p] += int((~valid).sum())
            for name in methods:
                pred = predictions[p][name]
                if pred.shape != target.shape or np.any((pred[valid]<0)|(pred[valid]>=bank.class_count)):
                    raise RuntimeError('Invalid prediction or class mapping.')
                encoded = target[valid].astype(np.int64)*bank.class_count+pred[valid]
                cm = np.bincount(encoded,minlength=bank.class_count**2).reshape(bank.class_count,-1)
                matrices[p][name] += cm
                per_image[p+'__'+name].append(cm)
        if number==1 or number%10==0 or number==len(selected):
            row = dict(status='running',processed_images=number,total_images=len(selected),signature=signature,
                metrics={p:{name:summary(cm,b[p].class_names,ignored[p]) for name,cm in group.items()} for p,group in matrices.items()},
                diagnostics={p:dict(tiles=number) for p in b},target_masks_used_only_after_prediction=True,
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json',row)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(selected),
                seconds=row['wall_seconds'])),flush=True)
        del predictions,diagnostics
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{key:np.stack(value) for key,value in per_image.items()})
    checks = check_frozen(frozen,g,v)
    if reference:check_frozen(ref_frozen,reference.geometry,reference.vip)
    row.update(status='complete',**checks,all_visual_budgets_verified=True)
    save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full','benchmark'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
