"""Full paired Potsdam/VOC20 evaluation without shuffled-weight comparisons."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.geometry_execution import GeometryExecution
from dinotool import local_role_transfer as trial
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, _load_upstream
from eval_curated20_patchonly2 import inputs, models
from eval_development_readout import observations, probabilities, wide_scores
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def setup(args, protocol):
    root = Path(args.suite_root)
    entry = protocol['datasets'][args.dataset]
    args.family = entry['family']
    args.original_vocabulary = str(root/'vocabularies'/(args.dataset+'_original20.json'))
    args.curated_vocabulary = str(root/'vocabularies'/(args.dataset+'_curated20.json'))
    args.text_cache = str(root/'source_text_cache'/(args.dataset+'.pt'))
    for name, expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: '+name)
    samples, specs, loader, masks = inputs(args)
    segmenter, banks, vip, queries, identity = models(args, specs)
    geometry = GeometryExecution(segmenter)
    key = args.dataset+'__'+entry['bank']
    original, query = banks[key], queries[key]
    meta_path = root/'vocabularies'/(args.dataset+'_roles.json')
    if hashlib.sha256(meta_path.read_bytes()).hexdigest() != entry['role_sha256']:
        raise RuntimeError('Frozen role metadata changed.')
    meta = json.loads(meta_path.read_text())
    if tuple(v for row in meta['classes'] for v in row['synonyms']) != original.alias_names:
        raise RuntimeError('Role metadata and text strings/order differ.')
    policy = trial.POLICIES[trial.TRANSFER[args.dataset]]
    if asdict(policy) != entry['policy']:
        raise RuntimeError('Frozen transferred profile differs.')
    text_identity = dict(source=identity, selected_key=key, template=policy.template,
                         role_sha256=entry['role_sha256'])
    cache = root/'task_text_cache'/(args.dataset+'.pt')
    stored = torch.load(cache, map_location=geometry.device, weights_only=True) if cache.exists() else None
    if stored and stored['identity'] != text_identity:
        raise RuntimeError('Task text cache identity changed.')
    if stored:
        query = VIPQueries(stored['wide'], query.parents, query.class_names, query.aliases)
        local = stored['local']
    else:
        if policy.template == 'seg_template':
            module = _load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py', 'role_transfer_templates')
            previous = vip.templates
            vip.templates = module.get_text_template(policy.template)
            try:
                query = vip.encode_queries(original.class_names, tuple(tuple(row['synonyms']) for row in meta['classes']))
            finally:
                vip.templates = previous
        local = F.normalize(query.features.float().mean(1), dim=-1)
        temporary = cache.with_suffix('.tmp')
        torch.save(dict(identity=text_identity, wide=query.features.cpu(), local=local.cpu()), temporary)
        temporary.rename(cache)
    bank = TCPRTextBank(local, original.parent_indices, original.canonical_mask,
                        original.class_names, original.alias_names)
    bank.validate()
    plan = trial.role.prepare_plan(bank, meta['family_ids'], meta.get('semantic_roots'))
    return samples, loader, masks, geometry, vip, bank, query, plan, policy, identity, text_identity


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    device = torch.device(args.device)
    require_available_gpu(device)
    lease = torch.empty(1, device=device)
    output.mkdir(parents=True)
    def state(phase, **kw):
        save(output/'worker_status.json', dict(status='running', phase=phase, **kw))
    state('loading')
    root = Path(args.suite_root)
    protocol = json.loads((root/'protocol.json').read_text())
    samples, loader, masks, g, v, bank, query, plan, policy, identity, text_identity = setup(args, protocol)
    frozen = frozen_state(g, v)
    def call(name, image):
        return trial.predict(image, g, v, bank, query, plan, policy, (name,))
    if args.mode == 'benchmark':
        smoke, timing = [], []
        for index in sorted({0, len(samples)//2, len(samples)-1}):
            image = loader(samples[index])
            state('smoke', sample_key=samples[index].key)
            joint, diag = trial.predict(image, g, v, bank, query, plan, policy)
            source = observations(image, g, v, (policy.strength,), wide_policy=policy.wide_policy)
            broad = wide_scores(source, query, policy.tau, policy.tem)
            inherited = probabilities(source, policy.profile(), bank, broad, {}).argmax(0).cpu().numpy()
            if not np.array_equal(joint[trial.BASE], inherited):
                raise RuntimeError('Vectorized unweighted baseline differs from inherited prediction.')
            del source, broad
            for name in trial.METHODS:
                if not np.array_equal(call(name, image)[0][name], joint[name]):
                    raise RuntimeError('Singleton/joint output differs: '+name)
            smoke.append(dict(sample_key=samples[index].key, unweighted_inherited_prediction_exact=True,
                singleton_joint_predictions_exact=True, diagnostics=diag))
            row = {name:dict(seconds=[], peaks=[]) for name in trial.METHODS}
            state('timing', sample_key=samples[index].key)
            for repeat in range(args.repetitions):
                order = trial.METHODS[repeat % 3:]+trial.METHODS[:repeat % 3]
                for name in order:
                    require_available_gpu(device)
                    actual, val = measure(call, (name, image), device, 1)
                    require_available_gpu(device)
                    if not np.array_equal(actual[0][name], joint[name]):
                        raise RuntimeError('Warm prediction changed.')
                    row[name]['seconds'] += val['seconds']
                    row[name]['peaks'].append(val['peak_allocated_mib'])
            for val in row.values():
                val.update(median_seconds=statistics.median(val['seconds']),
                           peak_allocated_mib=max(val.pop('peaks')))
            timing.append(dict(sample_key=samples[index].key, timings=row))
        save(output/'smoke.json', dict(status='complete', images=smoke, target_masks_loaded=False,
             text_identity=text_identity, **check_frozen(frozen, g, v)))
        save(output/'timing.json', dict(status='complete', images=timing, target_masks_loaded=False,
            whole_image_scope=True, exclusive_gpu_verified=True,
            includes='resize, both visual branches, alias reduction, writeback, restoration, CPU argmax output',
            excludes='decode and model/text setup', memory_scope='two frozen backbones and selected bank; source banks also resident',
            both_primary_baseline_use_vectorized_local=True))
        save(output/'worker_status.json', dict(status='complete', phase='complete'))
        return
    for d in protocol['datasets']:
        smoke = json.loads((root/'benchmark'/d/'smoke.json').read_text())
        if smoke['status'] != 'complete' or smoke['target_masks_loaded']:
            raise RuntimeError('Completed mask-free smoke checks required.')
    keys = [s.key for s in samples]
    selected = samples[args.shard_index::args.num_shards]
    if (len(keys) != protocol['datasets'][args.dataset]['total_images'] or len(keys) != len(set(keys))
            or not selected or not 0 <= args.shard_index < args.num_shards):
        raise RuntimeError('Unique full coverage/shard differs.')
    nc = bank.class_count
    signature = dict(implementation=trial.IMPLEMENTATION, dataset=args.dataset, methods=trial.METHODS,
        classes={args.dataset:bank.class_names}, checkpoints=identity['checkpoints'],
        vocabulary=dict(aliases=bank.alias_names, counts=[20]*nc,
                        source_sha256=identity['vocabulary_sha256'], role_sha256=protocol['datasets'][args.dataset]['role_sha256']),
        gear=dict(profile=asdict(policy), text_identity=text_identity, method_sources=protocol['method_sources']),
        competitive=dict(static_roles=True, all20_wide_retained=True, class_square_tensor=False),
        global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected], sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards, shard_index=args.shard_index, config=vars(args))
    cm = {name:np.zeros((nc,nc), np.int64) for name in trial.METHODS}
    per = {args.dataset+'__'+name:[] for name in trial.METHODS}
    ignored = 0
    totals = {}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = loader(sample)
        pred, diag = trial.predict(image, g, v, bank, query, plan, policy)
        target = masks(sample, args.dataset+'__'+protocol['datasets'][args.dataset]['bank'], tuple(image.shape[-2:]))
        valid = (target >= 0)&(target < nc)
        ignored += int((~valid).sum())
        for name in trial.METHODS:
            if pred[name].shape != target.shape or np.any((pred[name][valid] < 0)|(pred[name][valid] >= nc)):
                raise RuntimeError('Prediction size/class mismatch.')
            current = np.bincount(target[valid].astype(np.int64)*nc+pred[name][valid], minlength=nc*nc).reshape(nc,nc)
            cm[name] += current
            per[args.dataset+'__'+name].append(current)
        if diag['geometry_encodings'] > 4 or diag['wide_encodings'] > 4 or diag['fine_forwards']:
            raise RuntimeError('Frozen visual budget exceeded.')
        for key, val in diag.items():
            totals[key] = totals.get(key, 0.)+val
        if number == 1 or number % 20 == 0 or number == len(selected):
            row = dict(status='running', processed_images=number, total_images=len(selected), signature=signature,
                metrics={args.dataset:{name:summary(matrix, bank.class_names, ignored) for name,matrix in cm.items()}},
                diagnostics={args.dataset:{k:val/number for k,val in totals.items()}},
                wall_seconds=time.perf_counter()-start, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_masks_used_only_after_prediction=True, target_label_tuning=False)
            save(output/'results.json', row)
            state('full', processed=number, total=len(selected))
            print(json.dumps(dict(dataset=args.dataset, shard=args.shard_index, processed=number, total=len(selected))), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{name:np.stack(values) for name,values in per.items()})
    row.update(status='complete', **check_frozen(frozen, g, v))
    save(output/'results.json', row)
    save(output/'worker_status.json', dict(status='complete', phase='complete', processed=len(selected), total=len(selected)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mode', choices=('benchmark','full'), required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sample-seed', type=int, default=20260923)
    p.add_argument('--vdd-ontology', default='official')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--repetitions', type=int, default=5)
    main(p.parse_args())
