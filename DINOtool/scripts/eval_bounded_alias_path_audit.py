"""Complete-image fixed-source path audit, with explicitly privileged controls."""
import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_alias_path_audit import (IMPLEMENTATION, PATHS, POOLS, CHUNK,
    assemble, capture, candidate_predictions, privileged_prediction, routed_change)
from dinotool.bounded_patch_only import PRIMARY as BASELINE, PROTOCOL, predict_image as original_predict
from dinotool.model import checkpoint_manifest
from eval_bounded_crop_head_alias import panel_inputs
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save


ENDPOINTS = ('Geometry', 'NoAdmission_Exact', BASELINE)
ARMS = tuple((path, pool) for path in PATHS for pool in POOLS)
METHODS = (*ENDPOINTS, *(f'Privileged_{path}_{pool}' for path, pool in ARMS))


def confusion(target, prediction, classes):
    valid = (target >= 0) & (target < classes)
    encoded = target[valid].astype(np.int64)*classes+prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=classes**2).reshape(classes, classes)


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    sources = capture(image, geometry, banks, vip, queries)
    expected, _ = original_predict(image, geometry, banks, vip, queries, methods=ENDPOINTS)
    for p, source in sources.items():
        for method in ENDPOINTS:
            if not np.array_equal(assemble(source, method=method), expected[p][method]):
                raise RuntimeError('Original complete-image source endpoint differs: '+p+'/'+method)
        for path, pool in ARMS:
            start, predictions = next(candidate_predictions(source, path, pool))
            if start:
                raise RuntimeError('Counterfactual enumeration changed.')
            for k, prediction in enumerate(predictions.cpu().numpy()):
                changes = []
                for tile in source.tiles:
                    delta = routed_change(tile.local_changes[POOLS.index(pool)].flatten(1),
                                          tile.wide_changes[POOLS.index(pool)].flatten(1), tile.operator, path)[:, k]
                    full = torch.zeros_like(tile.scores[BASELINE])
                    full[:, 0] = delta
                    changes.append(full)
                if not np.array_equal(prediction, assemble(source, changes)):
                    raise RuntimeError('Batched full-size audit differs from singleton: '+path+'/'+pool)
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, target_masks_loaded=False, original_endpoints_exact=True,
        batch_singleton_exact=True, diagnostic_only=True,
        diagnostics={p: source.diagnostics for p, source in sources.items()},
        **check_frozen(state, geometry, vip)))


@torch.inference_mode()
def evaluate(args):
    output, samples, load_image, load_mask, geometry, banks, vip, queries, checkpoints, old = panel_inputs(args)
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError('The fixed diagnostic uses one shard per domain.')
    state = frozen_state(geometry, vip)
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={p: bank.class_names for p, bank in banks.items()},
        vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        sample_keys=[sample.key for sample in samples], global_sample_count=len(samples),
        global_sample_keys_sha256=digest([sample.key for sample in samples]),
        geometry=asdict(geometry.config), input_protocol=PROTOCOL,
        paths=PATHS, pools=POOLS, action='one global alias removal; original salience fixed',
        privileged='one strongest noncanonical alias removal per wrong-class labeled donor',
        counterfactual_restore_batch=CHUNK, diagnostic_only=True)
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    singles = {p: [] for p in banks}
    ignored = dict.fromkeys(banks, 0)
    diagnostics = {p: [] for p in banks}
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        sources = capture(image, geometry, banks, vip, queries)
        baselines = {p: {m: assemble(source, method=m) for m in ENDPOINTS} for p, source in sources.items()}
        for p, source in sources.items():
            bank = banks[p]
            target = load_mask(sample, p, source.output_size)
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for m in ENDPOINTS:
                cm = confusion(target, baselines[p][m], bank.class_count)
                matrices[p][m] += cm
                per_image[p+'__'+m].append(cm)
            target_gpu = torch.as_tensor(target, device=geometry.device).long()
            valid_gpu = (target_gpu >= 0) & (target_gpu < bank.class_count)
            encoded_target = target_gpu[valid_gpu]*bank.class_count
            single = np.zeros((len(ARMS), bank.class_count, 20, bank.class_count, bank.class_count), np.int64)
            for arm, (path, pool) in enumerate(ARMS):
                for start, predictions in candidate_predictions(source, path, pool):
                    for offset, prediction in enumerate(predictions):
                        cm = torch.bincount(encoded_target+prediction[valid_gpu].long(),
                            minlength=bank.class_count**2).reshape(bank.class_count, bank.class_count).cpu().numpy()
                        owner, alias = divmod(start+offset, 20)
                        single[arm, owner, alias] = cm
                name = f'Privileged_{path}_{pool}'
                prediction = privileged_prediction(source, target, path, pool)
                cm = confusion(target, prediction, bank.class_count)
                matrices[p][name] += cm
                per_image[p+'__'+name].append(cm)
                print(json.dumps(dict(dataset=args.dataset, sample=sample.key, protocol=p,
                    completed_path=path, pool=pool, words=bank.class_count*20)), flush=True)
            if not np.array_equal(single.sum(-1), np.broadcast_to(
                    per_image[p+'__'+BASELINE][-1].sum(-1)[None, None, None], single.shape[:-1])):
                raise RuntimeError('Counterfactual scored target counts differ.')
            singles[p].append(single)
            diagnostics[p].append(source.diagnostics)
            del target_gpu, valid_gpu, encoded_target
        result = dict(status='running', processed_images=number, total_images=len(samples), signature=signature,
            metrics={p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                     for p, group in matrices.items()},
            diagnostics={p: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
                         for p, rows in diagnostics.items()},
            source_frozen_before_target_masks=True, target_masks_used_for_privileged_controls=True,
            target_label_tuning=False, diagnostic_only=True,
            wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
        save(output/'results.json', result)
        print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(samples),
            baseline={p: group[BASELINE]['mean_iou_percent'] for p, group in result['metrics'].items()})), flush=True)
        del sources, baselines
    arrays = {k: np.stack(v) for k, v in per_image.items()}
    arrays.update({p+'__single': np.stack(rows, axis=1) for p, rows in singles.items()})
    arrays.update({p+'__members': source for p, source in
                   ((p, np.asarray([(banks[p].parent_indices == c).nonzero().flatten().cpu().tolist()
                                    for c in range(banks[p].class_count)])) for p in banks)})
    np.savez_compressed(output/'per_image_audit.npz', sample_keys=np.asarray(signature['sample_keys']), **arrays)
    result.update(status='complete', **check_frozen(state, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    args = parser.parse_args()
    {'smoke': smoke, 'full': evaluate}[args.mode](args)
