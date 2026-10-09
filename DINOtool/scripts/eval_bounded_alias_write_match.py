"""Complete-image fixed-action audit with magnitude-matched alternative writers."""
import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_alias_path_audit import CHUNK, assemble, capture, candidate_predictions_from_changes
from dinotool.bounded_alias_write_match import (
    BASELINE, IMPLEMENTATION, WRITERS, privileged_changes, writer_changes,
)
from dinotool.bounded_patch_only import PROTOCOL, predict_image as original_predict
from dinotool.model import checkpoint_manifest
from eval_bounded_alias_path_audit import confusion
from eval_bounded_crop_head_alias import panel_inputs
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save


ENDPOINTS = ('Geometry', 'NoAdmission_Exact', BASELINE)
METHODS = (*ENDPOINTS, *(f'Privileged_{writer}' for writer in WRITERS))


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    sources = capture(image, geometry, banks, vip, queries)
    expected, _ = original_predict(image, geometry, banks, vip, queries, methods=ENDPOINTS)
    diagnostics = {}
    for p, source in sources.items():
        for method in ENDPOINTS:
            if not np.array_equal(assemble(source, method=method), expected[p][method]):
                raise RuntimeError('Original complete-image endpoint differs: '+p+'/'+method)
        changes = [t.wide_changes[0].flatten(1) for t in source.tiles]
        written, metrics = writer_changes(source, changes)
        for writer in WRITERS:
            start, predictions = next(candidate_predictions_from_changes(source, written[writer]))
            if start:
                raise RuntimeError('Counterfactual enumeration changed.')
            for k, prediction in enumerate(predictions.cpu().numpy()):
                singleton, _ = writer_changes(source, [field[:, k:k+1] for field in changes])
                full = []
                for tile, value in zip(source.tiles, singleton[writer]):
                    delta = torch.zeros_like(tile.scores[BASELINE])
                    delta[:, 0] = value[:, 0]
                    full.append(delta)
                if not np.array_equal(prediction, assemble(source, full)):
                    raise RuntimeError('Batched counterfactual differs from singleton: '+p+'/'+writer)
        zero, _ = writer_changes(source, [torch.zeros_like(field) for field in changes])
        if any(bool(value.any()) for fields in zero.values() for value in fields):
            raise RuntimeError('Zero action changed a writer.')
        diagnostics[p] = {**source.diagnostics, **metrics}
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, target_masks_loaded=False, original_endpoints_exact=True,
        batch_singleton_exact=True, diagnostic_only=True, diagnostics=diagnostics,
        **check_frozen(state, geometry, vip)))


@torch.inference_mode()
def evaluate(args):
    output, samples, load_image, load_mask, geometry, banks, vip, queries, checkpoints, old = panel_inputs(args)
    if args.num_shards != 1 or args.shard_index:
        raise ValueError('One fixed diagnostic shard per domain required.')
    state = frozen_state(geometry, vip)
    keys = [sample.key for sample in samples]
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={p: bank.class_names for p, bank in banks.items()},
        vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        sample_keys=keys, global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
        geometry=asdict(geometry.config), input_protocol=PROTOCOL, writers=WRITERS,
        source='Wide FixedSlots; original20 aliases/salience',
        magnitude_matching='per intervention column; summed squared norm over all valid tile fields',
        padding='matched routes exclude padded donors/queries; DirectLegacy preserves historical fields',
        privileged='same strongest noncanonical wrong-class donor deletion for every writer; labels audit only',
        counterfactual_restore_batch=CHUNK, diagnostic_only=True)
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    singles, diagnostics = {p: [] for p in banks}, {p: [] for p in banks}
    ignored = dict.fromkeys(banks, 0)
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        sources = capture(image, geometry, banks, vip, queries)
        frozen = {p: writer_changes(source, [t.wide_changes[0].flatten(1) for t in source.tiles])
                  for p, source in sources.items()}
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
            written, metrics = frozen[p]
            privileged, privileged_metrics = writer_changes(source, privileged_changes(source, target))
            single = np.zeros((len(WRITERS), bank.class_count, 20, bank.class_count, bank.class_count), np.int64)
            for arm, writer in enumerate(WRITERS):
                for start, predictions in candidate_predictions_from_changes(source, written[writer]):
                    for offset, prediction in enumerate(predictions):
                        cm = torch.bincount(encoded_target+prediction[valid_gpu].long(),
                            minlength=bank.class_count**2).reshape(bank.class_count, bank.class_count).cpu().numpy()
                        owner, alias = divmod(start+offset, 20)
                        single[arm, owner, alias] = cm
                name = 'Privileged_'+writer
                cm = confusion(target, assemble(source, privileged[writer]), bank.class_count)
                matrices[p][name] += cm
                per_image[p+'__'+name].append(cm)
                print(json.dumps(dict(dataset=args.dataset, sample=sample.key, protocol=p,
                    completed_writer=writer, words=bank.class_count*20)), flush=True)
            if not np.array_equal(single.sum(-1), np.broadcast_to(
                    per_image[p+'__'+BASELINE][-1].sum(-1)[None, None, None], single.shape[:-1])):
                raise RuntimeError('Counterfactual scored target counts differ.')
            singles[p].append(single)
            diagnostics[p].append({**source.diagnostics, **metrics,
                **{'privileged_'+k: v for k, v in privileged_metrics.items()}})
            del target_gpu, valid_gpu, encoded_target
        result = dict(status='running', processed_images=number, total_images=len(samples), signature=signature,
            metrics={p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                     for p, group in matrices.items()},
            diagnostics={p: {k: float(max(row[k] for row in rows)) if 'error' in k else
                float(np.mean([row[k] for row in rows])) for k in rows[0]} for p, rows in diagnostics.items()},
            source_frozen_before_target_masks=True, single_action_magnitude_matching_before_target_masks=True,
            privileged_magnitude_matching_after_target_masks=True,
            target_masks_used_for_privileged_controls=True, target_label_tuning=False, diagnostic_only=True,
            wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
        save(output/'results.json', result)
        print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(samples))), flush=True)
        del sources, baselines, frozen
    arrays = {k: np.stack(v) for k, v in per_image.items()}
    arrays.update({p+'__single': np.stack(rows, axis=1) for p, rows in singles.items()})
    arrays.update({p+'__members': np.asarray([(bank.parent_indices == c).nonzero().flatten().cpu().tolist()
        for c in range(bank.class_count)]) for p, bank in banks.items()})
    np.savez_compressed(output/'per_image_audit.npz', sample_keys=np.asarray(keys), **arrays)
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
