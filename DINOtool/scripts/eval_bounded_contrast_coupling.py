"""Fixed all20 full-domain test of zero-DC coupling and matched local heads."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.bounded_contrast_coupling import IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL, predict_image
from dinotool.geometry_execution import GeometryExecution
from dinotool.model import checkpoint_manifest
from dinotool.vip_official_adapter import upstream_settings
from eval_bounded_physical_coupling import predict_image as original_predict
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_shared_rival_soft_full import prepare


def inputs(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing output.')
    samples, load_image, load_mask, models, old = prepare(args)
    geometry, banks, vip, queries, checkpoints = models
    return output, samples, load_image, load_mask, GeometryExecution(geometry), banks, vip, queries, checkpoints, old


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    combined, diagnostics = predict_image(image, geometry, banks, vip, queries)
    original, _ = original_predict(image, geometry, banks, vip, queries, None, methods=METHODS[:2])
    for p in banks:
        for m in METHODS[:2]:
            if not np.array_equal(combined[p][m], original[p][m]):
                raise RuntimeError('Original bounded endpoint differs: ' + p + '/' + m)
    for m in METHODS[2:]:
        single, _ = predict_image(image, geometry, banks, vip, queries, methods=(m,))
        if any(not np.array_equal(single[p][m], combined[p][m]) for p in banks):
            raise RuntimeError('Singleton differs from simultaneous control: ' + m)
    if any(row['correction_mean_error'] > 1e-10 for row in diagnostics.values()):
        raise RuntimeError('Valid-token mean invariant failed.')
    output.mkdir(parents=True)
    save(output / 'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, diagnostics=diagnostics, target_masks_loaded=False,
        original_endpoints_exact=True, singleton_primary_exact=True, **check_frozen(state, geometry, vip)))


@torch.inference_mode()
def evaluate(args, *, predictor=predict_image, implementation=IMPLEMENTATION,
             methods=METHODS, view_protocol=PROTOCOL, competitive=None):
    output, samples, load_image, load_mask, geometry, banks, vip, queries, checkpoints, old = inputs(args)
    state = frozen_state(geometry, vip)
    selected = samples[args.shard_index::args.num_shards]
    if not selected or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid or empty shard.')
    keys = [s.key for s in samples]
    signature = dict(implementation=implementation, dataset=args.dataset, methods=methods,
        classes={p: bank.class_names for p, bank in banks.items()},
        gear=dict(geometry=asdict(geometry.config), input_protocol=view_protocol,
                  fine_forwards=view_protocol.get('fine_forwards', 0),
                  local_tiles=[512, 128], reconstruction=view_protocol['reconstruction']),
        competitive=competitive if competitive is not None else dict(alias_admission='none', fixed_aliases_per_class=20,
                         reconstruction='valid-token P H P; no fitted threshold, class or domain routing'),
        vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
        num_shards=args.num_shards, shard_index=args.shard_index,
        sample_keys=[s.key for s in selected], sample_keys_sha256=digest([s.key for s in selected]), config=vars(args))
    matrices = {p: {m: np.zeros((bank.class_count,) * 2, np.int64) for m in methods} for p, bank in banks.items()}
    per_image = {p + '__' + m: [] for p in banks for m in methods}
    ignored = dict.fromkeys(banks, 0)
    totals = {p: dict(tiles=0) for p in banks}
    cache = CheckedGroupedCache()
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostics = predictor(image, geometry, banks, vip, queries, cache=cache)
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for m in methods:
                encoded = target[valid].astype(np.int64) * bank.class_count + predictions[p][m][valid]
                cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                matrices[p][m] += cm
                per_image[p + '__' + m].append(cm)
            tiles = diagnostics[p]['tiles']
            totals[p]['tiles'] += tiles
            for field, value in diagnostics[p].items():
                if field != 'tiles':
                    totals[p][field] = totals[p].get(field, 0.) + value * tiles
        if number == 1 or number % 5 == 0 or number == len(selected):
            result = dict(status='running', processed_images=number, total_images=len(selected), signature=signature,
                metrics={p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                         for p, group in matrices.items()},
                diagnostics={p: {k: v if k == 'tiles' else v / row['tiles'] for k, v in row.items()} for p, row in totals.items()},
                wall_seconds=time.perf_counter() - started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1048576,
                target_masks_used_only_after_prediction=True,
                fine_observer_forbidden=view_protocol.get('fine_forwards', 0) == 0)
            save(output / 'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(selected),
                miou={p: {m: v['mean_iou_percent'] for m, v in group.items()} for p, group in result['metrics'].items()})), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result.update(status='complete', **check_frozen(state, geometry, vip))
    save(output / 'results.json', result)


@torch.inference_mode()
def benchmark(args, *, predictor=predict_image, implementation=IMPLEMENTATION,
              view_protocol=PROTOCOL, names=None):
    require_available_gpu(torch.device(args.device))
    output, samples, load_image, _, geometry, banks, vip, queries, _, old = inputs(args)
    state = frozen_state(geometry, vip)
    settings = upstream_settings(args.dataset)
    names = names or ('NoAdmission_Exact', PRIMARY, 'SCLIP_NoAdmission', 'SCLIP_ContrastCoupled', 'VIP_All20')
    rows = []
    for index in sorted({0, len(samples) // 2, len(samples) - 1}):
        sample = samples[index]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        def call(name):
            if name == 'VIP_All20':
                return {p: vip.predict(image, q, settings)[0] for p, q in queries.items()}
            return predictor(image, geometry, banks, vip, queries, methods=(name,))[0]
        expected = {name: call(name) for name in names}
        timings = {name: dict(seconds=[], peaks=[]) for name in names}
        for repeat in range(args.repetitions):
            for name in names[repeat % len(names):] + names[:repeat % len(names)]:
                actual, timing = measure(call, (name,), torch.device(args.device), 1)
                for p in banks:
                    a = actual[p] if name == 'VIP_All20' else actual[p][name]
                    b = expected[name][p] if name == 'VIP_All20' else expected[name][p][name]
                    if not np.array_equal(a, b):
                        raise RuntimeError('Warm singleton prediction changed.')
                timings[name]['seconds'].extend(timing['seconds'])
                timings[name]['peaks'].append(timing['peak_allocated_mib'])
        for value in timings.values():
            value.update(median_seconds=statistics.median(value['seconds']), peak_allocated_mib=max(value.pop('peaks')))
        rows.append(dict(sample_key=sample.key, image_size=list(image.shape[-2:]), timings=timings))
    output.mkdir(parents=True)
    save(output / 'results.json', dict(status='complete', implementation=implementation, images=rows,
        input_protocol=view_protocol, source_vocabulary=old['signature']['vocabulary'], target_masks_loaded=False,
        memory_scope=('Geometry, VIP and independent semantic observer resident; not standalone deployment memory'
                      if view_protocol.get('semantic_source') else
                      'both backbones resident; not standalone deployment memory'),
        **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    {'smoke': smoke, 'full': evaluate, 'benchmark': benchmark}[args.mode](args)
