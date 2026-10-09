"""Developed-window feasibility and singleton complete-image cost, without fine crops."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_count import canonical_indices
from dinotool.sparse_alias_reuse import (IMPLEMENTATION, METHODS, PRIMARY, alias_layout,
                                         cache_sparse, profile_aliases, sparse_scores)
from dinotool.stratified_soft_alias import WideCrop
from eval_bounded_alias_anchored import make_models
from eval_fine_responsibility_complete import predict_image as slow_image
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_contribution_alias import crop_from_features
from eval_rival_fine_graph import bind
from run_region_semantic_suite_a800 import TOOL
import eval_rival_fine_full as reference


SOURCE = TOOL / 'results/rival_alias_speed_soft_20261005'


@contextmanager
def forbid_fine():
    original = reference.observe_fine
    def forbidden(*args, **kwargs):
        raise RuntimeError('A zero-extra-forward candidate called the fine observer.')
    reference.observe_fine = forbidden
    try:
        yield
    finally:
        reference.observe_fine = original


def layouts_for(queries):
    return {p: alias_layout(q.parents, canonical_indices(q.class_names, q.aliases, q.parents),
                            len(q.class_names)) for p, q in queries.items()}


@torch.inference_mode()
def tile(image, top, left, geometry, banks, vip, queries, wide, crops, wide_count, *,
         methods=METHODS, layouts=None, readers=None):
    height, width = image.shape[-2:]
    prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
    coordinates = reference.tile_coordinates(top, left, geometry.device)
    valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
    operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    output, diagnostics, fields = {}, {}, {'operator': operator, 'coordinates': coordinates, 'valid': valid}
    layouts = layouts_for(queries) if layouts is None else layouts
    for p, bank in banks.items():
        query, layout = queries[p], layouts[p]
        raw = alias_class_scores((prepared.geometry_projected.float()
            @ F.normalize(bank.features.float(), dim=-1).T)[0], bank.parent_indices, bank.class_count)
        local = raw / .07
        broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
        base = local.double() + operator.double() @ (broad.double() - local.double())
        values = {'Geometry': local.double(), 'NoAdmission_Exact': base}
        stats = {'fine_forwards': 0., 'operator_residual': residual}
        active = [m for m in methods if m.startswith('Sparse')]
        if active:
            observed = cache_sparse(crops['clean', p], wide_count, coordinates, (height, width), layout)
            blank = WideCrop(local.new_empty(1024, 1), local.new_empty(1), 0, 0, 512, 512, 32, 512)
            sources = {}
            if any(m.startswith('SparseNative') for m in active):
                crop = crop_from_features(prepared.native_projected, query, blank)
                sources['native'] = profile_aliases(crop.alias_logits, crop.salience, layout)
            if 'SparseGeometrySoft' in active:
                crop = crop_from_features(prepared.geometry_projected, query, blank)
                sources['geometry'] = profile_aliases(crop.alias_logits, crop.salience, layout)
            for name in active:
                witness = sources['geometry' if name == 'SparseGeometrySoft' else 'native']
                reader = (readers or {}).get(name, sparse_scores)
                values[name], diagnostic, _ = reader(local, operator, broad, observed, witness,
                    valid, layout, soft=name != 'SparseNativeHard')
                stats.update({name + '__' + k: value for k, value in diagnostic.items()})
        output[p] = {m: values[m] for m in methods}
        diagnostics[p] = stats
        fields[p + '__raw'], fields[p + '__local'] = raw, local
    return output, diagnostics, fields


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work, *, methods=(PRIMARY,), layouts=None, readers=None):
    layouts = layouts_for(queries) if layouts is None else layouts
    def predictor(*args):
        return tile(*args, methods=methods, layouts=layouts, readers=readers)
    with forbid_fine():
        return bind(reference.predict_image, predict_tile=predictor, METHODS=methods)(
            image, geometry, banks, vip, queries, work)


@torch.inference_mode()
def window(image, geometry, banks, vip, queries, layouts, *, methods=METHODS):
    wide, crops, _, count = reference.prepare_wide(image, vip, {'clean': (banks, queries)})
    with forbid_fine():
        scores, diagnostics, fields = tile(image, 0, 0, geometry, banks, vip, queries,
                                          wide, crops, count, methods=methods, layouts=layouts)
    predictions = {}
    for p, bank in banks.items():
        predictions[p] = {}
        for method, score in scores[p].items():
            raw = fields[p + '__raw'] if method == 'Geometry' else score
            dense = F.interpolate(raw.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                  mode='bilinear', align_corners=False)[0]
            if method == 'Geometry':
                dense = dense / .07
            predictions[p][method] = dense[:, :min(512, image.shape[-2]), :min(512, image.shape[-1])].argmax(0).cpu().numpy()
    return predictions, scores, diagnostics


@torch.inference_mode()
def complete_image_benchmark(image, models, layouts, work, repetitions):
    geometry, banks, vip, queries = models
    def call(name):
        if name == 'FineResponsibilityOnly_Exact':
            return slow_image(image, *models, work, methods=(name,))
        return predict_image(image, *models, work, methods=(name,), layouts=layouts)
    names = ('NoAdmission_Exact', PRIMARY, 'FineResponsibilityOnly_Exact')
    full = predict_image(image, *models, work, methods=('NoAdmission_Exact', PRIMARY), layouts=layouts)
    for name in names:
        value = call(name)
        if name != 'FineResponsibilityOnly_Exact':
            if not all(np.array_equal(value[0][p][name], full[0][p][name]) for p in banks):
                raise RuntimeError('Singleton versus combined complete-image predictions differ.')
    timings = {m: {'seconds': [], 'peak_allocated_mib': []} for m in names}
    predictions = {}
    diagnostics = {}
    for repeat in range(repetitions):
        for name in names if repeat % 2 == 0 else names[::-1]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            value = call(name)
            torch.cuda.synchronize()
            timings[name]['seconds'].append(time.perf_counter() - begun)
            timings[name]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated() / 1048576)
            predictions[name] = {p: value[0][p][name] for p in banks}
            diagnostics[name] = value[1]
    for row in timings.values():
        row['median_seconds'] = statistics.median(row['seconds'])
        row['peak_allocated_mib'] = max(row['peak_allocated_mib'])
    return {'scope': 'singleton COMPLETE image including stitching and argmax; not window latency',
        'repetitions': repetitions, 'image_size': list(image.shape[-2:]), 'results': timings,
        'singleton_multi_arm_predictions_equal': True,
        'candidate_fine_observer_forbidden_at_runtime': True,
        'memory_scope': 'both backbones and slow graph co-resident, not isolated deployment',
        'loading_text_decoding_masks_excluded': True, 'diagnostics': diagnostics}, predictions


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing sparse reuse output.')
    prior = json.loads((SOURCE / args.dataset / 'results.json').read_text())
    if prior['status'] != 'complete' or prior['processed_images'] != 8:
        raise RuntimeError('Existing complete developed eight-window source required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    models = make_models(args, specs)
    geometry, banks, vip, queries, checkpoints = models
    if (checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']
            or asdict(geometry.config) != prior['signature']['geometry']
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest()
               != prior['signature']['vocabulary_sha256']):
        raise RuntimeError('Frozen checkpoint, Geometry or vocabulary changed.')
    layouts = layouts_for(queries)
    state = reference.frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output / 'scores').mkdir()
    matrices = {p: {m: np.zeros((bank.class_count,) * 2, np.int64) for m in METHODS} for p, bank in banks.items()}
    per_image = {p + '__' + m: [] for p in banks for m in METHODS}
    diagnostics, ignored = [], dict.fromkeys(banks, 0)
    begun = time.perf_counter()
    with np.load(SOURCE / args.dataset / 'per_image_confusions.npz', allow_pickle=False) as old_confusions:
        for i, key in enumerate(prior['sample_keys']):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            predictions, scores, diagnostic = window(image, *models[:4], layouts)
            with np.load(SOURCE / args.dataset / 'scores' / f'{i}.npz', allow_pickle=False) as old:
                for p in banks:
                    for method in ('Geometry', 'NoAdmission_Exact'):
                        if not np.array_equal(scores[p][method].cpu().numpy(), old[p + '__' + method]):
                            raise RuntimeError('Original score replay differs: ' + p + '/' + method)
            np.savez_compressed(output / 'scores' / f'{i}.npz', **{
                p + '__' + m: score.cpu().numpy() for p, group in scores.items() for m, score in group.items()})
            diagnostics.append({'sample_key': key, 'protocols': diagnostic})
            for p, bank in banks.items():
                target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
                valid = (target >= 0) & (target < bank.class_count)
                ignored[p] += int((~valid).sum())
                for m in METHODS:
                    encoded = target[valid].astype(np.int64) * bank.class_count + predictions[p][m][valid]
                    cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                    if m in ('Geometry', 'NoAdmission_Exact') and not np.array_equal(cm, old_confusions[p + '__' + m][i]):
                        raise RuntimeError('Original per-image endpoint differs: ' + p + '/' + m)
                    matrices[p][m] += cm
                    per_image[p + '__' + m].append(cm)
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'primary': PRIMARY,
                'methods': METHODS, 'signature': prior['signature'], 'processed_images': i + 1, 'total_images': 8,
                'sample_keys': prior['sample_keys'][:i + 1], 'scores_persisted_before_masks': True,
                'original_two_endpoints_exact': True, 'candidate_fine_observer_forbidden_at_runtime': True,
                'target_labels_used_by_weights': False, 'diagnostics': diagnostics,
                'metrics': {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                            for p, group in matrices.items()}}
            reference.save(output / 'results.json', result)
            print(json.dumps({'phase': 'window', 'dataset': args.dataset, 'processed': i + 1}), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(prior['sample_keys']),
                        **{k: np.stack(v) for k, v in per_image.items()})
    sample = lookup[prior['sample_keys'][0]]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    print(json.dumps({'phase': 'complete_image_timing', 'dataset': args.dataset,
                      'image_size': list(image.shape[-2:])}), flush=True)
    result['benchmark'], predictions = complete_image_benchmark(image, models[:4], layouts, output, args.repetitions)
    np.savez_compressed(output / 'benchmark_predictions.npz', **{
        m + '__' + p: pred for m, group in predictions.items() for p, pred in group.items()})
    result['benchmark']['sample_key'] = sample.key
    result['benchmark_image_metrics'] = {}
    for p, bank in banks.items():
        target = load_mask(sample, p, tuple(image.shape[-2:]))
        valid = (target >= 0) & (target < bank.class_count)
        result['benchmark_image_metrics'][p] = {}
        for method, group in predictions.items():
            encoded = target[valid].astype(np.int64) * bank.class_count + group[p][valid]
            cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
            result['benchmark_image_metrics'][p][method] = summary(cm, bank.class_names, int((~valid).sum()))
    result.update(status='complete', coverage_verified=True, wall_seconds=time.perf_counter() - begun,
                  **reference.check_frozen(state, geometry, vip))
    reference.save(output / 'results.json', result)
    print(json.dumps({'status': 'complete', 'dataset': args.dataset}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=3)
    main(parser.parse_args())
