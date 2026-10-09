"""Frozen-model64-window reader study, exact replay and independent stage timing."""
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

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.fine_alias_view import CONFIG
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_fast import (IMPLEMENTATION, METHODS, SOFT_METHODS,
    retained_scores_fast, cached_risk, soft_scores)
from dinotool.rival_alias_count import canonical_indices
from dinotool.rival_fine_full import retained_scores
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, predict_image
from eval_rival_fine_fast import predict_image_fast
from eval_stratified_soft_alias import tile_coordinates
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL / 'results/rival_fine_coupling_20261003'
COST_METHODS = ('ReferenceHard', 'RivalFineHard_Exact', 'FineSoft_Weighted', 'FineSoft_Excess',
                'ReuseSoft_Weighted', 'ReuseSoft_Excess', 'NoAdmission_Exact')


class StageTimer:
    def __init__(self):
        self.seconds = {}

    def call(self, stage, function, *args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = function(*args, **kwargs)
        torch.cuda.synchronize()
        self.seconds[stage] = self.seconds.get(stage, 0.) + time.perf_counter() - started
        return output


@torch.inference_mode()
def window(image, geometry, banks, vip, queries, methods, verify_reference=False):
    timer = StageTimer()
    h, w = image.shape[-2:]
    wide, crops, _, count = timer.call('wide_observation', prepare_wide, image, vip, {'clean': (banks, queries)})
    prepared = timer.call('geometry_forward', geometry.prepare_image, _crop_at(image, 0, 0, 512).to(geometry.device))
    coordinates = tile_coordinates(0, 0, geometry.device)
    valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
    fine_required = verify_reference or any(m.startswith('Fine') or m in ('ReferenceHard', 'RivalFineHard_Exact') for m in methods)
    if fine_required:
        fine, fine_count, costs = timer.call('fine_observation', observe_fine, image[:, :512, :512], vip, queries, coordinates, valid)
    else:
        fine, fine_count, costs = None, None, {'fine_forwards': 0}
    operator, residual = timer.call('reconstruction_operator', reconstruction_operator, prepared.geometry_patch_conditional[0], valid)
    predictions, diagnostics, scores, errors = {}, {}, {}, {}
    for p, bank in banks.items():
        def local_fields():
            aliases = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
            raw = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
            local = raw / .07
            broad = sample_broad(wide['clean', p], 0, 0, h, w).reshape_as(local)
            return aliases, raw, local, broad
        aliases, raw, local, broad = timer.call('local_text_and_sampling', local_fields)
        query = queries[p]
        members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = canonical_indices(query.class_names, query.aliases, query.parents)
        inputs = (local, operator, broad, crops['clean', p], count, fine[p] if fine else None,
                  fine_count, coordinates, coordinates, valid, members, canonical, query.parents, (h, w))
        values = {'Geometry': local.double()}
        stats, risk, observations = {}, None, None
        if verify_reference or 'ReferenceHard' in methods:
            original, original_risk, original_stats = timer.call('reference_reader', retained_scores, *inputs)
            values.update(original)
            values['ReferenceHard'] = original['RivalFineHard_Exact']
        if verify_reference or 'RivalFineHard_Exact' in methods:
            cached, risk, stats = timer.call('cached_reader', retained_scores_fast, *inputs)
            if verify_reference:
                if not torch.equal(risk, original_risk) or stats != original_stats:
                    raise RuntimeError('Risk/diagnostic equivalence failed: ' + p)
                errors[p] = {m: float((cached[m] - original[m]).abs().max()) for m in cached}
                if any(not torch.equal(cached[m], original[m]) for m in cached):
                    raise RuntimeError('Bitwise cached score equivalence failed: ' + p)
            values.update(cached)
        elif any(m.startswith('Fine') for m in methods):
            risk, observations = timer.call('cached_risk', cached_risk, crops['clean', p], count,
                fine[p], fine_count, coordinates, coordinates, valid, members, canonical, query.parents, (h, w))
        requested = tuple(m for m in methods if m in SOFT_METHODS)
        if requested:
            soft, soft_stats = timer.call('soft_reader', soft_scores, local, operator, broad, crops['clean', p], count,
                coordinates, valid, members, canonical, query.parents, (h, w), aliases, risk, requested, observations)
            values.update(soft)
        else:
            soft_stats = {}
        if 'NoAdmission_Exact' not in values:
            values['NoAdmission_Exact'] = timer.call('no_admission_reconstruction',
                lambda: local.double() + operator.double() @ (broad.double() - local.double()))
        scores[p] = {m: values[m] for m in methods}
        def assemble():
            output = {}
            for m in methods:
                source = raw if m == 'Geometry' else values[m]
                dense = F.interpolate(source.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                      mode='bilinear', align_corners=False)[0]
                if m == 'Geometry': dense = dense / .07
                output[m] = dense[:, :min(h, 512), :min(w, 512)].argmax(0).byte().cpu().numpy()
            return output
        predictions[p] = timer.call('dense_readout', assemble)
        diagnostics[p] = {'hard': stats, 'soft': soft_stats, **costs, 'operator_residual': residual,
                          'wide_forwards': len(crops['clean', p]), 'geometry_forwards': 1}
    return predictions, scores, diagnostics, errors, timer.seconds


@torch.inference_mode()
def benchmark(image, geometry, banks, vip, queries, repetitions):
    for method in COST_METHODS:
        window(image, geometry, banks, vip, queries, (method,))
    rows = {m: {'seconds': [], 'stage_seconds': [], 'peak_allocated_mib': []} for m in COST_METHODS}
    for repeat in range(repetitions):
        order = COST_METHODS if repeat % 2 == 0 else COST_METHODS[::-1]
        for method in order:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, _, diagnostics, _, stages = window(image, geometry, banks, vip, queries, (method,))
            torch.cuda.synchronize()
            rows[method]['seconds'].append(time.perf_counter() - started)
            rows[method]['stage_seconds'].append(stages)
            rows[method]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated()/1048576)
            rows[method]['forwards'] = {p: {f: d[f] for f in ('wide_forwards', 'fine_forwards', 'geometry_forwards')}
                                       for p, d in diagnostics.items()}
            del prediction
    for method, row in rows.items():
        row['median_seconds'] = statistics.median(row['seconds'])
        fields = set().union(*(r.keys() for r in row['stage_seconds']))
        row['stage_median_seconds'] = {f: statistics.median(r.get(f, 0.) for r in row['stage_seconds']) for f in fields}
        row['peak_allocated_mib'] = max(row['peak_allocated_mib'])
    return rows


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing output.')
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if prior['status'] != 'complete' or len(keys) != 8 or len(set(keys)) != 8:
        raise RuntimeError('Frozen eight-key development reference required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    states = frozen_state(geometry, vip)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Historical checkpoints differ.')
    vocabulary_sha = hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    transitions = {p: {m: np.zeros((bank.class_count,)*3, np.int64) for m in METHODS[2:]} for p, bank in banks.items()}
    ignored, diagnostics, replays = dict.fromkeys(banks, 0), [], []
    started = time.perf_counter()
    for i, key in enumerate(keys):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, scores, diag, errors, stages = window(image, geometry, banks, vip, queries, METHODS, verify_reference=True)
        with np.load(PREVIOUS/args.dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as old:
            replay = {p+'__'+m: float((scores[p][m] - torch.from_numpy(old['clean__'+p+'__'+m+'__scores']).to(args.device)).abs().max())
                      for p in banks for m in ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact')}
        if max(replay.values()) > 1e-4:
            raise RuntimeError('Historical score replay failed: ' + str(replay))
        # Persist scores before any target mask is loaded.
        cache = output/'scores'
        cache.mkdir(exist_ok=True)
        np.savez_compressed(cache/f'{i}.npz', sample_key=np.asarray(key),
                            **{p+'__'+m: s.cpu().numpy() for p, group in scores.items() for m, s in group.items()})
        diagnostics.append({'sample_key': key, 'protocols': diag, 'shared_stages': stages})
        replays.append({'sample_key': key, 'cached_bitwise_score_errors': errors, 'historical_score_errors': replay})
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for m in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count + predictions[p][m][valid]
                cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                matrices[p][m] += cm
                per_image[p+'__'+m].append(cm)
                if m in transitions[p]:
                    transitions[p][m] += transition_counts(predictions[p]['NoAdmission_Exact'], predictions[p][m], target, bank.class_count)
        result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': i+1, 'total_images': 8,
                  'sample_keys': keys[:i+1], 'methods': METHODS, 'admission_config': asdict(CONFIG),
                  'signature': {'checkpoints': checkpoint_manifest(checkpoints), 'vocabulary_sha256': vocabulary_sha,
                                'geometry': asdict(geometry.config), 'classes': {p: b.class_names for p, b in banks.items()},
                                'aliases': {p: b.alias_names for p, b in banks.items()}},
                  'metrics': {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()} for p, group in matrices.items()},
                  'transitions': {p: {m: {'counts': cm.tolist(), **transition_summary(cm)} for m, cm in group.items()} for p, group in transitions.items()},
                  'diagnostics': diagnostics, 'replay': replays, 'scores_persisted_before_masks': True,
                  'window': 'top-left512 of unchanged full image; full-image wide context', 'wall_seconds': time.perf_counter()-started}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': i+1, 'miou': {p: {m: x['mean_iou_percent'] for m, x in group.items()} for p, group in result['metrics'].items()}}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{k: np.stack(v) for k, v in per_image.items()})
    first = lookup[keys[0]]
    image = load_image(first.image_path if args.dataset == 'loveda' else first)
    result['benchmark'] = {'sample_key': first.key, 'source_image_size': list(image.shape[-2:]),
                           'scope': 'one512-window inference with full-image wide context; models/text/decoding/GT excluded; warmed independent arms',
                           'results': benchmark(image, geometry, banks, vip, queries, args.benchmark_repetitions)}
    if args.dataset == 'vdd':
        expected, old_diagnostics = predict_image(image, geometry, banks, vip, queries, output)
        actual, fast_diagnostics = predict_image_fast(image, geometry, banks, vip, queries, output)
        mismatches = {p: {m: int(np.count_nonzero(expected[p][m] != actual[p][m])) for m in expected[p]} for p in banks}
        if any(v for group in mismatches.values() for v in group.values()) or old_diagnostics != fast_diagnostics:
            raise RuntimeError('Full-image cached equivalence failed.')
        result['full_image_equivalence'] = {'image_size': list(image.shape[-2:]), 'pixel_mismatches': mismatches, 'diagnostics_equal': True}
    result.update(status='complete', coverage_verified=True, **check_frozen(states, geometry, vip),
                  wall_seconds=time.perf_counter()-started)
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--benchmark-repetitions', type=int, default=3)
    args = parser.parse_args()
    with torch.inference_mode():
        main(args)
