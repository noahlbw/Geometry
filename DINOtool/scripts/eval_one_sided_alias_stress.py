"""Exact incumbent stress on96 complete developed inputs; no new selector."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_grouped_class_readout import CheckedGroupedCache
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.model import checkpoint_manifest
from dinotool.one_sided_alias_audit import predict_image as original_predict
from dinotool.one_sided_alias_stress import (IMPLEMENTATION, METHODS, PRIMARY,
    SCENARIOS, predict_image, validate_variants)
from eval_bounded_crop_head_alias import panel_inputs
from eval_gear_ov import digest
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_rival_projected_vocabulary_stress import SOURCE, VOCAB, variants_for


def inputs(args):
    values = panel_inputs(args)
    output, samples, load_image, load_mask, geometry, banks, vip, queries, checkpoints, old = values
    source = json.loads(VOCAB.read_text())
    prior = json.loads((SOURCE/args.dataset/'merged.json').read_text())
    variants, vocabulary = variants_for(args.dataset, source, prior, banks, vip, queries)
    validate_variants(banks, variants, vocabulary)
    manifest = dict(local_vocabulary=old['signature']['vocabulary'], context_vocabularies=vocabulary,
        historical_source_sha256=hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
        context_counts={s: {p: [len(c['synonyms']) for c in classes]
                           for p, classes in protocols.items()} for s, protocols in vocabulary.items()})
    return values, variants, manifest


def verify_diagnostics(diagnostics):
    for group in diagnostics.values():
        for row in group.values():
            if (not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4
                    or row['wide_encodings'] > 4 or row['native_resolution_encodings'] != 0
                    or row['canonical_risk_max'] != 0 or row['fine_coverage_max_error'] > 1e-6
                    or row['positive_directed_delta_max'] > 1e-12 or row['class_budget_max_error'] > 1e-10
                    or row['rival_budget_max_error'] > 1e-10 or row['matched_unmatchable'] != 0
                    or row['matched_norm_relative_error'] > 1e-10
                    or row['normal_equation_max_error'] > 1e-10 or row['gauge_max_error'] > 1e-10):
                raise RuntimeError('Shared visual budget, protection or matched action invariant failed.')


@torch.inference_mode()
def smoke(args):
    values, variants, manifest = inputs(args)
    output, samples, load_image, _, geometry, banks, vip, _, _, _ = values
    state = frozen_state(geometry, vip)
    image = load_image(samples[0].image_path if args.dataset == 'loveda' else samples[0])
    execution = FineCoverageExecution(cached=True, burst=True)
    actual, diagnostics = predict_image(image, geometry, banks, vip, variants, execution=execution)
    for scenario, (_, queries) in variants.items():
        independent, _ = original_predict(image, geometry, banks, vip, queries,
                                           methods=METHODS, execution=execution)
        for p in banks:
            for m in METHODS:
                if not np.array_equal(actual[scenario][p][m], independent[p][m]):
                    raise RuntimeError('Independent scenario prediction changed: '+scenario+'/'+p+'/'+m)
    singleton, _ = predict_image(image, geometry, banks, vip,
        {'wrong_parent': variants['wrong_parent']}, methods=(PRIMARY,), execution=execution)
    if any(not np.array_equal(actual['wrong_parent'][p][PRIMARY], singleton['wrong_parent'][p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs from shared scenarios.')
    verify_diagnostics(diagnostics)
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=samples[0].key, target_masks_loaded=False, original_endpoints_exact=True,
        all_scenarios_independent_exact=True, singleton_primary_exact=True,
        manifest=manifest, diagnostics=diagnostics, execution=execution.report(),
        **check_frozen(state, geometry, vip)))


@torch.inference_mode()
def evaluate(args):
    values, variants, manifest = inputs(args)
    output, samples, load_image, load_mask, geometry, banks, vip, _, checkpoints, _ = values
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError('This developed stress panel has one shard per domain.')
    state = frozen_state(geometry, vip)
    keys = [s.key for s in samples]
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={p: bank.class_names for p, bank in banks.items()},
        geometry=asdict(geometry.config), checkpoints=checkpoint_manifest(checkpoints),
        sample_keys=keys, sample_keys_sha256=digest(keys), global_sample_keys_sha256=digest(keys),
        global_sample_count=len(keys), num_shards=1, shard_index=0, config=vars(args), **manifest)
    matrices = {s: {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS}
                   for p, bank in banks.items()} for s in SCENARIOS}
    per_image = {s+'__'+p+'__'+m: [] for s in SCENARIOS for p in banks for m in METHODS}
    ignored, totals = dict.fromkeys(banks, 0), {s: {p: dict(tiles=0) for p in banks} for s in SCENARIOS}
    cache, execution = CheckedGroupedCache(), FineCoverageExecution(cached=True, burst=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostics = predict_image(image, geometry, banks, vip, variants,
                                                 execution=execution, cache=cache)
        verify_diagnostics(diagnostics)
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for s in SCENARIOS:
                for m in METHODS:
                    encoded = target[valid].astype(np.int64)*bank.class_count+predictions[s][p][m][valid]
                    cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                    matrices[s][p][m] += cm
                    per_image[s+'__'+p+'__'+m].append(cm)
                row, tile_count = totals[s][p], diagnostics[s][p]['tiles']
                row['tiles'] += tile_count
                for field, value in diagnostics[s][p].items():
                    if field != 'tiles':
                        row[field] = row.get(field, 0.)+tile_count*value
        if number == 1 or number % 5 == 0 or number == len(samples):
            result = dict(status='running', implementation=IMPLEMENTATION, signature=signature,
                processed_images=number, total_images=len(samples),
                metrics={s: {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                             for p, group in protocols.items()} for s, protocols in matrices.items()},
                diagnostics={s: {p: {f: v if f == 'tiles' else v/row['tiles'] for f, v in row.items()}
                                 for p, row in group.items()} for s, group in totals.items()},
                target_masks_used_only_after_prediction=True, scenarios_share_visual_observations=True,
                local_geometry_uses_unchanged20=True, rule_changed=False,
                wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
            save(output/'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(samples))), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result.update(status='complete', coverage_verified=len(keys) == len(set(keys)),
                  execution=execution.report(), **check_frozen(state, geometry, vip))
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
