"""One prospectively fixed influence-weighted reader; retained controls are unchanged."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import FunctionType

import numpy as np
import torch

from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_influence_soft import IMPLEMENTATION, PRIMARY, soft_scores
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import summary
from eval_rival_alias_speed_soft import window as original_window
from eval_rival_fine_full import frozen_state, check_frozen, save
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/rival_alias_speed_soft_20261005'
scope = dict(original_window.__wrapped__.__globals__, SOFT_METHODS=(PRIMARY,), soft_scores=soft_scores)
window = FunctionType(original_window.__wrapped__.__code__, scope,
                      original_window.__wrapped__.__name__, original_window.__wrapped__.__defaults__)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing output.')
    prior = json.loads((SOURCE/args.dataset/'results.json').read_text())
    if prior['status'] != 'complete' or prior['processed_images'] != 8:
        raise RuntimeError('Complete unchanged source64 study required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Changed source checkpoint identity.')
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'scores').mkdir()
    matrices = {p: np.zeros((b.class_count,)*2, np.int64) for p, b in banks.items()}
    transitions = {p: np.zeros((b.class_count,)*3, np.int64) for p, b in banks.items()}
    per_image, ignored, diagnostics = {p+'__'+PRIMARY: [] for p in banks}, dict.fromkeys(banks, 0), []
    started = time.perf_counter()
    with np.load(SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as source:
        for i, key in enumerate(prior['sample_keys']):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            preds, scores, stats, _, stages = window(image, geometry, banks, vip, queries, ('NoAdmission_Exact', PRIMARY))
            with np.load(SOURCE/args.dataset/'scores'/f'{i}.npz', allow_pickle=False) as old:
                for p in banks:
                    if not np.array_equal(scores[p]['NoAdmission_Exact'].cpu().numpy(), old[p+'__NoAdmission_Exact']):
                        raise RuntimeError('Original unscreened score changed.')
            np.savez_compressed(output/'scores'/f'{i}.npz', **{p+'__'+PRIMARY: s[PRIMARY].cpu().numpy() for p, s in scores.items()})
            for p, bank in banks.items():
                target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
                valid = (target >= 0) & (target < bank.class_count)
                ignored[p] += int((~valid).sum())
                for m in ('NoAdmission_Exact', PRIMARY):
                    enc = target[valid].astype(np.int64)*bank.class_count+preds[p][m][valid]
                    cm = np.bincount(enc, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                    if m == 'NoAdmission_Exact':
                        if not np.array_equal(cm, source[p+'__'+m][i]):
                            raise RuntimeError('Original unscreened prediction changed.')
                    else:
                        matrices[p] += cm
                        per_image[p+'__'+m].append(cm)
                        transitions[p] += transition_counts(preds[p]['NoAdmission_Exact'], preds[p][m], target, bank.class_count)
            diagnostics.append({'sample_key': key, 'protocols': stats, 'stages': stages})
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'method': PRIMARY,
                'processed_images': i+1, 'total_images': 8, 'sample_keys': prior['sample_keys'][:i+1], 'signature': prior['signature'],
                'scores_persisted_before_masks': True, 'source_no_admission_scores_and_predictions_exact': True,
                'metrics': {p: summary(cm, banks[p].class_names, ignored[p]) for p, cm in matrices.items()},
                'transitions': {p: {'counts': cm.tolist(), **transition_summary(cm)} for p, cm in transitions.items()}, 'diagnostics': diagnostics}
            save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': i+1, 'miou': {p: m['mean_iou_percent'] for p, m in result['metrics'].items()}}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(prior['sample_keys']), **{k: np.stack(v) for k, v in per_image.items()})
    first = lookup[prior['sample_keys'][0]]
    image = load_image(first.image_path if args.dataset == 'loveda' else first)
    window(image, geometry, banks, vip, queries, (PRIMARY,))
    seconds, stages, peaks = [], [], []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begun = time.perf_counter()
        _, _, _, _, timing = window(image, geometry, banks, vip, queries, (PRIMARY,))
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-begun)
        stages.append(timing)
        peaks.append(torch.cuda.max_memory_allocated()/1048576)
    result['benchmark'] = {'sample_key': first.key, 'seconds': seconds, 'median_seconds': statistics.median(seconds),
                           'stage_seconds': stages, 'peak_allocated_mib': max(peaks), 'scope': prior['benchmark']['scope']}
    result.update(status='complete', coverage_verified=True, wall_seconds=time.perf_counter()-started, **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    main(parser.parse_args())
