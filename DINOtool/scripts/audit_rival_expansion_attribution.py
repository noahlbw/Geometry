"""Cached20/40 class-restoration diagnosis with zero network forwards."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.prompts import load_class_specs
from dinotool.rival_expansion_attribution import (IMPLEMENTATION, METHODS,
    centered_restorations, common_count_shift)
from audit_alias_action_capacity import dense
from eval_geometry_semantic_innovation import parse_args
from eval_gear_ov import protocol
from eval_native_alias_noise import ORIGINAL
from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/rival_projected_vocabulary_stress_20261005'


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing attribution output.')
    prior = json.loads((SOURCE/args.dataset/'results.json').read_text())
    if (prior['status'] != 'complete' or not prior['coverage_verified']
            or not prior['historical_per_image_controls_exact']
            or not prior['source_scores_persisted_before_masks']):
        raise RuntimeError('Completed exact stress sources required.')
    samples, _, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    keys = prior['sample_keys']
    if len(keys) != 8 or len(set(keys)) != 8 or not set(keys).issubset(lookup):
        raise RuntimeError('Frozen eight actual input keys required.')
    output.mkdir(parents=True)
    (output/'scores').mkdir()
    matrices = {p: {m: {} for m in METHODS} for p in prior['metrics']['k20']}
    diagnostics = []
    started = time.perf_counter()
    with np.load(SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as historical:
        for index, key in enumerate(keys):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            scores, raw, packed = {}, {}, {'sample_key': np.asarray(key)}
            with np.load(SOURCE/args.dataset/'source_cache'/f'{index}.npz', allow_pickle=False) as cached, np.load(
                    ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original:
                if cached['sample_key'].item() != key:
                    raise RuntimeError('Cached sample identity changed.')
                operator = torch.from_numpy(original['operator']).to(args.device)
                offset = common_count_shift(operator)
                for p in matrices:
                    scores[p], raw[p] = {}, {}
                    for m in METHODS:
                        old, expanded = [torch.from_numpy(cached[s+'__'+p+'__'+m+'__scores']).to(args.device)
                                         for s in ('k20', 'k40')]
                        raw[p][m] = (old, expanded)
                        scores[p][m] = centered_restorations(old, expanded)
                        scores[p][m]['All40_CountOffsetRemoved'] = expanded-offset
                        packed.update({p+'__'+m+'__'+arm: value.cpu().numpy()
                                       for arm, value in scores[p][m].items()})
            np.savez_compressed(output/'scores'/f'{index}.npz', **packed)
            # All interventions are fixed and persisted before this audit reads masks.
            for p in matrices:
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                classes = len(prior['metrics']['k20'][p][METHODS[0]]['per_class'])
                valid = (target >= 0) & (target < classes)
                for m in METHODS:
                    predictions = {arm: dense(value).argmax(-1).cpu().numpy()
                                   for arm, value in scores[p][m].items()}
                    for arm, value in zip(('All20', 'All40'), raw[p][m]):
                        if not np.array_equal(predictions[arm], dense(value).argmax(-1).cpu().numpy()):
                            raise RuntimeError('Gauge centering changed an endpoint prediction.')
                    if not np.array_equal(predictions['All40'], predictions['All40_CountOffsetRemoved']):
                        raise RuntimeError('Common count offset changed an endpoint prediction.')
                    for arm, prediction in predictions.items():
                        encoded = target[valid]*classes+prediction[valid]
                        cm = np.bincount(encoded, minlength=classes**2).reshape(classes, classes)
                        if arm in ('All20', 'All40'):
                            scene = 'k20' if arm == 'All20' else 'k40'
                            if not np.array_equal(cm, historical[scene+'__'+p+'__'+m][index]):
                                raise RuntimeError('Historical per-image endpoint differs.')
                        matrices[p][m].setdefault(arm, []).append(cm)
            diagnostics.append({'sample_key': key, 'gauge_endpoint_predictions_exact': True,
                'common_count_offset_prediction_mismatches': 0, 'network_forwards': 0,
                'intervention_scores_persisted_before_masks': True})
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': index+1,
                'total_images': 8, 'sample_keys': keys[:index+1], 'source_signature': prior['base_signature'],
                'vocabulary_source_sha256': prior['vocabulary_source_sha256'], 'methods': METHODS,
                'class_names': {p: [c['name'] for c in prior['metrics']['k20'][p][METHODS[0]]['per_class']]
                                for p in matrices}, 'diagnostics': diagnostics,
                'metrics': {p: {m: {a: np.stack(v).sum(0).tolist() for a, v in arms.items()}
                                    for m, arms in methods.items()} for p, methods in matrices.items()},
                'wall_seconds': time.perf_counter()-started, 'new_selector': False}
            save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': index+1}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
        **{p+'__'+m+'__'+a: np.stack(v) for p, methods in matrices.items()
           for m, arms in methods.items() for a, v in arms.items()})
    result.update(status='complete', coverage_verified=True)
    save(output/'results.json', result)


if __name__ == '__main__':
    main(parse_args())
