"""Compare qualified queries to v2 at identical settings using the existing reader."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dinotool.context_sense_evaluation import confusion
from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
from dinotool.natural_text_adaptation import SEED
from dinotool.qualified_sense_words import IMPLEMENTATION, paired_inputs, rename_predictions
from dinotool.rival_fine_full import IMPLEMENTATION as CORE
from dinotool.vip_official_adapter import VIPQueries
from eval_natural_sense_adaptation import prepare, predict, sha
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_vip_official_eight import digest


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Existing qualified-word output; no overwrite.')
    loaded = prepare(args)
    geometry, vip, candidate_queries, source_profiles, _, _, _, _, identity = loaded
    baseline = json.loads(Path(args.baseline_vocabulary).read_text())
    if (tuple(baseline['class_names']) != tuple(identity['classes']) or baseline['target_masks_loaded']
            or baseline['image_data_loaded'] or baseline['target_label_tuning']):
        raise ValueError('Aligned metadata-only v2 query baseline required.')
    cached = torch.load(args.baseline_cache, map_location=geometry.device, weights_only=True)
    expected = dict(source_identity=identity, vocabulary_sha256=sha(args.baseline_vocabulary), family='seg_template')
    if cached['identity'] != expected:
        raise ValueError('Changed baseline text-cache/vocabulary/checkpoint identity.')
    old = VIPQueries(cached['features'], cached['parents'], tuple(baseline['class_names']),
                     tuple(a for g in baseline['aliases_by_class'] for a in g))
    queries, profiles = paired_inputs(candidate_queries['default'], old, source_profiles['default']['prob_thd'])
    states = frozen_state(geometry, vip)
    samples = discover_samples(args.dataset, args.data_root)
    if args.smoke:
        samples = [random.Random(SEED+1).choice(samples)]
    keys = [s.key for s in samples]
    if len(keys) != len(set(keys)) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Unique sample keys and valid shard indices required.')
    chosen = samples[args.shard_index::args.num_shards]
    if not chosen:
        raise ValueError('Empty word-comparison shard.')
    output.mkdir(parents=True)
    started, ignored, tiles, action = time.perf_counter(), 0, 0, 0.
    matrices, arrays, classes = {}, {}, len(identity['classes'])
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        predictions, diag, _ = predict(image, geometry, vip, queries, profiles, output)
        predictions = rename_predictions(predictions)
        if number == 1:
            control, _, _ = predict(image, geometry, vip, {'default': old}, {'default': profiles['frozen']}, output)
            for suffix in ('', '_NoThreshold'):
                if not np.array_equal(predictions['V2_Words'+suffix], control['Sense_Default'+suffix]):
                    raise ValueError('Standalone v2 versus paired-reader equivalence failed.')
        if any(p.shape != tuple(image.shape[-2:]) or np.any(p >= classes) for p in predictions.values()):
            raise ValueError('Incomplete full-image prediction or changed categories.')
        if args.smoke:
            save(output/'smoke.json', dict(status='complete', implementation=IMPLEMENTATION,
                processed_images=1, total_images=1, image_keys=keys, source_equivalence_first_image_verified=True,
                target_masks_loaded=False, target_label_tuning=False, main_model_changed=False,
                profiles=profiles, baseline_vocabulary_sha256=sha(args.baseline_vocabulary),
                candidate_vocabulary_sha256=sha(args.candidate_vocabulary), **check_frozen(states, geometry, vip)))
            return
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        ignored += int(((target < 0) | (target >= classes)).sum())
        for method, prediction in predictions.items():
            matrix, _ = confusion(target, prediction, classes)
            matrices.setdefault(method, np.zeros_like(matrix))[:] += matrix
            arrays.setdefault(method, []).append(matrix)
        tiles += diag['tiles']
        action += diag['mean_absolute_admission_potential']*diag['tiles']
        signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=list(matrices),
            classes={args.dataset: identity['classes']}, checkpoints=identity['checkpoints'],
            gear=dict(core_implementation=CORE, geometry=asdict(geometry.config), profiles=profiles,
                main_model_changed=False, source_selection_sha256=sha(args.source_selection),
                baseline_vocabulary_sha256=sha(args.baseline_vocabulary),
                candidate_vocabulary_sha256=sha(args.candidate_vocabulary)),
            vocabulary={k: dict(aliases=q.aliases, parents=q.parents.tolist()) for k, q in queries.items()},
            global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
            sample_keys=[s.key for s in chosen], sample_keys_sha256=digest([s.key for s in chosen]),
            num_shards=args.num_shards, shard_index=args.shard_index, partial_run=False, config=vars(args))
        result = dict(status='running', processed_images=number, total_images=len(chosen), signature=signature,
            metrics={args.dataset: {m: summary(cm, identity['classes'], ignored) for m, cm in matrices.items()}},
            diagnostics={args.dataset: dict(tiles=tiles, mean_absolute_admission_potential=action/tiles)},
            source_equivalence_first_image_verified=True, target_masks_used_only_after_prediction=True,
            target_label_tuning=False, wall_seconds=time.perf_counter()-started,
            peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
        if number == 1 or number % 10 == 0 or number == len(chosen):
            save(output/'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(chosen))), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{args.dataset+'__'+m: np.stack(a) for m, a in arrays.items()})
    result.update(status='complete', **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir',
                 'baseline-vocabulary', 'baseline-cache'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    main(parser.parse_args())
