"""Transfer frozen sense inputs and actual-reader calibration to Context59/60."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dinotool.context_sense_evaluation import IMPLEMENTATION, confusion
from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
from dinotool.natural_text_adaptation import SEED
from dinotool.rival_fine_full import IMPLEMENTATION as CORE
from eval_natural_sense_adaptation import prepare, predict, verify_source, sha, main as select
from eval_natural_text_adaptation import official_prediction
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_vip_official_eight import digest


@torch.inference_mode()
def evaluate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Existing Context evaluation output; no overwrite.')
    calibration = json.loads(Path(args.selection).read_text())
    if (calibration['status'] != 'complete' or calibration['target_masks_loaded'] or calibration['smoke_only']
            or calibration['source_selection_sha256'] != sha(args.source_selection)
            or calibration['candidate_vocabulary_sha256'] != sha(args.candidate_vocabulary)):
        raise ValueError('Frozen full image-only Context calibration required.')
    loaded = prepare(args)
    geometry, vip, queries, profiles, source, _, reference, settings, identity = loaded
    states = frozen_state(geometry, vip)
    samples = discover_samples(args.dataset, args.data_root)
    if args.max_images:
        selected_keys = set(random.Random(SEED+1).sample([s.key for s in samples], min(args.max_images, len(samples))))
        samples = [s for s in samples if s.key in selected_keys]
    keys = [s.key for s in samples]
    chosen = samples[args.shard_index::args.num_shards]
    if not chosen or len(keys) != len(set(keys)) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Valid unique nonempty Context shard required.')
    output.mkdir(parents=True)
    matrices, arrays, ignored, tiles, actions = {}, {}, 0, 0, 0.
    started, classes = time.perf_counter(), len(identity['classes'])
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        predictions, diag, _ = predict(image, geometry, vip, queries, profiles, output, calibration)
        predictions['VIP_Official_Finite'], predictions['VIP_Official_NoThreshold'] = official_prediction(image, vip, reference, settings)
        if number == 1:
            verify_source(image, predictions, loaded, output)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        ignored += int(((target < 0) | (target >= classes)).sum())
        for method, pred in predictions.items():
            reject = method == 'VIP_Official_Finite' and not settings.background and settings.prob_thd > 0
            cm, _ = confusion(target, pred, classes, reject)
            matrices.setdefault(method, np.zeros_like(cm))[:] += cm
            arrays.setdefault(method, []).append(cm)
        tiles += diag['tiles']
        actions += diag['mean_absolute_admission_potential']*diag['tiles']
        signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=list(matrices),
            classes={args.dataset: identity['classes']}, checkpoints=identity['checkpoints'],
            gear=dict(core_implementation=CORE, geometry=asdict(geometry.config), profiles=profiles,
                calibration=calibration['profile'], background=calibration['background'],
                source_selection_sha256=sha(args.source_selection), selection_sha256=sha(args.selection),
                candidate_vocabulary_sha256=sha(args.candidate_vocabulary), main_model_changed=False,
                source_input_type=source['source_type'], reference_settings=asdict(settings),
                official_reference='Original Context city templates, queries/settings and resize; explicit empty-row Self-Value repair.'),
            vocabulary={k: dict(aliases=q.aliases, parents=q.parents.tolist()) for k, q in queries.items()},
            global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
            sample_keys=[s.key for s in chosen], sample_keys_sha256=digest([s.key for s in chosen]),
            num_shards=args.num_shards, shard_index=args.shard_index, partial_run=bool(args.max_images), config=vars(args))
        result = dict(status='running', processed_images=number, total_images=len(chosen), signature=signature,
            metrics={args.dataset: {m: summary(cm, identity['classes'], ignored) for m, cm in matrices.items()}},
            diagnostics={args.dataset: dict(tiles=tiles, mean_absolute_admission_potential=actions/tiles)},
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
    parser.add_argument('action', choices=('select', 'evaluate'))
    parser.add_argument('--dataset', choices=('context59', 'context60'), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    select(args) if args.action == 'select' else evaluate(args)
