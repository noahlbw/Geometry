"""Paired full-image tests of frozen text curation and external count calibration."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.natural_alias_calibration import IMPLEMENTATION, validate_selections
from dinotool.natural_count_calibration import calibrate_coupled_counts
from dinotool.natural_evaluation import TOTALS, discover_samples, load_rgb, load_target
from dinotool.natural_text_adaptation import SEED, class_logits
from dinotool.natural_variable_alias_reader import retained_variable_scores
from dinotool.rival_fine_full import IMPLEMENTATION as CORE
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_vip_reliability import sample_broad, summary
from eval_natural_text_adaptation import bank_from_query, models, predict as reference_predict, subset
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, tile_reference_coordinates
from eval_stratified_soft_alias import tile_coordinates
from eval_vip_official_eight import digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def predict_paired(image, geometry, vip, queries, choice, strength, work):
    h, w = image.shape[-2:]
    banks = {k: bank_from_query(q)[0] for k, q in queries.items()}
    wide, crops, _, count = prepare_wide(image, vip, {'text': (banks, queries)})
    for variant, query in queries.items():
        dense_map = torch.zeros_like(wide['text', variant])
        for crop in crops['text', variant]:
            logits = class_logits(crop.alias_logits, crop.salience, query.parents,
                                  len(query.class_names), choice['tau'], choice['tem'])
            dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336),
                                  mode='bilinear', align_corners=False)[0]
            dense_map[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
        wide['text', variant] = dense_map / count
    tile_methods = ('Source_NoThreshold', 'Count_Selected_NoThreshold', 'Count_FullNorm_NoThreshold')
    if 'curated' in queries:
        tile_methods += ('Curated_NoThreshold',)
    blend = hann_blend_window(512)
    tiles, actions = 0, 0.
    with ExitStack() as stack:
        acc = {m: stack.enter_context(ProbabilityAccumulator(len(queries['source'].class_names), h, w, 256, work))
               for m in tile_methods}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coords = tile_coordinates(top, left, geometry.device)
                valid = (coords[:, 0] < h) & (coords[:, 1] < w)
                fine_coords = tile_reference_coordinates(coords, top, left)
                fine, fine_count, _ = observe_fine(image[:, top:top+512, left:left+512], vip,
                                                  queries, fine_coords, valid)
                operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                scores = {}
                for variant, bank in banks.items():
                    raw = alias_class_scores((prepared.geometry_projected.float() @ bank.features.T)[0],
                                             bank.parent_indices, bank.class_count)/.07
                    broad = sample_broad(wide['text', variant], top, left, h, w).reshape_as(raw)
                    selected_wide = [replace(c, salience=c.salience/choice['tem']) for c in crops['text', variant]]
                    selected_fine = [replace(c, salience=c.salience/choice['tem']) for c in fine[variant]]
                    values, stats = retained_variable_scores(raw, operator, broad, selected_wide, count,
                        selected_fine, fine_count, coords, fine_coords, valid, bank.parent_indices,
                        bank.canonical_mask.nonzero().flatten(), (h, w), beta=choice['tau'])
                    control = values['RivalFineHard_Exact']
                    if variant == 'source':
                        counts = torch.bincount(bank.parent_indices, minlength=bank.class_count)
                        scores['Source_NoThreshold'] = control
                        scores['Count_Selected_NoThreshold'] = calibrate_coupled_counts(control, operator, counts, choice['tau'], strength)
                        scores['Count_FullNorm_NoThreshold'] = calibrate_coupled_counts(control, operator, counts, choice['tau'], 1.)
                        actions += stats['mean_absolute_admission_potential']
                    else:
                        scores['Curated_NoThreshold'] = control
                ah, aw = min(512, h-top), min(512, w-left)
                for method, logits in scores.items():
                    dense = F.interpolate(logits.T.reshape(1, -1, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    acc[method].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(),
                                    blend[:ah, :aw], left, top)
                tiles += 1
        predictions = {}
        for method, accumulator in acc.items():
            predictions[method] = accumulator.finalize(None)[0]
            predictions[method.removesuffix('_NoThreshold')] = accumulator.finalize(choice['prob_thd'], background_index=0)[0]
    return predictions, dict(tiles=tiles, mean_absolute_admission_potential=actions/tiles)


@torch.inference_mode()
def evaluate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Existing paired evaluation output; refusing overwrite.')
    source = json.loads(Path(args.source_selection).read_text())
    count = json.loads(Path(args.count_selection).read_text())
    curated = json.loads(Path(args.curated_selection).read_text()) if args.curated_selection else None
    strength = validate_selections(source, count, curated, sha(args.source_selection))
    geometry, vip, encoded, reference, settings, identity, _ = models(args)
    if json.loads(json.dumps(identity)) != source['identity'] or args.dataset != source['dataset']:
        raise ValueError('Changed source text/checkpoints or dataset.')
    states = frozen_state(geometry, vip)
    choice = source['chosen']
    query = encoded[choice['template']]
    queries = {'source': subset(query, torch.tensor(source['selected_indices'], device=geometry.device))}
    if curated is not None:
        queries['curated'] = subset(query, torch.tensor(curated['selected_indices'], device=geometry.device))
    samples = discover_samples(args.dataset, args.data_root)
    if args.max_images:
        keys = set(random.Random(SEED+1).sample([s.key for s in samples], min(args.max_images, len(samples))))
        samples = [s for s in samples if s.key in keys]
    keys = [s.key for s in samples]
    chosen = samples[args.shard_index::args.num_shards]
    if not chosen or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Nonempty valid evaluation shard required.')
    output.mkdir(parents=True)
    started, ignored = time.perf_counter(), 0
    matrices, arrays, tile_total, action_sum = {}, {}, 0, 0.
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        predictions, diagnostics = predict_paired(image, geometry, vip, queries, choice, strength, output)
        if number == 1:
            control, _ = reference_predict(image, geometry, vip, encoded['seg_template'],
                                           queries['source'], reference, settings, choice, output)
            for ours, original in (('Source_NoThreshold', 'Adaptive_NoThreshold'), ('Source', 'Adaptive_RivalFine')):
                if not np.array_equal(predictions[ours], control[original]):
                    raise RuntimeError('Source predictions differ from the existing frozen reader: '+ours)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        classes = len(identity['classes'])
        valid = (target >= 0) & (target < classes)
        ignored += int((~valid).sum())
        for method, pred in predictions.items():
            if pred.shape != target.shape or np.any(pred[valid] >= classes):
                raise ValueError('Changed full-image coverage or out-of-range class.')
            cm = np.bincount(target[valid]*classes+pred[valid], minlength=classes**2).reshape(classes, classes)
            matrices.setdefault(method, np.zeros_like(cm))[:] += cm
            arrays.setdefault(method, []).append(cm)
        tile_total += diagnostics['tiles']
        action_sum += diagnostics['mean_absolute_admission_potential']*diagnostics['tiles']
        signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=list(matrices),
            classes={args.dataset: identity['classes']}, checkpoints=identity['checkpoints'],
            gear=dict(core_implementation=CORE, geometry=asdict(geometry.config), selected_calibration=choice,
                      source_selection_sha256=sha(args.source_selection), count_selection_sha256=sha(args.count_selection),
                      curated_selection_sha256=sha(args.curated_selection) if curated is not None else None,
                      count_strength=strength, full_normalization_arm='Diagnostic only; never selects the retained strength.',
                      local_tiles=[512, 128], wide_view=[448, 336, 112], fine_physical_pixels_per_token=8),
            vocabulary={k: dict(aliases=q.aliases, parents=q.parents.tolist()) for k, q in queries.items()},
            global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
            sample_keys=[s.key for s in chosen], sample_keys_sha256=digest([s.key for s in chosen]),
            num_shards=args.num_shards, shard_index=args.shard_index, partial_run=bool(args.max_images),
            config=vars(args), selection_masks_used=False)
        result = dict(status='running', processed_images=number, total_images=len(chosen), signature=signature,
            metrics={args.dataset: {m: summary(cm, identity['classes'], ignored) for m, cm in matrices.items()}},
            diagnostics={args.dataset: dict(tiles=tile_total, mean_absolute_admission_potential=action_sum/tile_total)},
            wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
            source_equivalence_first_image_verified=True, target_masks_used_only_after_prediction=True,
            target_label_tuning=False)
        save(output/'results.json', result)
        print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(chosen),
                              miou={m: x['mean_iou_percent'] for m, x in result['metrics'][args.dataset].items()})), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{args.dataset+'__'+m: np.stack(a) for m, a in arrays.items()})
    result.update(status='complete', **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(TOTALS), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection', 'count-selection'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--curated-selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=16)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    evaluate(parser.parse_args())
