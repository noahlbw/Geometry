"""Paired semantic-input calibration using the unchanged deployed full-image reader."""
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
from dinotool.natural_evaluation import TOTALS, discover_samples, load_rgb, load_target
from dinotool.natural_sense_calibration import (IMPLEMENTATION, choose_profile, foreground_thresholds,
    finalize_thresholds, sample_accumulated_probabilities, witness_error)
from dinotool.natural_text_adaptation import SEED, CALIBRATION_IMAGES, class_logits, trusted_witnesses
from dinotool.natural_variable_alias_reader import retained_variable_scores
from dinotool.rival_fine_full import IMPLEMENTATION as CORE
from dinotool.vip_official_adapter import VIPQueries, _load_upstream
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_vip_reliability import sample_broad, summary
from eval_natural_text_adaptation import (models, bank_from_query, subset, sampled_aliases,
                                        predict as source_predict)
from eval_rival_fine_full import observe_fine, tile_reference_coordinates, frozen_state, check_frozen, save
from eval_stratified_soft_alias import tile_coordinates
from eval_vip_official_eight import digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    source = json.loads(Path(args.source_selection).read_text())
    vocabulary = json.loads(Path(args.candidate_vocabulary).read_text())
    if source['status'] != 'complete' or source['target_masks_loaded'] or source['target_label_tuning']:
        raise ValueError('Complete mask-free frozen source required.')
    if vocabulary['dataset'] != args.dataset or vocabulary['class_names'] != source['identity']['classes']:
        raise ValueError('Candidate annotation classes/order changed.')
    groups = vocabulary['aliases_by_class']
    if len(groups) != len(vocabulary['class_names']) or any(not g or any(not a.strip() for a in g) for g in groups):
        raise ValueError('Every candidate class requires nonempty descriptions.')
    geometry, vip, encoded, reference, settings, identity, _ = models(args)
    if json.loads(json.dumps(identity)) != source['identity']:
        raise ValueError('Source checkpoint or vocabulary identity changed.')
    cache = Path(args.candidate_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    prompt = _load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py', 'sense_input_templates')
    candidates = {}
    for family in dict.fromkeys(('seg_template', source['chosen']['template'])):
        path = cache/(family+'.pt')
        expected = dict(source_identity=identity, vocabulary_sha256=sha(args.candidate_vocabulary), family=family)
        if path.exists():
            row = torch.load(path, map_location=geometry.device, weights_only=True)
            if row['identity'] != expected:
                raise ValueError('Candidate text cache changed.')
            query = VIPQueries(row['features'], row['parents'], tuple(vocabulary['class_names']),
                               tuple(a for group in groups for a in group))
        else:
            vip.templates = prompt.get_text_template(family)
            query = vip.encode_queries(tuple(vocabulary['class_names']), groups)
            torch.save(dict(identity=expected, features=query.features.cpu(), parents=query.parents.cpu()), path)
        candidates[family] = query
    indices = torch.tensor(source['selected_indices'], device=geometry.device)
    queries = dict(source=subset(encoded[source['chosen']['template']], indices),
                   default=candidates['seg_template'], frozen=candidates[source['chosen']['template']])
    profiles = dict(source=source['chosen'], frozen=source['chosen'],
                    default=dict(template='seg_template', tau=1., tem=1., prob_thd=source['chosen']['prob_thd']))
    return geometry, vip, queries, profiles, source, encoded, reference, settings, identity


@torch.inference_mode()
def predict(image, geometry, vip, queries, profiles, work, calibrated=None, witnesses=False):
    h, w = image.shape[-2:]
    banks = {k: bank_from_query(q)[0] for k, q in queries.items()}
    _, crops, _, count = prepare_wide(image, vip, {'text': (banks, queries)})
    wide = {}
    for key, query in queries.items():
        dense_map = torch.zeros(len(query.class_names), *count.shape, device=geometry.device)
        p = profiles[key]
        for crop in crops['text', key]:
            logits = class_logits(crop.alias_logits, crop.salience, query.parents, len(query.class_names), p['tau'], p['tem'])
            dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336), mode='bilinear', align_corners=False)[0]
            dense_map[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
        wide[key] = dense_map/count
    center = (max(0, (h-512)//2), max(0, (w-512)//2))
    ys, xs = tile_starts(h, 512, 128), tile_starts(w, 512, 128)
    witness_tile = (min(ys, key=lambda y: abs(y-center[0])), min(xs, key=lambda x: abs(x-center[1])))
    blend, tiles, actions = hann_blend_window(512), 0, 0.
    witness = None
    with ExitStack() as stack:
        acc = {key: stack.enter_context(ProbabilityAccumulator(len(q.class_names), h, w, 256, work))
               for key, q in queries.items()}
        for top in ys:
            for left in xs:
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coords = tile_coordinates(top, left, geometry.device)
                valid = (coords[:, 0] < h) & (coords[:, 1] < w)
                fine_coords = tile_reference_coordinates(coords, top, left)
                fine, fine_count, _ = observe_fine(image[:, top:top+512, left:left+512], vip, queries, fine_coords, valid)
                operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                ah, aw = min(512, h-top), min(512, w-left)
                for key, bank in banks.items():
                    alias = (prepared.geometry_projected.float() @ bank.features.T)[0]
                    raw = alias_class_scores(alias, bank.parent_indices, bank.class_count)/.07
                    p = profiles[key]
                    values, stats = retained_variable_scores(raw, operator,
                        sample_broad(wide[key], top, left, h, w).reshape_as(raw),
                        [replace(c, salience=c.salience/p['tem']) for c in crops['text', key]], count,
                        [replace(c, salience=c.salience/p['tem']) for c in fine[key]], fine_count,
                        coords, fine_coords, valid, bank.parent_indices,
                        bank.canonical_mask.nonzero().flatten(), (h, w), beta=p['tau'])
                    dense = F.interpolate(values['RivalFineHard_Exact'].T.reshape(1, -1, 32, 32),
                                          (512, 512), mode='bilinear', align_corners=False)[0]
                    acc[key].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(), blend[:ah, :aw], left, top)
                    actions += stats['mean_absolute_admission_potential']/len(queries)
                    if witnesses and key == 'default' and (top, left) == witness_tile:
                        canonical = bank.canonical_mask.nonzero().flatten()
                        labels, quality, _ = trusted_witnesses(alias[valid]*40,
                            sampled_aliases(crops['text', key], count, coords[valid], (h, w)),
                            canonical, bank.parent_indices, bank.class_count, bank.class_names[0] == 'background')
                        witness = dict(coordinates=coords[valid].cpu().numpy(), labels=labels.cpu().numpy(),
                                       quality=quality.cpu().numpy())
                tiles += 1
        predictions = {}
        display = dict(source='Source', default='Sense_Default', frozen='Sense_Words')
        for key, accumulator in acc.items():
            predictions[display[key]+'_NoThreshold'] = accumulator.finalize(None)[0]
            predictions[display[key]] = accumulator.finalize(profiles[key]['prob_thd'], background_index=0)[0]
        if calibrated is not None:
            key = calibrated['profile']['profile']
            predictions['Sense_Calibrated_NoThreshold'] = predictions[display[key]+'_NoThreshold']
            predictions['Sense_Calibrated'] = finalize_thresholds(acc[key], calibrated['background']['thresholds'],
                                                                  queries[key].class_names[0] == 'background')
        if witnesses:
            if witness is None:
                raise RuntimeError('Missing image-only witness tile.')
            witness['probabilities'] = {key: sample_accumulated_probabilities(acc[key], witness['coordinates'])
                                        for key in ('default', 'frozen')}
    return predictions, dict(tiles=tiles, mean_absolute_admission_potential=actions/tiles), witness


def verify_source(image, predictions, loaded, output):
    geometry, vip, queries, profiles, source, encoded, reference, settings, _ = loaded
    control, _ = source_predict(image, geometry, vip, encoded['seg_template'], queries['source'],
                                reference, settings, source['chosen'], output)
    for ours, theirs in (('Source', 'Adaptive_RivalFine'), ('Source_NoThreshold', 'Adaptive_NoThreshold')):
        if not np.array_equal(predictions[ours], control[theirs]):
            raise RuntimeError('Original deployed-source equivalence failed: '+ours)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Existing sense experiment output; no overwrite.')
    loaded = prepare(args)
    geometry, vip, queries, profiles, source, _, _, _, identity = loaded
    states = frozen_state(geometry, vip)
    samples = discover_samples(args.dataset, args.data_root)
    if args.action == 'select':
        indices = sorted(random.Random(SEED).sample(range(len(samples)), min(CALIBRATION_IMAGES, len(samples))))
        samples = [samples[i] for i in indices]
        if args.max_images:
            samples = samples[:args.max_images]
    elif args.max_images:
        keys = set(random.Random(SEED+1).sample([s.key for s in samples], min(args.max_images, len(samples))))
        samples = [s for s in samples if s.key in keys]
    keys = [s.key for s in samples]
    chosen = samples[args.shard_index::args.num_shards]
    if not chosen or len(keys) != len(set(keys)) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Nonempty unique sample coverage required.')
    calibration = None
    if args.action == 'evaluate':
        calibration = json.loads(Path(args.selection).read_text())
        if (calibration['status'] != 'complete' or calibration['target_masks_loaded'] or calibration['smoke_only']
                or calibration['source_selection_sha256'] != sha(args.source_selection)
                or calibration['candidate_vocabulary_sha256'] != sha(args.candidate_vocabulary)):
            raise ValueError('Frozen full image-only sense calibration required.')
    output.mkdir(parents=True)
    started, ignored, tiles, actions = time.perf_counter(), 0, 0, 0.
    records, witness_records, matrices, arrays = [], [], {}, {}
    classes = len(identity['classes'])
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        active_queries = queries if number == 1 or args.action == 'evaluate' else {k: q for k, q in queries.items() if k != 'source'}
        predictions, diag, witness = predict(image, geometry, vip, active_queries, profiles, output,
                                             calibration, witnesses=args.action == 'select')
        if number == 1:
            verify_source(image, predictions, loaded, output)
        tiles += diag['tiles']
        actions += diag['mean_absolute_admission_potential']*diag['tiles']
        if args.action == 'select':
            errors = {k+'_error': witness_error(v, witness['labels'], witness['quality'])
                      for k, v in witness['probabilities'].items()}
            records.append(dict(key=sample.key, **errors, witness_tokens=int((witness['quality'] > 0).sum())))
            witness_records.append(witness)
            save(output/'progress.json', dict(status='running', processed_images=number, total_images=len(chosen),
                                              target_masks_loaded=False, record=records[-1]))
        else:
            target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < classes)
            ignored += int((~valid).sum())
            for method, pred in predictions.items():
                if pred.shape != target.shape or np.any(pred[valid] >= classes):
                    raise ValueError('Prediction coverage or class order changed.')
                cm = np.bincount(target[valid]*classes+pred[valid], minlength=classes**2).reshape(classes, classes)
                matrices.setdefault(method, np.zeros_like(cm))[:] += cm
                arrays.setdefault(method, []).append(cm)
            signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=list(matrices),
                classes={args.dataset: identity['classes']}, checkpoints=identity['checkpoints'],
                gear=dict(core_implementation=CORE, geometry=asdict(geometry.config), profiles=profiles,
                          calibration=calibration['profile'], background=calibration['background'],
                          source_selection_sha256=sha(args.source_selection), selection_sha256=sha(args.selection),
                          candidate_vocabulary_sha256=sha(args.candidate_vocabulary), main_model_changed=False),
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
            save(output/'results.json', result)
        print(json.dumps(dict(dataset=args.dataset, action=args.action, processed=number, total=len(chosen))), flush=True)
    frozen = check_frozen(states, geometry, vip)
    if args.action == 'select':
        profile = choose_profile(records)
        selected = profile['profile']
        background = foreground_thresholds([dict(probabilities=r['probabilities'][selected], labels=r['labels'],
                                                  quality=r['quality']) for r in witness_records],
            classes, identity['classes'][0] == 'background', source['chosen']['prob_thd'])
        result = dict(status='complete', implementation=IMPLEMENTATION, dataset=args.dataset,
            profile=profile, background=background, records=records, image_keys=keys, processed_images=len(keys),
            total_images=len(keys), smoke_only=bool(args.max_images), identity=identity,
            candidate_vocabulary_sha256=sha(args.candidate_vocabulary), source_selection_sha256=sha(args.source_selection),
            candidate_counts=[int((queries['default'].parents == c).sum()) for c in range(classes)],
            target_masks_loaded=False, target_label_tuning=False, transductive=True,
            source_equivalence_first_image_verified=True, profiles=profiles, **frozen,
            wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
        torch.save(witness_records, output/'witness_replay.pt')
        save(output/'selection.json', result)
    else:
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                            **{args.dataset+'__'+m: np.stack(a) for m, a in arrays.items()})
        result.update(status='complete', **frozen)
        save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'evaluate'))
    parser.add_argument('--dataset', choices=tuple(TOTALS), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    main(parser.parse_args())
