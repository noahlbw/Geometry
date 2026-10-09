"""Matched full-image local-head replacements with frozen contextual evidence."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.coupled_readout_swap import (IMPLEMENTATION, READOUTS, METHODS, REFERENCE_METHODS,
                                          FIXED_PROTOCOL, local_features, replace_local)
from dinotool.fine_alias_view import CONFIG
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_count import canonical_indices
from dinotool.rival_fine_full import retained_scores, tile_reference_coordinates
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save
from eval_rival_fine_full import predict_tile as historical_predict_tile
from eval_stratified_soft_alias import tile_coordinates
from eval_vip_official_eight import PINNED_COMMIT


@torch.inference_mode()
def predict_tile(image, top, left, geometry, banks, vip, queries, wide, crops, wide_count):
    h, w = image.shape[-2:]
    prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
    features, head_stats = local_features(geometry, prepared)
    coordinates = tile_coordinates(top, left, geometry.device)
    valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
    fine_coordinates = tile_reference_coordinates(coordinates, top, left)
    fine, fine_count, costs = observe_fine(image[:, top:top+512, left:left+512], vip, queries,
                                          fine_coordinates, valid)
    operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    output, diagnostics, raw_scores = {}, {}, {}
    for p, bank in banks.items():
        text = F.normalize(bank.features.float(), dim=-1)
        raw = {name: alias_class_scores((feature.float() @ text.T)[0], bank.parent_indices, bank.class_count)
               for name, feature in features.items()}
        local = raw['Geometry'] / .07
        broad = sample_broad(wide['clean', p], top, left, h, w).reshape_as(local)
        query = queries[p]
        members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = canonical_indices(query.class_names, query.aliases, query.parents)
        scores, _, stats = retained_scores(local, operator, broad, crops['clean', p], wide_count,
            fine[p], fine_count, coordinates, fine_coordinates, valid, members, canonical, query.parents, (h, w))
        replay_unscreened, replay_screened = replace_local(local, local, operator,
            scores['NoAdmission_Exact'], scores['RivalFineHard_Exact'])
        if not (torch.equal(replay_unscreened, scores['NoAdmission_Exact'])
                and torch.equal(replay_screened, scores['RivalFineHard_Exact'])):
            raise RuntimeError('Geometry replacement identity failed.')
        for name in READOUTS[1:]:
            unscreened, screened = replace_local(raw[name]/.07, local, operator,
                scores['NoAdmission_Exact'], scores['RivalFineHard_Exact'])
            scores.update({name: (raw[name]/.07).double(), name+'_NoAdmission': unscreened,
                           name+'_RivalFineHard': screened})
        output[p], raw_scores[p] = scores, raw
        diagnostics[p] = {**stats, **costs, **head_stats, 'operator_residual': residual,
                          'shared_admission_identity_max_error': 0.}
    return output, diagnostics, raw_scores


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work):
    h, w = image.shape[-2:]
    wide, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    totals = {p: {'tiles': 0} for p in banks}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for p, bank in banks.items() for m in METHODS}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                scores, diagnostics, raw = predict_tile(image, top, left, geometry, banks, vip, queries,
                                                       wide, crops, count)
                ah, aw = min(512, h-top), min(512, w-left)
                for p, bank in banks.items():
                    for name, value in scores[p].items():
                        source = raw[p][name] if name in READOUTS else value
                        dense = F.interpolate(source.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                              mode='bilinear', align_corners=False)[0]
                        if name in READOUTS:
                            dense = dense/.07
                        accumulators[p, name].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(),
                                                 blend[:ah, :aw], left, top)
                    totals[p]['tiles'] += 1
                    for field, value in diagnostics[p].items():
                        totals[p][field] = totals[p].get(field, 0.) + value
        predictions = {p: {m: accumulators[p, m].finalize(None)[0] for m in METHODS} for p in banks}
    return predictions, {p: {f: v if f == 'tiles' else v/row['tiles'] for f, v in row.items()}
                         for p, row in totals.items()}


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing smoke output.')
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    geometry, banks, vip, queries, _ = make_models(args, specs)
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    wide, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    h, w = image.shape[-2:]
    positions = list(dict.fromkeys([(0, 0), (tile_starts(h, 512, 128)[-1], tile_starts(w, 512, 128)[-1])]))
    errors = {}
    for top, left in positions:
        actual, _, _ = predict_tile(image, top, left, geometry, banks, vip, queries, wide, crops, count)
        expected, _, _ = historical_predict_tile(image, top, left, geometry, banks, vip, queries, wide, crops, count)
        for p in banks:
            for m in REFERENCE_METHODS:
                error = float((actual[p][m]-expected[p][m]).abs().max())
                errors[f'{top},{left}__{p}__{m}'] = error
                if error != 0.:
                    raise RuntimeError('Historical numerical replay differs: '+str(errors))
    predictions, diagnostics = predict_image(image, geometry, banks, vip, queries, output)
    if any(tuple(pred.shape) != tuple(image.shape[-2:]) for ps in predictions.values() for pred in ps.values()):
        raise RuntimeError('Full-image dimensions differ.')
    save(output/'results.json', {'status': 'complete', 'implementation': IMPLEMENTATION,
        'dataset': args.dataset, 'sample_keys': [sample.key], 'processed_images': 1, 'total_images': 1,
        'image_size': [h, w], 'replayed_tile_offsets': positions, 'historical_score_errors': errors,
        'target_masks_loaded': False, 'full_image_diagnostics': diagnostics, **check_frozen(states, geometry, vip)})


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError('Existing output or invalid shard.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    keys, selected = [s.key for s in samples], samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise RuntimeError('Empty shard or duplicate samples.')
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    signature = {'implementation': IMPLEMENTATION, 'dataset': args.dataset, 'methods': METHODS,
        'classes': {p: bank.class_names for p, bank in banks.items()},
        'gear': {'geometry': asdict(geometry.config), 'observation': asdict(SETTINGS), 'admission': asdict(CONFIG),
                 'upstream_commit': PINNED_COMMIT, 'local_tiles': [512, 128], 'fine_physical_pixels_per_token': 8,
                 'local_head_swap': FIXED_PROTOCOL}, 'competitive': None,
        'vocabulary': {'sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
            'aliases': {p: bank.alias_names for p, bank in banks.items()},
            'counts': {p: [20]*bank.class_count for p, bank in banks.items()}},
        'checkpoints': checkpoint_manifest(checkpoints), 'global_sample_count': len(keys),
        'global_sample_keys_sha256': digest(keys), 'sample_keys': [s.key for s in selected],
        'sample_keys_sha256': digest([s.key for s in selected]), 'num_shards': args.num_shards,
        'shard_index': args.shard_index, 'config': vars(args)}
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    transitions = {p: {m: np.zeros((bank.class_count,)*3, np.int64) for m in METHODS[1:]}
                   for p, bank in banks.items()}
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    ignored, totals = dict.fromkeys(banks, 0), {p: {'tiles': 0} for p in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostics = predict_image(image, geometry, banks, vip, queries, output)
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for m in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count + predictions[p][m][valid]
                cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                matrices[p][m] += cm
                per_image[p+'__'+m].append(cm)
                if m != 'Geometry':
                    transitions[p][m] += transition_counts(predictions[p]['Geometry'], predictions[p][m], target,
                                                           bank.class_count)
            tiles = diagnostics[p]['tiles']
            totals[p]['tiles'] += tiles
            for field, value in diagnostics[p].items():
                if field != 'tiles':
                    totals[p][field] = totals[p].get(field, 0.) + value*tiles
        result = {'status': 'running', 'processed_images': number, 'total_images': len(selected), 'signature': signature,
            'metrics': {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in ms.items()}
                        for p, ms in matrices.items()},
            'transitions': {p: {m: {'counts': cm.tolist(), **transition_summary(cm)} for m, cm in ms.items()}
                            for p, ms in transitions.items()},
            'diagnostics': {p: {f: v if f == 'tiles' else v/row['tiles'] for f, v in row.items()}
                            for p, row in totals.items()},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576,
            'target_masks_used_only_after_prediction': True, 'shared_context_admission_reconstruction': True}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': number, 'total': len(selected),
            'miou': {p: {m: methods[m]['mean_iou_percent'] for m in METHODS}
                     for p, methods in result['metrics'].items()}}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{key: np.stack(value) for key, value in per_image.items()})
    result.update(status='complete', **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
    (smoke if is_smoke else main)(parse_args())
