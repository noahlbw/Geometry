"""Same ADE image and frozen text/profile: remove fine crops and dense rivals."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
from dinotool.natural_sense_calibration import finalize_thresholds
from dinotool.natural_text_adaptation import class_logits
from dinotool.sparse_alias_reuse import (PRIMARY, IMPLEMENTATION, alias_layout,
                                        cache_sparse, profile_aliases, sparse_scores)
from dinotool.stratified_soft_alias import WideCrop
from benchmark_natural_sense_fast import measure, require_available_gpu, image_scores, predict_without_fine
from eval_matched_contribution_alias import crop_from_features
from eval_geometry_vip_reliability import sample_broad
from eval_natural_sense_adaptation import prepare, sha
from eval_natural_sense_fast import prepare_wide_observations, predict_fast
from eval_natural_text_adaptation import bank_from_query, official_prediction
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_stratified_soft_alias import tile_coordinates


@torch.inference_mode()
def predict_sparse(image, geometry, vip, queries, profiles, work, calibrated=None):
    height, width = image.shape[-2:]
    banks = {key: bank_from_query(query)[0] for key, query in queries.items()}
    layouts = {key: alias_layout(query.parents, banks[key].canonical_mask.nonzero().flatten(),
                                len(query.class_names)) for key, query in queries.items()}
    _, crops, _, count = prepare_wide_observations(image, vip, {'text': (banks, queries)})
    wide = {}
    for key, query in queries.items():
        dense_map = torch.zeros(len(query.class_names), *count.shape, device=geometry.device)
        profile = profiles[key]
        for crop in crops['text', key]:
            logits = class_logits(crop.alias_logits, crop.salience, query.parents,
                                  len(query.class_names), profile['tau'], profile['tem'])
            dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336),
                                  mode='bilinear', align_corners=False)[0]
            dense_map[:, crop.top:crop.top + crop.actual_height, crop.left:crop.left + crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
        wide[key] = dense_map / count
    blend, tiles, actions = hann_blend_window(512), 0, 0.
    with ExitStack() as stack:
        accumulators = {key: stack.enter_context(ProbabilityAccumulator(len(query.class_names), height, width,
                                                                      256, work)) for key, query in queries.items()}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coordinates = tile_coordinates(top, left, geometry.device)
                valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
                operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                ah, aw = min(512, height - top), min(512, width - left)
                for key, bank in banks.items():
                    query, profile, layout = queries[key], profiles[key], layouts[key]
                    alias = (prepared.geometry_projected.float() @ bank.features.T)[0]
                    local = alias_class_scores(alias, bank.parent_indices, bank.class_count) / .07
                    broad = sample_broad(wide[key], top, left, height, width).reshape_as(local)
                    observed = cache_sparse([replace(c, salience=c.salience / profile['tem'])
                                             for c in crops['text', key]], count, coordinates, (height, width), layout)
                    blank = WideCrop(local.new_empty(1024, 1), local.new_empty(1), 0, 0, 512, 512, 32, 512)
                    crop = crop_from_features(prepared.native_projected, query, blank)
                    witness = profile_aliases(crop.alias_logits, crop.salience / profile['tem'], layout)
                    score, diagnostic, _ = sparse_scores(local, operator, broad, observed, witness, valid,
                                                         layout, beta=profile['tau'])
                    dense = F.interpolate(score.T.reshape(1, -1, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    accumulators[key].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(),
                                          blend[:ah, :aw], left, top)
                    actions += diagnostic['mean_absolute_action'] / len(queries)
                tiles += 1
        predictions = {}
        display = {'source': 'Source', 'default': 'Sense_Default', 'frozen': 'Sense_Words'}
        for key, accumulator in accumulators.items():
            predictions[display[key]] = accumulator.finalize(profiles[key]['prob_thd'], background_index=0)[0]
        if calibrated is not None:
            key = calibrated['profile']['profile']
            predictions['Sense_Calibrated'] = finalize_thresholds(accumulators[key], calibrated['background']['thresholds'],
                                                                  queries[key].class_names[0] == 'background')
    return predictions, {'tiles': tiles, 'fine_forwards': 0, 'contenders_per_query': 2,
                          'mean_absolute_action': actions / tiles}, None


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing natural sparse benchmark, no overwrite.')
    loaded = prepare(args)
    geometry, vip, queries, profiles, _, _, official, settings, identity = loaded
    selection = json.loads(Path(args.selection).read_text())
    if selection['status'] != 'complete' or selection['target_masks_loaded'] or selection['target_label_tuning']:
        raise RuntimeError('Complete image-only frozen semantic profile required.')
    key = selection['profile']['profile']
    query = queries[key]
    sample = discover_samples(args.dataset, args.data_root)[0]
    image = load_rgb(sample)
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    inputs = (image, geometry, vip, {key: query}, {key: profiles[key]}, output)
    name = 'Sense_Calibrated'
    def sparse():
        return predict_sparse(*inputs, calibrated=selection)
    def cached():
        return predict_fast(*inputs, calibrated=selection)
    def base():
        return predict_without_fine(*inputs, calibrated=selection)
    def vip_matched():
        return official_prediction(image, vip, query, settings)
    calls = {PRIMARY: sparse, 'CachedSlow': cached, 'NoFineNoAdmission': base, 'VIP_MatchedWords': vip_matched}
    predictions, timings, diagnostics = {}, {}, {}
    for method, call in calls.items():
        call()
        result, timing = measure(call, (), torch.device(args.device), args.repetitions)
        predictions[method] = result[0] if method == 'VIP_MatchedWords' else result[0][name]
        timings[method] = timing
        diagnostics[method] = result[1] if method != 'VIP_MatchedWords' else {}
        print(json.dumps({'phase': method, 'median_ms': timing['median_seconds'] * 1000}), flush=True)
    np.savez_compressed(output / 'predictions.npz', **predictions)
    result = {'status': 'predictions_complete', 'implementation': IMPLEMENTATION, 'primary': PRIMARY,
        'scope': 'same fixed complete ADE image; frozen existing vocabulary/profile/thresholds, no new fitting',
        'sample_key': sample.key, 'image_size': list(image.shape[-2:]), 'dataset': args.dataset,
        'timings': timings, 'diagnostics': diagnostics, 'aliases': len(query.aliases),
        'profile': key, 'frozen_profile': profiles[key], 'checkpoint_identity': identity['checkpoints'],
        'vocabulary_sha256': sha(args.candidate_vocabulary), 'selection_sha256': sha(args.selection),
        'predictions_persisted_before_masks': True, 'target_label_tuning': False,
        'memory_scope': 'shared-resident models/caches, not standalone memory', **check_frozen(state, geometry, vip)}
    save(output / 'results.json', result)
    target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
    result['image_metrics'] = image_scores(target, predictions, query.class_names)
    result['status'] = 'complete'
    save(output / 'results.json', result)
    print(json.dumps({'status': 'complete', 'sample_key': sample.key}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ade150', choices=('ade150',))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=5)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
