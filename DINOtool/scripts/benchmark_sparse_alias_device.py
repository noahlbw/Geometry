"""Frozen sparse model: keep probability stitching and final reductions on device."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_sparse_alias_natural_cached import FrozenTextCache, cached_predictor
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.gear_ov import _crop_at
from dinotool.inference import hann_blend_window, tile_starts
from dinotool.natural_evaluation import discover_samples, load_rgb
from dinotool.sparse_alias_reuse import cache_sparse, profile_aliases, sparse_scores
from dinotool.stratified_soft_alias import WideCrop
from benchmark_natural_sense_fast import measure, require_available_gpu
from eval_geometry_vip_reliability import sample_broad
from eval_matched_contribution_alias import crop_from_features
from eval_natural_sense_adaptation import prepare, sha
from eval_natural_sense_fast import prepare_wide_observations
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_stratified_soft_alias import tile_coordinates


@torch.inference_mode()
def predict_device(image, geometry, vip, queries, profiles, work, calibrated, cache):
    height, width = image.shape[-2:]
    required = sum(len(q.class_names) + 1 for q in queries.values()) * height * width * 4
    if required > 512 * 1024 * 1024:
        raise ValueError('Device accumulator pilot requires at most512MiB of probability storage.')
    banks = {key: cache.bank(query)[0] for key, query in queries.items()}
    layouts = {key: cache.layout(query.parents, banks[key].canonical_mask.nonzero().flatten(),
                                len(query.class_names)) for key, query in queries.items()}
    _, crops, _, count = prepare_wide_observations(image, vip, {'text': (banks, queries)})
    wide = {}
    for key, query in queries.items():
        dense_map = torch.zeros(len(query.class_names), *count.shape, device=geometry.device)
        profile = profiles[key]
        for crop in crops['text', key]:
            logits = cache.wide(crop.alias_logits, crop.salience, query.parents,
                                len(query.class_names), profile['tau'], profile['tem'])
            dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336),
                                  mode='bilinear', align_corners=False)[0]
            dense_map[:, crop.top:crop.top + crop.actual_height, crop.left:crop.left + crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
        wide[key] = dense_map / count
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    tiles, actions = 0, 0.
    with ExitStack() as stack:
        accumulators = {key: stack.enter_context(DeviceProbabilityAccumulator(len(query.class_names), height, width,
                                                                              geometry.device)) for key, query in queries.items()}
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
                    local = cache.geometry(alias, bank.parent_indices, bank.class_count) / .07
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
                    accumulators[key].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                    actions += diagnostic['mean_absolute_action'] / len(queries)
                tiles += 1
        predictions = {}
        display = {'source': 'Source', 'default': 'Sense_Default', 'frozen': 'Sense_Words'}
        selected = calibrated['profile']['profile']
        for key, accumulator in accumulators.items():
            if key != selected:
                raise ValueError('The matched device pilot uses the single frozen selected profile.')
            default, adjusted = accumulator.finalize_outputs(profiles[key]['prob_thd'], calibrated['background']['thresholds'],
                                                             queries[key].class_names[0] == 'background')
            predictions[display[key]], predictions['Sense_Calibrated'] = default, adjusted
    return predictions, {'tiles': tiles, 'fine_forwards': 0, 'contenders_per_query': 2,
                          'mean_absolute_action': actions / tiles}, None


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing device-accumulator output.')
    geometry, vip, queries, profiles, _, _, _, _, identity = prepare(args)
    selection = json.loads(Path(args.selection).read_text())
    if selection['status'] != 'complete' or selection['target_masks_loaded'] or selection['target_label_tuning']:
        raise RuntimeError('Frozen image-only semantic inputs required.')
    key = selection['profile']['profile']
    query = queries[key]
    sample = discover_samples(args.dataset, args.data_root)[0]
    image = load_rgb(sample)
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    inputs = (image, geometry, vip, {key: query}, {key: profiles[key]}, output)
    cache, reference = FrozenTextCache(), cached_predictor()
    def call():
        return predict_device(*inputs, selection, cache)
    expected, actual = reference(*inputs, calibrated=selection), call()
    if (expected[1] != actual[1] or set(expected[0]) != set(actual[0])
            or any(not np.array_equal(expected[0][m], actual[0][m]) for m in expected[0])):
        raise RuntimeError('Device stitching changed complete-image predictions or diagnostics.')
    timings = {}
    for name, function in (('DeviceAccumulator', call), ('CPUAccumulator', lambda: reference(*inputs, calibrated=selection))):
        _, timings[name] = measure(function, (), geometry.device, args.repetitions)
        print(json.dumps({'method': name, 'median_ms': timings[name]['median_seconds'] * 1000}), flush=True)
    with np.load(Path(args.reference_result).with_name('predictions.npz'), allow_pickle=False) as old:
        if not np.array_equal(actual[0]['Sense_Calibrated'], old['SparseNativeSoft']):
            raise RuntimeError('Historical sparse calibrated predictions changed.')
    previous = json.loads(Path(args.reference_result).read_text())
    if (previous['sample_key'] != sample.key or previous['checkpoint_identity'] != identity['checkpoints']
            or previous['selection_sha256'] != sha(args.selection) or previous['vocabulary_sha256'] != sha(args.candidate_vocabulary)):
        raise RuntimeError('Frozen device benchmark inputs changed.')
    np.savez_compressed(output / 'predictions.npz', **actual[0])
    result = dict(status='complete', implementation='sparse-device-probability-stitching-v1-20261005',
        sample_key=sample.key, image_size=list(image.shape[-2:]), aliases=len(query.aliases), profile=key,
        timings=timings, complete_predictions_and_diagnostics_exact=True, historical_sparse_prediction_exact=True,
        fixed_profile=profiles[key], vocabulary_sha256=sha(args.candidate_vocabulary), selection_sha256=sha(args.selection),
        checkpoint_identity=identity['checkpoints'], target_masks_loaded=False, reference_diagnostics=actual[1],
        probability_storage_bytes=(len(query.class_names) + 1) * image.shape[-2] * image.shape[-1] * 4,
        scope='single complete image, both backbones resident; storage cap512MiB; no semantic-rule changes',
        **check_frozen(state, geometry, vip))
    save(output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ade150', choices=('ade150',))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=5)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection',
                 'output-dir', 'reference-result'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
