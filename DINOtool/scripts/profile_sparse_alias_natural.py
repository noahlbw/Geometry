"""Profile the frozen sparse ADE predictor, with exact prediction replay."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_sparse_alias_natural_cached import FrozenTextCache, cached_predictor
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.natural_evaluation import discover_samples, load_rgb
from dinotool.natural_sense_calibration import finalize_thresholds
from dinotool.sparse_alias_reuse import cache_sparse, profile_aliases, sparse_scores
from dinotool.stratified_soft_alias import WideCrop
from benchmark_natural_sense_fast import measure, require_available_gpu
from eval_geometry_vip_reliability import sample_broad
from eval_matched_contribution_alias import crop_from_features
from eval_natural_sense_adaptation import prepare, sha
from eval_natural_sense_fast import prepare_wide_observations
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_stratified_soft_alias import tile_coordinates


class Stages:
    def __init__(self, device):
        self.device, self.seconds, self.counts = device, {}, {}

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - started
            self.counts[name] = self.counts.get(name, 0) + 1


@torch.inference_mode()
def profiled_predict(image, geometry, vip, queries, profiles, work, calibrated, cache, timer):
    height, width = image.shape[-2:]
    with timer.stage('fixed_text_metadata'):
        banks = {key: cache.bank(query)[0] for key, query in queries.items()}
        layouts = {key: cache.layout(query.parents, banks[key].canonical_mask.nonzero().flatten(),
                                    len(query.class_names)) for key, query in queries.items()}
    with timer.stage('wide_observation'):
        _, crops, _, count = prepare_wide_observations(image, vip, {'text': (banks, queries)})
    with timer.stage('wide_classification_upsampling_stitch'):
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
    blend, tiles, actions = hann_blend_window(512), 0, 0.
    with ExitStack() as stack:
        with timer.stage('accumulator_setup'):
            accumulators = {key: stack.enter_context(ProbabilityAccumulator(len(query.class_names), height, width,
                                                                           256, work)) for key, query in queries.items()}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                with timer.stage('local_crop_to_device'):
                    rgb = _crop_at(image, top, left, 512).to(geometry.device)
                with timer.stage('geometry_forward'):
                    prepared = geometry.prepare_image(rgb)
                with timer.stage('geometry_reconstruction_operator'):
                    coordinates = tile_coordinates(top, left, geometry.device)
                    valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
                    operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                ah, aw = min(512, height - top), min(512, width - left)
                for key, bank in banks.items():
                    query, profile, layout = queries[key], profiles[key], layouts[key]
                    with timer.stage('geometry_classification'):
                        alias = (prepared.geometry_projected.float() @ bank.features.T)[0]
                        local = cache.geometry(alias, bank.parent_indices, bank.class_count) / .07
                    with timer.stage('sample_broad'):
                        broad = sample_broad(wide[key], top, left, height, width).reshape_as(local)
                    with timer.stage('wide_alias_cache_stencils'):
                        observed = cache_sparse([replace(c, salience=c.salience / profile['tem'])
                                                 for c in crops['text', key]], count, coordinates, (height, width), layout)
                    with timer.stage('reused_local_query_classification'):
                        blank = WideCrop(local.new_empty(1024, 1), local.new_empty(1), 0, 0, 512, 512, 32, 512)
                        crop = crop_from_features(prepared.native_projected, query, blank)
                        witness = profile_aliases(crop.alias_logits, crop.salience / profile['tem'], layout)
                    with timer.stage('sparse_alias_reader_and_reconstruction'):
                        score, diagnostic, _ = sparse_scores(local, operator, broad, observed, witness, valid,
                                                             layout, beta=profile['tau'])
                    with timer.stage('dense_upsampling_softmax_device_to_host'):
                        dense = F.interpolate(score.T.reshape(1, -1, 32, 32), (512, 512),
                                              mode='bilinear', align_corners=False)[0]
                        probability = dense[:, :ah, :aw].softmax(0).float().cpu().numpy()
                    with timer.stage('cpu_probability_accumulation'):
                        accumulators[key].add(probability, blend[:ah, :aw], left, top)
                    actions += diagnostic['mean_absolute_action'] / len(queries)
                tiles += 1
        predictions = {}
        display = {'source': 'Source', 'default': 'Sense_Default', 'frozen': 'Sense_Words'}
        with timer.stage('default_finalization'):
            for key, accumulator in accumulators.items():
                predictions[display[key]] = accumulator.finalize(profiles[key]['prob_thd'], background_index=0)[0]
        with timer.stage('calibrated_finalization'):
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
        raise RuntimeError('Preserve existing sparse profiling output.')
    geometry, vip, queries, profiles, _, _, _, _, identity = prepare(args)
    selection = json.loads(Path(args.selection).read_text())
    if selection['status'] != 'complete' or selection['target_masks_loaded'] or selection['target_label_tuning']:
        raise RuntimeError('The frozen image-only profile is required.')
    key = selection['profile']['profile']
    query = queries[key]
    sample = discover_samples(args.dataset, args.data_root)[0]
    image = load_rgb(sample)
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    inputs = (image, geometry, vip, {key: query}, {key: profiles[key]}, output)
    fast, cache = cached_predictor(), FrozenTextCache()
    expected = fast(*inputs, calibrated=selection)
    timer = Stages(geometry.device)
    profiled_predict(*inputs, selection, cache, timer)
    actual, timing = measure(fast, inputs, geometry.device, args.repetitions, {'calibrated': selection})
    stages, counts, profiled_seconds = [], None, []
    for _ in range(args.repetitions):
        timer = Stages(geometry.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = profiled_predict(*inputs, selection, cache, timer)
        torch.cuda.synchronize()
        profiled_seconds.append(time.perf_counter() - started)
        if (value[1] != expected[1] or set(value[0]) != set(expected[0])
                or any(not np.array_equal(value[0][m], expected[0][m]) for m in expected[0])):
            raise RuntimeError('Instrumentation changed complete-image predictions or diagnostics.')
        stages.append(timer.seconds)
        counts = timer.counts
    old = json.loads(Path(args.reference_result).read_text())
    if (sample.key != old['sample_key'] or list(image.shape[-2:]) != old['image_size']
            or sha(args.candidate_vocabulary) != old['vocabulary_sha256']
            or sha(args.selection) != old['selection_sha256'] or identity['checkpoints'] != old['checkpoint_identity']):
        raise RuntimeError('The fixed profiling image or semantic inputs changed.')
    with np.load(Path(args.reference_result).with_name('predictions.npz'), allow_pickle=False) as prior:
        if not np.array_equal(actual[0]['Sense_Calibrated'], prior['SparseNativeSoft']):
            raise RuntimeError('Historical sparse complete-image predictions changed.')
    medians = {name: statistics.median(row[name] for row in stages) for name in stages[0]}
    result = dict(status='complete', implementation='frozen-sparse-complete-image-profile-v1-20261005',
        sample_key=sample.key, image_size=list(image.shape[-2:]), aliases=len(query.aliases),
        profile=key, frozen_profile=profiles[key], vocabulary_sha256=sha(args.candidate_vocabulary),
        selection_sha256=sha(args.selection), checkpoint_identity=identity['checkpoints'],
        uninstrumented_timing=timing, synchronized_stage_seconds=stages, stage_median_seconds=medians,
        stage_call_counts=counts, instrumented_total_seconds=profiled_seconds,
        instrumented_unaccounted_seconds=[total - sum(row.values()) for total, row in zip(profiled_seconds, stages)],
        profiling_synchronization_changes_schedule=True, stage_medians_not_a_throughput_measure=True,
        exact_reference_predictions_and_diagnostics=True, historical_sparse_prediction_exact=True,
        target_masks_loaded=False, reference_diagnostics=expected[1], **check_frozen(state, geometry, vip))
    np.savez_compressed(output / 'predictions.npz', **actual[0])
    save(output / 'results.json', result)
    print(json.dumps({'status': 'complete', 'uninstrumented_ms': timing['median_seconds'] * 1000,
                      'stage_ms': {k: v * 1000 for k, v in medians.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ade150', choices=('ade150',))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=3)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection',
                 'output-dir', 'reference-result'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
