"""Same sparse model, exact grouped class reductions and matched complete-image cost."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from benchmark_natural_sense_fast import measure, require_available_gpu
from benchmark_sparse_alias_device import predict_device
from benchmark_sparse_alias_natural_cached import FrozenTextCache, GroupedFrozenTextCache, cached_predictor
from dinotool.natural_evaluation import discover_samples, load_rgb
from eval_natural_sense_adaptation import prepare, sha
from eval_natural_text_adaptation import official_prediction
from eval_rival_fine_full import frozen_state, check_frozen, save


class CheckedGroupedCache(GroupedFrozenTextCache):
    def __init__(self):
        super().__init__()
        self.reference = FrozenTextCache()
        self.verify = True
        self.checked_calls = {'geometry': 0, 'wide': 0}

    def geometry(self, *args, **kwargs):
        result = super().geometry(*args, **kwargs)
        if self.verify:
            if not torch.equal(result, self.reference.geometry(*args, **kwargs)):
                raise RuntimeError('Real Geometry class scores are not bitwise equal.')
            self.checked_calls['geometry'] += 1
        return result

    def wide(self, *args, **kwargs):
        result = super().wide(*args, **kwargs)
        if self.verify:
            if not torch.equal(result, self.reference.wide(*args, **kwargs)):
                raise RuntimeError('Real wide class scores are not bitwise equal.')
            self.checked_calls['wide'] += 1
        return result


def require_equal(actual, expected):
    if (actual[1] != expected[1] or set(actual[0]) != set(expected[0])
            or any(not np.array_equal(actual[0][name], expected[0][name]) for name in expected[0])):
        raise RuntimeError('Grouped execution changed complete-image labels or diagnostics.')


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)
    if output.exists() or args.repetitions < 3:
        raise ValueError('New output and at least three warmed repetitions required.')
    geometry, vip, queries, profiles, _, _, _, settings, identity = prepare(args)
    selection = json.loads(Path(args.selection).read_text())
    if selection['status'] != 'complete' or selection['target_masks_loaded'] or selection['target_label_tuning']:
        raise ValueError('Unchanged completed image-only selection required.')
    key = selection['profile']['profile']
    query = queries[key]
    sample = discover_samples(args.dataset, args.data_root)[0]
    image = load_rgb(sample)
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    inputs = (image, geometry, vip, {key: query}, {key: profiles[key]}, output, selection)
    original, grouped = FrozenTextCache(), CheckedGroupedCache()
    expected = predict_device(*inputs, original)
    actual = predict_device(*inputs, grouped)
    require_equal(actual, expected)
    cpu = cached_predictor()(*inputs[:-1], calibrated=selection)
    require_equal(actual, cpu)
    grouped.verify = False
    previous = json.loads(Path(args.reference_result).read_text())
    if (previous['sample_key'] != sample.key or previous['checkpoint_identity'] != identity['checkpoints']
            or previous['vocabulary_sha256'] != sha(args.candidate_vocabulary)
            or previous['selection_sha256'] != sha(args.selection)):
        raise RuntimeError('Historical frozen benchmark input identity changed.')
    with np.load(Path(args.reference_result).with_name('predictions.npz'), allow_pickle=False) as old:
        if not np.array_equal(actual[0]['Sense_Calibrated'], old['SparseNativeSoft']):
            raise RuntimeError('Historical sparse complete prediction changed.')
        expected_vip = old['VIP_MatchedWords'].copy()
    calls = {
        'OriginalDevice': lambda: predict_device(*inputs, original),
        'GroupedDevice': lambda: predict_device(*inputs, grouped),
        'VIP_MatchedWords': lambda: official_prediction(image, vip, query, settings),
    }
    for name, call in calls.items():
        result = call()
        if name == 'VIP_MatchedWords':
            if not np.array_equal(result[0], expected_vip):
                raise RuntimeError('Matched finite VIP prediction changed.')
        else:
            require_equal(result, expected)
    timings = {name: {'seconds': [], 'peaks': []} for name in calls}
    order = list(calls)
    for repetition in range(args.repetitions):
        for name in order[repetition % len(order):] + order[:repetition % len(order)]:
            result, timing = measure(calls[name], (), geometry.device, 1)
            timings[name]['seconds'].extend(timing['seconds'])
            timings[name]['peaks'].append(timing['peak_allocated_mib'])
            if name == 'VIP_MatchedWords':
                if not np.array_equal(result[0], expected_vip):
                    raise RuntimeError('VIP labels changed across timed repetitions.')
            else:
                require_equal(result, expected)
    for name, timing in timings.items():
        timing['median_seconds'] = statistics.median(timing['seconds'])
        timing['peak_allocated_mib'] = max(timing.pop('peaks'))
        print(json.dumps({'method': name, 'complete_image_ms': timing['median_seconds'] * 1000}), flush=True)
    groups = grouped.reduction_members(query.parents, len(query.class_names))
    np.savez_compressed(output / 'predictions.npz', **actual[0], VIP_MatchedWords=expected_vip)
    save(output / 'results.json', dict(status='complete', implementation='exact-grouped-class-readout-v1-20261005',
        sample_key=sample.key, image_size=list(image.shape[-2:]), classes=len(query.class_names), aliases=len(query.aliases),
        timings=timings, grouped_score_calls_checked=grouped.checked_calls,
        reduction_groups=[dict(aliases_per_class=count, classes=len(ids)) for ids, _, count in groups],
        class_scores_bitwise_equal=True, complete_predictions_and_diagnostics_exact=True,
        cpu_reference_predictions_and_diagnostics_exact=True, historical_sparse_prediction_exact=True,
        historical_vip_prediction_exact=True, fixed_profile=profiles[key], reference_diagnostics=actual[1],
        vocabulary_sha256=sha(args.candidate_vocabulary), selection_sha256=sha(args.selection),
        checkpoint_identity=identity['checkpoints'], target_masks_loaded=False, semantic_rule_changed=False,
        scope='one complete frozen ADE image; warmed synchronized rotated-order repeats, shared-resident memory',
        **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ade150', choices=('ade150',))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=7)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection',
                 'output-dir', 'reference-result'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
