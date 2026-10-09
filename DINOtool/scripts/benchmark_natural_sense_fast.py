"""Fixed-image timing; optional masks are loaded only after frozen inference."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import FunctionType

import numpy as np
import torch

from dinotool.natural_variable_alias_fast import retained_variable_scores_fast
from dinotool.natural_variable_alias_reader import retained_variable_scores
from dinotool.stratified_soft_alias import WideCrop


def measure(function, args, device, repetitions, kwargs=None):
    seconds, peak, result = [], [], None
    for _ in range(repetitions):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = function(*args, **(kwargs or {}))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak.append(torch.cuda.max_memory_allocated(device)/1048576)
        seconds.append(time.perf_counter()-start)
    return result, dict(seconds=seconds, median_seconds=statistics.median(seconds),
                        peak_allocated_mib=max(peak) if peak else None)


def require_available_gpu(device):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in tuple(str(i) for i in range(8)) or device.index not in (None, 0):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one physical GPU0-7 and use --device cuda.')
    lines = subprocess.check_output(['nvidia-smi', '-i', visible, '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    others = {int(line.strip()) for line in lines if line.strip()} - {os.getpid()}
    if others:
        raise RuntimeError('GPU occupied by other processes; no benchmark: '+str(sorted(others)))


def without_fine_reader(local, operator, broad, *args, **kwargs):
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    return {'RivalFineHard_Exact': baseline}, {'mean_absolute_admission_potential': 0.}


def without_fine_observer(image, vip, queries, coordinates, valid):
    return {key: [] for key in queries}, None, {'fine_forwards': 0}


@torch.inference_mode()
def predict_without_fine(*args, **kwargs):
    from eval_natural_sense_adaptation import predict
    from eval_natural_sense_fast import prepare_wide_observations

    original = predict.__wrapped__
    scope = dict(original.__globals__, prepare_wide=prepare_wide_observations,
                 observe_fine=without_fine_observer, retained_variable_scores=without_fine_reader)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


def image_scores(target, predictions, names):
    from dinotool.context_sense_evaluation import confusion

    classes = len(names)
    matrices, supports, values = {}, {}, {}
    for method, prediction in predictions.items():
        matrix, ignored = confusion(target, prediction, classes, reject=True)
        tp = matrix[np.arange(classes), np.arange(classes)]
        union = matrix.sum(1)+matrix.sum(0)[:classes]-tp
        matrices[method], supports[method] = matrix, union > 0
        values[method] = np.divide(tp, union, out=np.zeros(classes, dtype=float), where=union > 0)*100
    common = np.logical_or.reduce(list(supports.values()))
    present = next(iter(matrices.values())).sum(1) > 0
    if not common.any() or not present.any():
        raise ValueError('Image has no scored target pixels.')
    pixels = {method: int(matrix.sum()) for method, matrix in matrices.items()}
    if len(set(pixels.values())) != 1:
        raise ValueError('Different scored target pixels.')
    return dict(common_class_indices=np.flatnonzero(common).tolist(),
        common_class_names=[names[c] for c in np.flatnonzero(common)],
        gt_present_class_names=[names[c] for c in np.flatnonzero(present)], ignored_target_pixels=ignored,
        methods={method: dict(common_support_miou_percent=float(values[method][common].mean()),
            own_union_miou_percent=float(values[method][supports[method]].mean()),
            gt_present_miou_percent=float(values[method][present].mean()), scored_target_pixels=pixels[method],
            confusion_matrix=matrix.tolist(), per_class_iou_percent=values[method].tolist())
            for method, matrix in matrices.items()})


@torch.inference_mode()
def micro(args):
    device = torch.device(args.device)
    torch.manual_seed(20261004)
    if args.vocabulary:
        row = json.loads(Path(args.vocabulary).read_text())
        sizes = [len(g) for g in row['aliases_by_class']]
    else:
        sizes = [1+c % 5 for c in range(args.classes)]
    parents = torch.arange(len(sizes), device=device).repeat_interleave(torch.tensor(sizes, device=device))
    canonical = torch.tensor([sum(sizes[:c]) for c in range(len(sizes))], device=device)
    q, n, classes = len(parents), args.points, len(sizes)
    wide = [WideCrop(torch.randn(441, q, device=device), torch.randn(q, device=device), 0, 0, 336, 336)]
    fine = [WideCrop(torch.randn(1024, q, device=device), torch.randn(q, device=device),
                     0, 0, 256, 256, 32, 256)]
    coordinates = torch.rand(n, 2, device=device)*255
    valid = torch.ones(n, dtype=torch.bool, device=device)
    inputs = (torch.randn(n, classes, device=device), torch.eye(n, device=device)*.25,
              torch.randn(n, classes, device=device), wide, torch.ones(336, 336, device=device),
              fine, torch.ones(512, 512, device=device), coordinates, coordinates, valid,
              parents, canonical, (336, 336))
    options = dict(query_chunk=args.query_chunk)
    old = retained_variable_scores(*inputs, **options)
    fast = retained_variable_scores_fast(*inputs, **options)
    for key in old[0]:
        if not torch.equal(old[0][key], fast[0][key]):
            raise RuntimeError('Reader score equivalence failed: '+key)
    if old[1] != fast[1]:
        raise RuntimeError('Reader diagnostic equivalence failed.')
    _, reference = measure(retained_variable_scores, inputs, device, args.repetitions, options)
    _, cached = measure(retained_variable_scores_fast, inputs, device, args.repetitions, options)
    return dict(scope='synthetic reader only; no visual forward or dataset evaluation',
                device=str(device), classes=classes, aliases=q, points=n, query_chunk=args.query_chunk,
                scores_bitwise_equal=True, diagnostics_equal=True, reference=reference, cached=cached,
                speedup=reference['median_seconds']/cached['median_seconds'], target_masks_loaded=False)


@torch.inference_mode()
def full_image(args):
    from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
    from eval_natural_sense_adaptation import prepare, predict, sha
    from eval_natural_sense_fast import predict_fast
    from eval_natural_text_adaptation import official_prediction
    from eval_rival_fine_full import frozen_state, check_frozen

    loaded = prepare(args)
    geometry, vip, queries, profiles, _, _, official, settings, identity = loaded
    selection = json.loads(Path(args.selection).read_text())
    if selection['status'] != 'complete' or selection['target_masks_loaded'] or selection['target_label_tuning']:
        raise ValueError('Complete frozen image-only calibration required.')
    calibrated = selection if args.profile == 'chosen' else None
    key = selection['profile']['profile'] if calibrated else args.profile
    query, profile = queries[key], profiles[key]
    states = frozen_state(geometry, vip)
    sample = discover_samples(args.dataset, args.data_root)[args.sample_index]
    image = load_rgb(sample)
    print(json.dumps(dict(phase='models_loaded', sample_key=sample.key, image_size=list(image.shape[-2:]))), flush=True)
    inputs = (image, geometry, vip, {key: query}, {key: profile}, Path(args.output_dir).parent)
    options = dict(calibrated=calibrated)
    expected = predict(*inputs, **options)
    actual = predict_fast(*inputs, **options)
    mismatches = {name: int(np.count_nonzero(expected[0][name] != actual[0][name])) for name in expected[0]}
    if any(mismatches.values()) or expected[1] != actual[1]:
        raise RuntimeError('Full-image equivalence failed: '+str(mismatches))
    print(json.dumps(dict(phase='full_image_equivalence_passed', pixel_mismatches=mismatches)), flush=True)
    device = torch.device(args.device)
    timed_old, reference = measure(predict, inputs, device, args.repetitions, options)
    print(json.dumps(dict(phase='reference_timed', **reference)), flush=True)
    timed_fast, cached = measure(predict_fast, inputs, device, args.repetitions, options)
    print(json.dumps(dict(phase='cached_timed', **cached)), flush=True)
    name = 'Sense_Calibrated' if calibrated else {'default': 'Sense_Default', 'frozen': 'Sense_Words'}[key]
    predictions = {'Our_Reference': timed_old[0][name], 'Our_Cached': timed_fast[0][name]}
    extra_timings = {}
    if args.without_fine:
        predict_without_fine(*inputs, **options)
        unfine, extra_timings['Our_WithoutFine'] = measure(predict_without_fine, inputs, device, args.repetitions, options)
        predictions['Our_WithoutFine'] = unfine[0][name]
        print(json.dumps(dict(phase='without_fine_timed', **extra_timings['Our_WithoutFine'])), flush=True)
    official_prediction(image, vip, official, settings)
    vip_predictions, vip_timing = measure(official_prediction, (image, vip, official, settings), device, args.repetitions)
    predictions['VIP_Official_Finite'] = vip_predictions[0]
    print(json.dumps(dict(phase='vip_official_timed', **vip_timing)), flush=True)
    if args.vip_matched_words:
        official_prediction(image, vip, query, settings)
        matched, extra_timings['VIP_MatchedWords'] = measure(official_prediction, (image, vip, query, settings),
                                                          device, args.repetitions)
        predictions['VIP_MatchedWords'] = matched[0]
    if not np.array_equal(predictions['Our_Reference'], predictions['Our_Cached']):
        raise RuntimeError('Timed full-image predictions changed between backends.')
    result = dict(scope='one full image, one coupled profile; no paired multi-arm evaluation',
        dataset=args.dataset, sample_key=sample.key, image_size=list(image.shape[-2:]), profile=key,
        aliases=len(query.aliases), vip_aliases=len(official.aliases), checkpoint_identity=identity['checkpoints'],
        vocabulary_sha256=sha(args.candidate_vocabulary), geometry_config=asdict(geometry.config),
        source_selection_sha256=sha(args.source_selection), selection_sha256=sha(args.selection),
        source_profile=profile, vip_settings=asdict(settings),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']) if device.type == 'cuda' else None,
        pixel_mismatches=mismatches, diagnostics_equal=True, reference=reference, cached=cached,
        vip_official_finite=vip_timing, speedup=reference['median_seconds']/cached['median_seconds'],
        memory_scope='shared resident model/cache state, not standalone VIP memory',
        target_masks_loaded=args.score, target_masks_used_only_after_prediction=True, target_label_tuning=False,
        ablation='WithoutFine removes all fine forwards and their rival-admission correction; Geometry/wide/thresholds unchanged',
        extra_timings=extra_timings, **check_frozen(states, geometry, vip))
    if args.score:
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        result['image_metrics'] = image_scores(target, predictions, query.class_names)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--classes', type=int, default=150)
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--query-chunk', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--vocabulary')
    parser.add_argument('--dataset', default='ade150')
    parser.add_argument('--profile', choices=('default', 'frozen', 'chosen'), default='chosen')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output-dir')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--without-fine', action='store_true')
    parser.add_argument('--vip-matched-words', action='store_true')
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if min(args.threads, args.classes, args.points, args.query_chunk, args.repetitions) < 1:
        parser.error('Require positive benchmark dimensions/repetitions.')
    if not args.micro and (not args.output_dir or any(getattr(args, n.replace('-', '_')) is None
            for n in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                      'cache-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir', 'selection'))):
        parser.error('Full-image benchmark requires the frozen model/input paths and --output-dir.')
    if torch.device(args.device).type == 'cuda':
        require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)/'benchmark.json' if args.output_dir else None
    if output:
        if output.exists():
            raise ValueError('Existing benchmark result; no overwrite.')
        output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    result = micro(args) if args.micro else full_image(args)
    result['status'] = 'complete'
    if output:
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
