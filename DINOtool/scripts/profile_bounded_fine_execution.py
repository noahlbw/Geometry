"""Mask-free complete-image cost attribution and exact-backend comparison."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

import dinotool.bounded_fine_coverage as model
from dinotool.bounded_fine_execution import FineCoverageExecution, IMPLEMENTATION
from dinotool.bounded_physical_coupling import _BoundedBranch
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.model import checkpoint_manifest
import eval_bounded_contrast_coupling as evaluator
import eval_rival_fine_full as reference
from benchmark_natural_sense_fast import measure, require_available_gpu
from eval_rival_fine_full import check_frozen, frozen_state, save


ARMS = ('legacy', 'cached', 'cached_burst')


def exact_fields(expected, actual):
    if len(expected) != len(actual):
        raise RuntimeError('Changed number of alias actions.')
    for a, b in zip(expected, actual):
        if not all(torch.equal(x, y) for x, y in zip(a, b)):
            raise RuntimeError('Cached/graph risk or score is not bitwise exact.')


def capture_reader(reader, records):
    def call(*args, **kwargs):
        values, risk, stats = reader(*args, **kwargs)
        records.append((values['RivalFineHard_Exact'].clone(), risk.clone()))
        return values, risk, stats
    return call


@contextmanager
def capture(execution, records):
    if execution is None:
        with patch.object(model, 'retained_scores', capture_reader(model.retained_scores, records)):
            yield
    else:
        original = execution.score_reader
        execution.score_reader = capture_reader(original, records)
        try:
            yield
        finally:
            execution.score_reader = original


@contextmanager
def stage_profile(device, execution, stages):
    def timed(name, function):
        def call(*args, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize(device)
            stages[name] = stages.get(name, 0.)+time.perf_counter()-start
            return result
        return call
    original_branch = _BoundedBranch._call
    def branch(self, rgb):
        name = 'geometry_prepare' if self.method == 'prepare_image' else 'wide_visual_reader'
        return timed(name, original_branch)(self, rgb)
    score_reader = model.retained_scores if execution is None else execution.score_reader
    with patch.object(_BoundedBranch, '_call', branch), \
            patch.object(model.FineCoverageReader, '__call__', timed('fine_visual_reader', model.FineCoverageReader.__call__)), \
            patch.object(reference, 'observe_fine', timed('fine_acquisition_inclusive', reference.observe_fine)), \
            patch.object(model, 'reconstruction_operator', timed('geometry_solver', model.reconstruction_operator)), \
            patch.object(model, 'readout', timed('patch_only_head', model.readout)), \
            patch.object(DeviceProbabilityAccumulator, 'add', timed('stitch_add', DeviceProbabilityAccumulator.add)), \
            patch.object(DeviceProbabilityAccumulator, 'finalize_resized', timed('restore_argmax', DeviceProbabilityAccumulator.finalize_resized)):
        if execution is None:
            with patch.object(model, 'retained_scores', timed('alias_risk_and_writer', score_reader)):
                yield
        else:
            execution.score_reader = timed('alias_risk_and_writer', score_reader)
            try:
                yield
            finally:
                execution.score_reader = score_reader


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output, samples, load_image, _, geometry, banks, vip, queries, checkpoints, old = evaluator.inputs(args)
    states = frozen_state(geometry, vip)
    executions = {'legacy': None, 'cached': FineCoverageExecution(), 'cached_burst': FineCoverageExecution(burst=True)}
    rows = []
    started = time.perf_counter()
    for index in sorted({0, len(samples)//2, len(samples)-1}):
        sample = samples[index]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        def call(name):
            return model.predict_image(image, geometry, banks, vip, queries,
                                       methods=(model.PRIMARY,), execution=executions[name])
        expected_fields = []
        with capture(None, expected_fields):
            expected, diagnostics = call('legacy')
        for name in ARMS[1:]:
            fields = []
            with capture(executions[name], fields):
                actual, diag = call(name)
            exact_fields(expected_fields, fields)
            if diagnostics != diag or any(not np.array_equal(actual[p][model.PRIMARY], expected[p][model.PRIMARY]) for p in banks):
                raise RuntimeError('Execution backend changes outputs or diagnostics.')
            del fields
        del expected_fields
        timings = {name: dict(seconds=[], peaks=[]) for name in ARMS}
        for repeat in range(args.repetitions):
            for name in ARMS[repeat % len(ARMS):]+ARMS[:repeat % len(ARMS)]:
                (actual, diag), timing = measure(call, (name,), torch.device(args.device), 1)
                if diag != diagnostics or any(not np.array_equal(actual[p][model.PRIMARY], expected[p][model.PRIMARY]) for p in banks):
                    raise RuntimeError('Warmed execution changed outputs.')
                timings[name]['seconds'].extend(timing['seconds'])
                timings[name]['peaks'].append(timing['peak_allocated_mib'])
        profiles = {}
        for name in ARMS:
            stages = {}
            with stage_profile(torch.device(args.device), executions[name], stages):
                (actual, diag), timing = measure(call, (name,), torch.device(args.device), 1)
            if any(not np.array_equal(actual[p][model.PRIMARY], expected[p][model.PRIMARY]) for p in banks):
                raise RuntimeError('Profiling instrumentation changed predictions.')
            inclusive = stages.pop('fine_acquisition_inclusive')
            stages['fine_crop_and_text_assembly'] = inclusive-stages['fine_visual_reader']
            stages['unattributed'] = timing['median_seconds']-sum(stages.values())
            if min(stages.values()) < -1e-6:
                raise RuntimeError('Overlapping stage attribution.')
            profiles[name] = dict(synchronized_seconds=timing['median_seconds'], stages=stages)
        for values in timings.values():
            values.update(median_seconds=statistics.median(values['seconds']), peak_allocated_mib=max(values.pop('peaks')))
        rows.append(dict(sample_key=sample.key, image_size=list(image.shape[-2:]), timings=timings,
            stage_profiles=profiles, actual_diagnostics=diagnostics, risk_and_score_bitwise_equal=True))
        print(dict(dataset=args.dataset, sample=sample.key, milliseconds={n: timings[n]['median_seconds']*1000 for n in ARMS}), flush=True)
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION, dataset=args.dataset,
        images=rows, methods=ARMS, source_implementation=model.IMPLEMENTATION, source_protocol=model.PROTOCOL,
        vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        geometry_config=asdict(geometry.config), execution={n: None if e is None else e.report() for n, e in executions.items()},
        target_masks_loaded=False, bitwise_alias_risks_and_scores_verified=True,
        stage_scope='synchronized host-plus-device intervals; launch/wait gaps included, not pure kernel timing',
        warm_scope='uninstrumented complete-image timing; model/text/graph initialization excluded and setup reported separately',
        memory_scope='both backbones, frozen-head snapshots and captured graph pools resident; not standalone deployment memory',
        wall_seconds=time.perf_counter()-started, **check_frozen(states, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('benchmark',), default='benchmark')
    main(parser.parse_args())
