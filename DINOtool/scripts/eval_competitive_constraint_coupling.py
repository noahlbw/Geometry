"""Frozen coupled spatial/class pilot; original and strongest-source replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.competitive_constraint_coupling import (BASELINE, IMPLEMENTATION, METHODS,
    OBSERVATION_MEAN, PRIMARY, PROTOCOL, UNIFORM, predict_image)
from dinotool.supported_positive_alias import (NO_GEOMETRY_RISK as OLD_UNIFORM,
    OBSERVATION_MEAN as OLD_MEAN, predict_image as original_predict)
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


def accelerated_predictor(execution):
    def call(*args, **kwargs):
        return predict_image(*args, **kwargs, execution=execution)
    return call


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    states = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    execution = FineCoverageExecution(cached=True, burst=True)
    combined, diagnostics = predict_image(image, geometry, banks, vip, queries, execution=execution)
    previous, _ = original_predict(image, geometry, banks, vip, queries,
        methods=(*METHODS[:3], OLD_MEAN, OLD_UNIFORM), execution=execution)
    for p in banks:
        for current, old in (*[(m, m) for m in METHODS[:3]], (OBSERVATION_MEAN, OLD_MEAN), (UNIFORM, OLD_UNIFORM)):
            if not np.array_equal(combined[p][current], previous[p][old]):
                raise RuntimeError('Original/positive/strong-uniform endpoint changed: '+p+'/'+current)
    single, _ = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,), execution=execution)
    if any(not np.array_equal(combined[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs from simultaneous controls.')
    for row in diagnostics.values():
        if (not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4
                or row['canonical_risk_max'] != 0 or row['fine_coverage_max_error'] > 1e-6
                or row['cg_relative_residual'] > 1e-7 or row['class_mean_preservation_error'] > 1e-8
                or row['constraint_trace_error'] > 1e-10 or row['norm_matching_error'] > 1e-7):
            raise RuntimeError('Source/graph/solver/canonical cap failed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, original_endpoints_exact=True, same_information_mean_exact=True,
        strong_uniform_exact=True, singleton_primary_exact=True, target_masks_loaded=False,
        diagnostics=diagnostics, execution=execution.report(), **check_frozen(states, geometry, vip)))


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
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    else:
        execution = FineCoverageExecution(cached=True, burst=True)
        predictor = accelerated_predictor(execution)
        if args.mode == 'full':
            evaluator.inputs = panel_inputs
            evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION, methods=METHODS,
                view_protocol=PROTOCOL, competitive=dict(alias_admission=PROTOCOL['alias_admission'],
                    fixed_aliases_per_class=20, target_labels_used_for_selection=False,
                    frozen_primary=PRIMARY, maximum_fine_encodings_per_image=16,
                    coupled_constraints=True, fitted_parameters=0, interpretation='developed joint-constraint pilot'))
        else:
            evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
                view_protocol=PROTOCOL, names=(BASELINE, PRIMARY, UNIFORM, 'VIP_All20'))
        with open(args.output_dir+'/results.json') as stream:
            result = json.load(stream)
        result['execution'] = execution.report()
        save(Path(args.output_dir)/'results.json', result)
