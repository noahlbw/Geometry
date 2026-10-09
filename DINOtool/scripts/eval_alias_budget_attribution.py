"""Audit lexical-dose dependence of the unchanged one-sided soft predictor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool import one_sided_alias_audit as previous
from dinotool.alias_budget_attribution import (IMPLEMENTATION, METHODS, NEW_NAMES,
    OBSERVATION_MEAN, POOLED, PRIMARY, PROTOCOL, predict_image)
from dinotool.bounded_fine_execution import FineCoverageExecution
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


def predictor_for(execution):
    def call(*args, **kwargs):
        return predict_image(*args, **kwargs, execution=execution)
    return call


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    execution = FineCoverageExecution(cached=True, burst=True)
    actual, diagnostics = predict_image(image, geometry, banks, vip, queries, execution=execution)
    old, _ = previous.predict_image(image, geometry, banks, vip, queries, execution=execution)
    for protocol in banks:
        for name in METHODS:
            original = NEW_NAMES.get(name, name)
            if original in previous.METHODS and not np.array_equal(actual[protocol][name], old[protocol][original]):
                raise RuntimeError('Changed unchanged-rule endpoint: '+protocol+'/'+name)
    for name in (PRIMARY, POOLED):
        single, _ = predict_image(image, geometry, banks, vip, queries, methods=(name,), execution=execution)
        if any(not np.array_equal(actual[p][name], single[p][name]) for p in banks):
            raise RuntimeError('Singleton differs from combined controls: '+name)
    for row in diagnostics.values():
        if (not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4
                or row['wide_encodings'] > 4 or row['fine_coverage_max_error'] > 1e-6
                or row['canonical_risk_max'] != 0 or row['positive_directed_delta_max'] > 1e-12
                or row['matched_unmatchable'] != 0 or row['matched_norm_relative_error'] > 1e-10
                or row['class_budget_max_error'] > 1e-10 or row['rival_budget_max_error'] > 1e-10):
            raise RuntimeError('Unchanged source/control invariant failed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, original_endpoints_exact=True, same_information_mean_exact=True,
        strong_hard_exact=True, strong_soft_exact=True, primary_exact=True,
        singleton_primary_exact=True, singleton_pooled_exact=True, target_masks_loaded=False,
        diagnostics=diagnostics, execution=execution.report(), **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root',
                 'vocabulary-config', 'upstream-root', 'output-dir'):
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
        predictor = predictor_for(execution)
        if args.mode == 'full':
            evaluator.inputs = panel_inputs
            evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION,
                methods=METHODS, view_protocol=PROTOCOL,
                competitive=dict(alias_admission=PROTOCOL['alias_admission'], fixed_aliases_per_class=20,
                    target_labels_used_for_selection=False, frozen_primary=PRIMARY,
                    maximum_fine_encodings_per_image=16, fitted_parameters=0,
                    interpretation='unchanged rule and deployable pooled-class control; developed96'))
        else:
            evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
                view_protocol=PROTOCOL, names=(METHODS[2], OBSERVATION_MEAN, POOLED, PRIMARY, 'VIP_All20'))
        with open(args.output_dir+'/results.json') as stream:
            result = json.load(stream)
        result['execution'] = execution.report()
        save(Path(args.output_dir)/'results.json', result)
