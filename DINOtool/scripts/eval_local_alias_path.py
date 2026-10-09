"""Evaluate one frozen local lexical path with exact previous controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.local_alias_path import (IMPLEMENTATION, METHODS, OBSERVATION_MEAN,
    OLD_HARD, OLD_SOFT, PRIMARY, PROTOCOL, predict_image)
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as PREVIOUS_MEAN,
    ONE_SIDED_HARD as PREVIOUS_HARD, ONE_SIDED_SOFT as PREVIOUS_SOFT, predict_image as previous_predict)
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
    previous, _ = previous_predict(image, geometry, banks, vip, queries,
        methods=(*METHODS[:3], PREVIOUS_MEAN, PREVIOUS_SOFT, PREVIOUS_HARD), execution=execution)
    replay = (*[(m, m) for m in METHODS[:3]], (OBSERVATION_MEAN, PREVIOUS_MEAN),
              (OLD_SOFT, PREVIOUS_SOFT), (OLD_HARD, PREVIOUS_HARD))
    for p in banks:
        for new, old in replay:
            if not np.array_equal(actual[p][new], previous[p][old]):
                raise RuntimeError('Changed original/positive/hard/soft endpoint: '+p+'/'+new)
    single, _ = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,), execution=execution)
    if any(not np.array_equal(actual[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Singleton differs from combined controls.')
    for row in diagnostics.values():
        if (not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4
                or row['fine_coverage_max_error'] > 1e-6 or row['canonical_risk_max'] != 0
                or row['local_canonical_risk_max'] != 0 or row['local_invalid_write_max'] > 1e-10
                or row['positive_directed_delta_max'] > 1e-12 or row['class_budget_max_error'] > 1e-10
                or row['matched_previous_unmatchable'] != 0 or row['matched_previous_norm_relative_error'] > 1e-10
                or row['direct_match_unmatchable'] != 0 or row['direct_match_norm_relative_error'] > 1e-10):
            raise RuntimeError('Local lexical source/fidelity invariants failed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, original_endpoints_exact=True, same_information_mean_exact=True,
        strong_hard_exact=True, strong_soft_exact=True, singleton_primary_exact=True,
        target_masks_loaded=False, diagnostics=diagnostics, execution=execution.report(),
        **check_frozen(state, geometry, vip)))


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
        predictor = predictor_for(execution)
        if args.mode == 'full':
            evaluator.inputs = panel_inputs
            evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION, methods=METHODS,
                view_protocol=PROTOCOL, competitive=dict(alias_admission=PROTOCOL['alias_admission'],
                    fixed_aliases_per_class=20, target_labels_used_for_selection=False,
                    frozen_primary=PRIMARY, maximum_fine_encodings_per_image=16,
                    fitted_parameters=0, interpretation='developed local lexical fidelity pilot'))
        else:
            evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
                view_protocol=PROTOCOL, names=(METHODS[2], PRIMARY, OLD_SOFT, 'VIP_All20'))
        with open(args.output_dir+'/results.json') as stream:
            result = json.load(stream)
        result['execution'] = execution.report()
        save(Path(args.output_dir)/'results.json', result)
