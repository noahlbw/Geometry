"""Shared-local witness pilot and paired complete-image timing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.bounded_patch_only import predict_image as original_predict
from dinotool.one_sided_alias_stream import PRIMARY as RETAINED, predict_image as retained_predict
from dinotool.shared_local_alias import IMPLEMENTATION, METHODS, PRIMARY, MEAN, PROTOCOL, predict_image
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    actual, diagnostics = predict_image(image, geometry, banks, vip, queries)
    original, _ = original_predict(image, geometry, banks, vip, queries, methods=METHODS[:3])
    single, _ = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,))
    for p in banks:
        for m in METHODS[:3]:
            if not np.array_equal(actual[p][m], original[p][m]):
                raise RuntimeError('Original bounded prediction changed: '+p+'/'+m)
        if not np.array_equal(actual[p][PRIMARY], single[p][PRIMARY]):
            raise RuntimeError('Singleton primary differs from joint controls.')
    for diag in diagnostics.values():
        if (diag['fine_forwards'] != 0 or diag['additional_visual_forwards'] != 0
                or not 0 < diag['extra_semantic_heads'] <= 4 or diag['canonical_risk_max'] != 0):
            raise RuntimeError('Shared observation budget or protection failed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, original_endpoints_exact=True, singleton_primary_exact=True,
        target_masks_loaded=False, diagnostics=diagnostics, **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    elif args.mode == 'full':
        evaluator.inputs = panel_inputs
        evaluator.evaluate(args, predictor=predict_image, implementation=IMPLEMENTATION,
            methods=METHODS, view_protocol=PROTOCOL,
            competitive=dict(alias_admission=PROTOCOL['alias_admission'], fixed_aliases_per_class=20,
                             target_labels_used_for_selection=False, fitted_parameters=0))
    else:
        execution = FineCoverageExecution(cached=True, burst=True)
        def predictor(*pos, **kwargs):
            if kwargs.get('methods') == (RETAINED,):
                return retained_predict(*pos, **kwargs, execution=execution)
            return predict_image(*pos, **kwargs)
        evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
            view_protocol=PROTOCOL, names=(METHODS[2], MEAN, PRIMARY, RETAINED, 'VIP_All20'))
        path = Path(args.output_dir)/'results.json'
        result = json.loads(path.read_text())
        result['retained_execution'] = execution.report()
        save(path, result)
