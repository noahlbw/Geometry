"""Frozen bounded Geometry-supported covariance alias pilot."""
import argparse

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_covariance_alias import BASELINE, CAPACITY, IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL, predict_image
from dinotool.bounded_patch_only import predict_image as original_predict
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    combined, diagnostics = predict_image(image, geometry, banks, vip, queries)
    previous, _ = original_predict(image, geometry, banks, vip, queries, methods=METHODS[:3])
    for p in banks:
        for method in METHODS[:3]:
            if not np.array_equal(combined[p][method], previous[p][method]):
                raise RuntimeError('Frozen original endpoint differs: '+p+'/'+method)
    single, single_diagnostics = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,))
    if any(not np.array_equal(combined[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs from simultaneous controls.')
    for row in (*diagnostics.values(), *single_diagnostics.values()):
        if row['additional_visual_forwards'] or row['additional_semantic_heads'] or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4 or row['maximum_support_neighbours'] != CAPACITY:
            raise RuntimeError('Actual bounded zero-extra-forward budget violated.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION, sample_key=sample.key,
        original_endpoints_exact=True, singleton_primary_exact=True, target_masks_loaded=False,
        diagnostics=diagnostics, **check_frozen(state, geometry, vip)))


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
    elif args.mode == 'full':
        evaluator.inputs = panel_inputs
        evaluator.evaluate(args, predictor=predict_image, implementation=IMPLEMENTATION, methods=METHODS,
            view_protocol=PROTOCOL, competitive=dict(alias_admission=PROTOCOL['alias_admission'],
                fixed_aliases_per_class=20, additional_visual_forwards=0,
                target_labels_used_for_selection=False, frozen_primary=PRIMARY))
    else:
        evaluator.benchmark(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL,
                            names=(BASELINE, PRIMARY, 'VIP_All20'))
