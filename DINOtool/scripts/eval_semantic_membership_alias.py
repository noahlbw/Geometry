"""One frozen semantic source on the unchanged bounded complete-image reader."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_patch_only import predict_image as original_predict
from dinotool.semantic_membership_alias import (BASELINE, IMPLEMENTATION, METHODS,
    PRIMARY, PROTOCOL, SemanticSource, predict_image)
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def smoke(args, predictor):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    actual, diagnostics = predictor(image, geometry, banks, vip, queries)
    previous, _ = original_predict(image, geometry, banks, vip, queries, methods=METHODS[:3])
    for protocol in banks:
        for name in METHODS[:3]:
            if not np.array_equal(actual[protocol][name], previous[protocol][name]):
                raise RuntimeError('Frozen original endpoint changed: '+protocol+'/'+name)
    single, single_diagnostics = predictor(image, geometry, banks, vip, queries, methods=(PRIMARY,))
    if any(not np.array_equal(actual[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs from simultaneous controls.')
    for row in (*diagnostics.values(), *single_diagnostics.values()):
        if (row['fine_forwards'] or row['additional_visual_forwards'] or row['additional_semantic_heads']
                or row['per_image_nli_forwards'] or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4
                or row['canonical_weight_max_error'] or row['gauge_max_error'] > 1e-10):
            raise RuntimeError('Frozen source/bounded observation cap violated.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, original_endpoints_exact=True, singleton_primary_exact=True,
        target_masks_loaded=False, diagnostics=diagnostics, **check_frozen(state, geometry, vip)))


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
    source = SemanticSource(Path(args.output_dir).parents[2]/'semantic_membership.json')
    predictor = partial(predict_image, semantic=source)
    if args.mode == 'smoke':
        smoke(args, predictor)
    elif args.mode == 'full':
        evaluator.inputs = panel_inputs
        evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION,
            methods=METHODS, view_protocol=PROTOCOL,
            competitive=dict(alias_admission=PROTOCOL['alias_admission'], fixed_aliases_per_class=20,
                target_labels_used_for_selection=False, frozen_primary=PRIMARY,
                semantic_model=PROTOCOL['semantic_model'], semantic_revision=PROTOCOL['semantic_revision'],
                additional_visual_forwards=0, additional_semantic_heads=0, per_image_nli_forwards=0,
                fitted_parameters=0))
    else:
        evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
            view_protocol=PROTOCOL, names=(BASELINE, PRIMARY, 'VIP_All20'))
