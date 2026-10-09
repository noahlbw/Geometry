"""Frozen independent semantic witness with identical Geometry/wide endpoints."""
import argparse
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_siglip_alias import (
    IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL, QUERY_BUDGET,
    cached_probe_pool, observer_for, predict_image,
)
from dinotool.bounded_patch_only import PRIMARY as BASELINE, predict_image as original_predict
from dinotool.region_semantic_readout import restricted_pool
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    observer = observer_for(geometry, banks)['observer']
    head_state = {name: parameter.detach().clone() for name, parameter in observer.model.vision_model.head.named_parameters()}
    with patch.object(observer, 'visual', wraps=observer.visual) as calls:
        combined, diagnostics = predict_image(image, geometry, banks, vip, queries)
        if calls.call_count != 1:
            raise RuntimeError('Combined inference did not use exactly one observer encoding.')
    previous, _ = original_predict(image, geometry, banks, vip, queries, methods=METHODS[:3])
    for p in banks:
        for method in METHODS[:3]:
            if not np.array_equal(combined[p][method], previous[p][method]):
                raise RuntimeError('Frozen original endpoint differs: '+p+'/'+method)
    with patch.object(observer, 'visual', wraps=observer.visual) as calls:
        single, single_diagnostics = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,))
        if calls.call_count != 1:
            raise RuntimeError('Singleton did not use exactly one observer encoding.')
    if any(not np.array_equal(combined[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs from simultaneous controls.')
    hidden, _ = observer.visual(image[None].to(geometry.device))
    support = torch.ones(3, 256, device=geometry.device)
    support[1, 128:] = 0
    support[2] = torch.linspace(.01, 1., 256, device=geometry.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = cached_probe_pool(observer.model.vision_model.head, hidden, support)
        expected = restricted_pool(observer.model.vision_model.head, hidden.expand(3, -1, -1), support)
    raw_pool_error = float((actual.float()-expected.float()).abs().max())
    pool_error = float((F.normalize(actual.float(), dim=-1)-F.normalize(expected.float(), dim=-1)).abs().max())
    if pool_error > 3e-3:
        raise RuntimeError('Cached native MAP replay differs: '+str(pool_error))
    if any(p.requires_grad for p in observer.model.parameters()):
        raise RuntimeError('Independent source is not frozen.')
    if any(not torch.equal(parameter, head_state[name]) for name, parameter in observer.model.vision_model.head.named_parameters()):
        raise RuntimeError('Independent semantic head weights changed.')
    for row in (*diagnostics.values(), *single_diagnostics.values()):
        if (row['additional_visual_forwards'] != 1 or row['geometry_encodings'] > 4
                or row['wide_encodings'] > 4 or row['fine_forwards']
                or not 0 < row['witness_queries'] <= QUERY_BUDGET):
            raise RuntimeError('Actual whole-image observation/query cap violated.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION, sample_key=sample.key,
        original_endpoints_exact=True, singleton_primary_exact=True, target_masks_loaded=False,
        observer_weights_frozen=True, observer_head_weights_unchanged=True,
        actual_observer_encodings_per_call=1, native_pool_descriptor_max_error=pool_error,
        native_pool_raw_max_error=raw_pool_error,
        observer_source=observer.manifest, diagnostics=diagnostics, **check_frozen(state, geometry, vip)))


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
                fixed_aliases_per_class=20, additional_visual_forwards=1,
                semantic_source=PROTOCOL['semantic_source'], target_labels_used_for_selection=False,
                frozen_primary=PRIMARY))
    else:
        evaluator.benchmark(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL,
                            names=(BASELINE, PRIMARY, 'VIP_All20'))
