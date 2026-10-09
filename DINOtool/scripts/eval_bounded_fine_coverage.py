"""Frozen bounded fine-witness coverage diagnosis and full-image timing."""
import argparse

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_coverage import BASELINE, IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL, predict_image
from dinotool.bounded_patch_only import predict_image as original_predict
from dinotool.bounded_physical_coupling import resize_geometry
from dinotool.bounded_fine_coverage import FineCoverageReader
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, observe_fine, save, tile_coordinates


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = evaluator.inputs(args)
    states = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    combined, diagnostics = predict_image(image, geometry, banks, vip, queries)
    previous, _ = original_predict(image, geometry, banks, vip, queries, methods=METHODS[:3])
    if any(not np.array_equal(combined[p][m], previous[p][m]) for p in banks for m in METHODS[:3]):
        raise RuntimeError('Original bounded endpoints changed.')
    single, _ = predict_image(image, geometry, banks, vip, queries, methods=(PRIMARY,))
    if any(not np.array_equal(combined[p][PRIMARY], single[p][PRIMARY]) for p in banks):
        raise RuntimeError('Singleton primary differs from simultaneous controls.')
    resized = resize_geometry(image)
    tile = resized[:, :512, :512]
    coordinates = tile_coordinates(0, 0, geometry.device)
    valid = (coordinates[:, 0] < tile.shape[-2]) & (coordinates[:, 1] < tile.shape[-1])
    legacy, count, _ = observe_fine(tile, vip, queries, coordinates, valid)
    current, changed_count, _ = observe_fine(tile, vip, queries, coordinates, valid, feature_batch=FineCoverageReader(vip))
    exact = torch.equal(count, changed_count) and all(
        torch.equal(a.alias_logits, b.alias_logits) and torch.equal(a.salience, b.salience)
        for p in banks for a, b in zip(legacy[p], current[p]))
    if not exact:
        raise RuntimeError('Established fine-source score replay differs.')
    for row in diagnostics.values():
        if not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4:
            raise RuntimeError('Whole-image fine observation budget exceeded.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION, sample_key=sample.key,
        original_endpoints_exact=True, singleton_primary_exact=True, legacy_fine_source_exact=True,
        target_masks_loaded=False, diagnostics=diagnostics, **check_frozen(states, geometry, vip)))


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
            view_protocol=PROTOCOL, competitive=dict(alias_admission='unchanged legacy hard admission in bounded fields',
                fixed_aliases_per_class=20, target_labels_used_for_selection=False,
                maximum_fine_encodings_per_image=16, frozen_primary=PRIMARY,
                interpretation='coverage/source diagnosis, not a new final model'))
    else:
        evaluator.benchmark(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL,
                            names=(BASELINE, PRIMARY, 'VIP_All20'))
