"""Full-image frozen local-contrast/coupling experiment and singleton timing."""
import argparse

import numpy as np
import torch

from dinotool.bounded_local_contrast import IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL, predict_image
from eval_bounded_contrast_coupling import benchmark, evaluate, inputs
from eval_bounded_physical_coupling import predict_image as original_predict
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def smoke(args, *, predictor=predict_image, implementation=IMPLEMENTATION,
          methods=METHODS, view_protocol=PROTOCOL):
    output, samples, load_image, _, geometry, banks, vip, queries, _, _ = inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    combined, diagnostics = predictor(image, geometry, banks, vip, queries)
    original, _ = original_predict(image, geometry, banks, vip, queries, None, methods=methods[:2])
    for p in banks:
        for m in methods[:2]:
            if not np.array_equal(combined[p][m], original[p][m]):
                raise RuntimeError('Original bounded endpoint changed: ' + p + '/' + m)
    for m in methods[2:]:
        single, _ = predictor(image, geometry, banks, vip, queries, methods=(m,))
        if any(not np.array_equal(single[p][m], combined[p][m]) for p in banks):
            raise RuntimeError('Singleton differs from simultaneous arms: ' + m)
    output.mkdir(parents=True)
    save(output / 'results.json', dict(status='complete', implementation=implementation,
        input_protocol=view_protocol, diagnostics=diagnostics, target_masks_loaded=False,
        sample_key=sample.key, original_endpoints_exact=True, singleton_primary_exact=True,
        **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
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
        evaluate(args, predictor=predict_image, implementation=IMPLEMENTATION, methods=METHODS,
            view_protocol=PROTOCOL, competitive=dict(alias_admission='none', fixed_aliases_per_class=20,
                reconstruction=PROTOCOL['reconstruction'], local_contrast='unfitted headwise variance compensation'))
    else:
        benchmark(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL,
                  names=(*METHODS[1:], 'VIP_All20'))
