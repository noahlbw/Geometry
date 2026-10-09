"""Same-source contextual aggregation; replay every previous vocabulary arm."""
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.native_alias_noise import signed_potential
from dinotool.survival_alias import (IMPLEMENTATION, CONFIG, METHODS, LEGACY, PRIMARY, SOURCE,
    SHUFFLES, OLD_SHUFFLES, SPATIAL, soft_survival_observation, signed_allocation_controls)
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_geometry_semantic_innovation import parse_args
from run_region_semantic_suite_a800 import TOOL


PREVIOUS_NOISE = TOOL/'results/native_alias_noise_screen_r2_20261003'


def extend(reference, *, values, stats, frozen, key, crops, count, coordinates, image_size,
           members, risks, valid, baseline, operator, local, broad):
    replay = {}
    for method in LEGACY:
        old = torch.from_numpy(reference[key+'__'+method+'__scores']).to(baseline.device)
        replay[method] = float((values[method]-old).abs().max())
        if not torch.equal(values[method], old):
            raise RuntimeError('Previous vocabulary numerical endpoint changed: '+method)
    gamma = risks[SOURCE]
    risk_error = float((gamma-torch.from_numpy(reference[key+'__risk']).to(gamma.device)).abs().max())
    if risk_error != 0:
        raise RuntimeError('Same-source risk replay differs.')
    zero = soft_survival_observation(crops, count, coordinates, image_size, members,
        torch.zeros_like(gamma), valid, CONFIG.beta, CONFIG.query_chunk)
    if bool((zero != 0).any()):
        raise RuntimeError('Zero-risk normalized writer is not identity.')
    binary = soft_survival_observation(crops, count, coordinates, image_size, members,
        (gamma > 0).to(gamma.dtype), valid, CONFIG.beta, CONFIG.query_chunk)
    binary_v = signed_potential(binary, valid)[0]
    binary_error = float((baseline+operator @ binary_v-values['HardDelete_Exact']).abs().max())
    if binary_error > 1e-10:
        raise RuntimeError('Binary writer fails same-risk hard endpoint.')
    stats['survival'] = {'legacy_score_replay_max_errors': replay, 'risk_replay_max_error': risk_error,
        'zero_risk_writer_max': float(zero.abs().max()), 'binary_hard_score_max_error': binary_error,
        'sources': {}, 'controls': {}}
    changed_risks = {PRIMARY: gamma, 'SurvivalText_Exact': risks['TextOnly_Exact'],
                    **{name: risks[old] for name, old in zip(SHUFFLES, OLD_SHUFFLES)}}
    primary_directed = None
    for method, risk in changed_risks.items():
        directed = soft_survival_observation(crops, count, coordinates, image_size, members,
            risk, valid, CONFIG.beta, CONFIG.query_chunk)
        potential, consistency = signed_potential(directed, valid)
        values[method] = baseline+operator @ potential
        stats['survival']['sources'][method] = {**consistency,
            'positive_directed_fraction': float((directed[valid] > 0).double().mean()),
            'mean_absolute_directed': float(directed[valid].abs().mean()),
            'mean_absolute_potential': float(potential[valid].abs().mean())}
        frozen[key+'__'+method+'__directed'] = directed.cpu().numpy()
        if method == PRIMARY:
            primary_directed = directed
            values['SurvivalMeanLogit'] = .5*(local.double()+broad.double()+potential)
    for method, array in signed_allocation_controls(primary_directed.cpu().numpy(), valid.cpu().numpy()).items():
        directed = torch.from_numpy(array).to(baseline.device)
        potential, consistency = signed_potential(directed, valid)
        values[method] = baseline+operator @ potential
        stats['survival']['controls'][method] = {**consistency,
            'pair_budget_max_error': float((directed.sum(0)-primary_directed.sum(0)).abs().max()),
            'pair_spectrum_max_error': float((directed[valid].sort(0).values-primary_directed[valid].sort(0).values).abs().max()),
            'pair_absolute_budget_max_error': float((directed.abs().sum(0)-primary_directed.abs().sum(0)).abs().max())}
        frozen[key+'__'+method+'__directed'] = array


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as reference:
        def extension(**kwargs):
            extend(reference, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
