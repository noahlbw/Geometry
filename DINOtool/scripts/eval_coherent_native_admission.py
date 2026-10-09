"""Actual unchanged observation paths, legacy replay, then coherent admission."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.coherent_native_admission import (IMPLEMENTATION, CONFIG, METHODS, LEGACY,
    PRIMARY, ALIAS_SHUFFLES, ACTION_SHUFFLES, source_controls, coherent_observation,
    hard_observation, action_controls)
from dinotool.native_class_ownership import family_class_field
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_native_ownership_reader import extend as ownership_extend, SOURCE, PREVIOUS_NOISE
from eval_geometry_semantic_innovation import parse_args
from run_native_ownership_feasibility import native_crops
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/native_class_ownership_prediction_r2_20261003'


def extend(prior, sources, metadata, cached, positions, text_features, historical, **kwargs):
    ownership_extend(prior, sources, metadata, cached, positions, text_features, **kwargs)
    values, stats, frozen = [kwargs[n] for n in ('values', 'stats', 'frozen')]
    key, members, valid = [kwargs[n] for n in ('key', 'members', 'valid')]
    operator, baseline, local, broad = [kwargs[n] for n in ('operator', 'baseline', 'local', 'broad')]
    device = baseline.device
    replay = {m: float((values[m]-torch.from_numpy(historical[key+'__'+m+'__scores']).to(device)).abs().max()) for m in LEGACY}
    if max(replay.values()) != 0:
        raise RuntimeError('Previous33 numerical prediction endpoints changed.')
    assignments = torch.empty(members.numel(), dtype=torch.long, device=device)
    for f, group in enumerate(metadata[key]['family_indices']):
        assignments[group] = f
    parents = torch.tensor(metadata[key]['parents'], device=device)
    canonical = torch.tensor(metadata[key]['canonical'], device=device)
    field, known = [torch.from_numpy(sources[key+'__'+n]).to(device) for n in ('reference', 'known')]
    native, native_count = native_crops(cached, key, positions, device)
    no_holdout, no_holdout_known = family_class_field(native, native_count, kwargs['coordinates'],
        members, torch.zeros(1, members.numel(), dtype=torch.bool, device=device), valid)
    risks = source_controls(kwargs['risks']['NativeAlias_Exact'], field, known, assignments,
        parents, canonical, valid, members, no_holdout, no_holdout_known)
    args = (kwargs['crops'], kwargs['count'], kwargs['coordinates'], kwargs['image_size'], members)
    zero, _, _ = coherent_observation(*args, torch.zeros_like(risks[PRIMARY]), valid)
    if bool((zero != 0).any()):
        raise RuntimeError('Zero-risk coherent writer is not identity.')
    diag = {'legacy33_score_replay_max_errors': replay, 'zero_risk_action_max': float(zero.abs().max()),
        'sources': {}, 'controls': {}, 'unknown_reference_fraction': float((~known[assignments]).double().mean())}
    for name, risk in risks.items():
        delta, cap, entry = coherent_observation(*args, risk, valid)
        values[name] = baseline+operator @ delta
        entry['canonical_mean_risk'] = float(risk[valid][:, canonical].mean()) if bool(valid.any()) else 0.
        entry['alias_spectrum_max_error'] = float((risk[:, members].sort(-1).values-
            risks[PRIMARY][:, members].sort(-1).values).abs().max()) if name in ALIAS_SHUFFLES else None
        diag['sources'][name] = entry
        if name == PRIMARY:
            primary, primary_cap = delta, cap
            values['CoherentMeanLogit'] = .5*(local.double()+broad.double()+delta)
            frozen[key+'__coherent_risk'] = risk.cpu().numpy()
            frozen[key+'__coherent_action'] = delta.cpu().numpy()
            frozen[key+'__coherent_prewrite'] = (broad.double()+delta).cpu().numpy()
            frozen[key+'__coherent_capacity'] = cap.cpu().numpy()
    hard, hard_stats = hard_observation(*args, risks[PRIMARY], valid)
    values['CoherentHardDelete_Exact'] = baseline+operator @ hard
    diag['hard'] = hard_stats
    for name, delta in action_controls(primary, valid).items():
        values[name] = baseline+operator @ delta
        diag['controls'][name] = {'class_budget_max_error': float((delta.sum(0)-primary.sum(0)).abs().max()),
            'class_spectrum_max_error': float((delta[valid].sort(0).values-primary[valid].sort(0).values).abs().max())
                if name in ACTION_SHUFFLES else None,
            'local_capacity_excess_max': float((-delta-primary_cap).clamp_min(0.).max()),
            'mean_suppression': float(-delta[valid].mean()) if bool(valid.any()) else 0.}
    stats['coherent'] = diag


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset = relative.parts[0]
    prior = json.loads((PREVIOUS_NOISE/dataset/'merged.json').read_text())
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    text_features = {}
    for key in metadata:
        with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as arrays:
            text_features[key] = arrays['mean_template_features']
    positions = prior['diagnostics'][prior['sample_keys'][int(relative.stem)]]['native_crop_positions']
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached, np.load(
            SOURCE/dataset/'fields'/(relative.stem+'.npz'), allow_pickle=False) as fields, np.load(
            PREVIOUS/relative, allow_pickle=False) as historical:
        def extension(**kwargs):
            extend(cached, fields, metadata, cached, positions, text_features, historical, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Completed preceding ownership predictions required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
