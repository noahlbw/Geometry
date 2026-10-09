"""Actual predictions with vocabulary-time frozen semantic role observations."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.semantic_role_admission import (IMPLEMENTATION, CONFIG, METHODS, LEGACY,
    PRIMARY, ALIAS_SHUFFLES, semantic_conflict, source_controls)
from dinotool.coherent_native_admission import coherent_observation
from eval_class_attachment_admission import extend as attachment_extend, PREVIOUS as COHERENT, OWNERSHIP
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_native_ownership_reader import SOURCE, PREVIOUS_NOISE
from eval_geometry_semantic_innovation import parse_args
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/class_attachment_admission_20261003'
SEMANTIC = TOOL/'results/semantic_role_source_20261003'


def extend(prior, sources, metadata, cached, positions, text_features, ownership,
           coherent, historical, semantic, **kwargs):
    attachment_extend(prior, sources, metadata, cached, positions, text_features, ownership, coherent, **kwargs)
    values, stats, frozen = [kwargs[n] for n in ('values', 'stats', 'frozen')]
    key, members, valid = [kwargs[n] for n in ('key', 'members', 'valid')]
    operator, baseline, local, broad = [kwargs[n] for n in ('operator', 'baseline', 'local', 'broad')]
    device = baseline.device
    replay = {m: float((values[m]-torch.from_numpy(historical[key+'__'+m+'__scores']).to(device)).abs().max()) for m in LEGACY}
    if max(replay.values()) != 0:
        raise RuntimeError('Previous61 numerical prediction endpoints changed.')
    assignments = torch.empty(members.numel(), dtype=torch.long, device=device)
    for f, group in enumerate(metadata[key]['family_indices']):
        assignments[group] = f
    parents = torch.tensor(metadata[key]['parents'], device=device)
    field, known = [torch.from_numpy(sources[key+'__'+n]).to(device) for n in ('reference', 'known')]
    entry = semantic['sources'][key]
    if (entry['aliases'] != metadata[key]['aliases'] or entry['parents'] != parents.tolist()
            or len(entry['classes']) != members.shape[0]):
        raise RuntimeError('Semantic source alias/class identity changed.')
    source_risk = semantic_conflict(torch.tensor(entry['role_logits']), torch.tensor(entry['parents']), len(entry['classes']))
    conflict = torch.tensor(entry['semantic_conflict'], dtype=torch.float64)
    if not torch.equal(source_risk, conflict):
        raise RuntimeError('Frozen semantic role score replay differs.')
    conflict = conflict.to(device)
    base = torch.from_numpy(frozen[key+'__class_base_risk']).to(device)
    risks, extra = source_controls(base, conflict, field, known, assignments, parents, valid, members)
    args = (kwargs['crops'], kwargs['count'], kwargs['coordinates'], kwargs['image_size'], members)
    base_action, _, _ = coherent_observation(*args, base, valid, CONFIG)
    diag = {'legacy61_score_replay_max_errors': replay, 'source_score_replay_exact': True,
        'extra_semantic_mean_risk': float(extra[valid].mean()),
        'semantic_competitor_alias_fraction': float((conflict.amax(-1) > 0).double().mean()),
        'sources': {}, 'no_per_image_language_forwards': True}
    for name, risk in risks.items():
        action, cap, details = coherent_observation(*args, risk, valid, CONFIG)
        values[name] = baseline+operator @ action
        details['incremental_suppression_over_class_base'] = float(-(action-base_action)[valid].mean())
        if name in ALIAS_SHUFFLES:
            resolved = 1-base > 1e-12
            recovered = (risk-base)/(1-base).clamp_min(1e-30)
            error = (recovered[:, members].sort(-1).values-extra[:, members].sort(-1).values).abs()
            selected = resolved[:, members].all(-1)
            details['extra_risk_spectrum_max_error'] = float(error[selected].max()) if bool(selected.any()) else 0.
        diag['sources'][name] = details
        if name == PRIMARY:
            values['SemanticRoleMeanLogit'] = .5*(local.double()+broad.double()+action)
            frozen[key+'__semantic_role_risk'] = risk.cpu().numpy()
            frozen[key+'__semantic_role_action'] = action.cpu().numpy()
            frozen[key+'__semantic_role_capacity'] = cap.cpu().numpy()
    stats['semantic_role'] = diag


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset = relative.parts[0]
    prior = json.loads((PREVIOUS_NOISE/dataset/'merged.json').read_text())
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    semantic = json.loads((SEMANTIC/(dataset+'.json')).read_text())
    if not semantic['sentinel_contract_passed'] or semantic['implementation'] != IMPLEMENTATION:
        raise RuntimeError('Semantic source contract must pass before target prediction.')
    text_features = {}
    for key in metadata:
        with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as arrays:
            text_features[key] = arrays['mean_template_features']
    positions = prior['diagnostics'][prior['sample_keys'][int(relative.stem)]]['native_crop_positions']
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached, np.load(
            SOURCE/dataset/'fields'/(relative.stem+'.npz'), allow_pickle=False) as fields, np.load(
            OWNERSHIP/relative, allow_pickle=False) as ownership, np.load(
            COHERENT/relative, allow_pickle=False) as coherent, np.load(PREVIOUS/relative, allow_pickle=False) as historical:
        def extension(**kwargs):
            extend(cached, fields, metadata, cached, positions, text_features, ownership,
                coherent, historical, semantic, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((SEMANTIC/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Completed mask-free semantic source required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
