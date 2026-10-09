"""Actual fixed-window predictions with semantically admissible native references."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.semantic_reference_admission import (IMPLEMENTATION, CONFIG, METHODS, LEGACY,
    PRIMARY, source_controls, weighted_family_field)
from dinotool.coherent_native_admission import coherent_observation
from dinotool.excess_alias_rejection import semantic_contradictions
from eval_semantic_role_admission import (extend as semantic_extend, PREVIOUS as ATTACHMENT,
    SEMANTIC, COHERENT, OWNERSHIP)
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_native_ownership_reader import SOURCE, PREVIOUS_NOISE
from eval_geometry_semantic_innovation import parse_args
from run_native_ownership_feasibility import native_crops
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/semantic_role_admission_20261003'


def extend(prior, sources, metadata, cached, positions, text_features, ownership,
           coherent, attachment, historical, semantic, **kwargs):
    semantic_extend(prior, sources, metadata, cached, positions, text_features,
        ownership, coherent, attachment, semantic, **kwargs)
    values, stats, frozen = [kwargs[n] for n in ('values', 'stats', 'frozen')]
    key, members, valid = [kwargs[n] for n in ('key', 'members', 'valid')]
    baseline, operator, local, broad = [kwargs[n] for n in ('baseline', 'operator', 'local', 'broad')]
    device = baseline.device
    replay = {m: float((values[m]-torch.from_numpy(historical[key+'__'+m+'__scores']).to(device)).abs().max()) for m in LEGACY}
    if max(replay.values()) != 0:
        raise RuntimeError('Previous69 numerical endpoints changed.')
    assignments = torch.empty(members.numel(), dtype=torch.long, device=device)
    excluded = torch.zeros(len(metadata[key]['family_indices']), members.numel(), dtype=torch.bool, device=device)
    for f, group in enumerate(metadata[key]['family_indices']):
        assignments[group], excluded[f, group] = f, True
    parents, canonical = [torch.tensor(metadata[key][n], device=device) for n in ('parents', 'canonical')]
    conflict = torch.tensor(semantic['sources'][key]['semantic_conflict'], dtype=torch.float64, device=device)
    cosine = semantic_contradictions(torch.from_numpy(text_features[key]).to(device), parents, canonical)
    native, native_count = native_crops(cached, key, positions, device)
    arguments = (native, native_count, kwargs['coordinates'], members, excluded, valid)
    neutral, neutral_known = weighted_family_field(*arguments, torch.ones(len(parents), device=device))
    neutral_error = float((neutral-torch.from_numpy(sources[key+'__reference']).to(device)).abs().max())
    if neutral_error > 1e-12 or not torch.equal(neutral_known, torch.from_numpy(sources[key+'__known']).to(device)):
        raise RuntimeError('Original unweighted held-family reference differs.')
    risks, fields = source_controls(*arguments, assignments, parents, canonical, conflict,
        cosine, kwargs['risks']['NativeAlias_Exact'])
    base = torch.from_numpy(frozen[key+'__class_base_risk']).to(device)
    base_error = float((fields['old_base']-base).abs().max())
    if base_error != 0:
        raise RuntimeError('Original class/view base changed.')
    identity_error = float((risks[PRIMARY]-base).abs().max()) if bool((conflict == 0).all()) else None
    if identity_error is not None and identity_error != 0:
        raise RuntimeError('Zero-semantic complete-model identity differs.')
    # Replacing excluded values must not change their own held reference.
    selected = excluded[0].nonzero().flatten()
    altered = [replace(c, alias_logits=c.alias_logits.clone(), salience=c.salience.clone()) for c in native]
    for crop in altered:
        crop.alias_logits[:, selected], crop.salience[selected] = 10000, 10000
    held_check = weighted_family_field(altered, native_count, kwargs['coordinates'], members,
        excluded[:1], valid, fields['admissible'])[0]
    independence_error = float((held_check[:, 0]-fields['held_reference'][:, 0]).abs().max())
    if independence_error != 0:
        raise RuntimeError('Semantically weighted held-out family confirms itself.')
    args = (kwargs['crops'], kwargs['count'], kwargs['coordinates'], kwargs['image_size'], members)
    base_action, _, _ = coherent_observation(*args, base, valid, CONFIG)
    diag = {'legacy69_score_replay_max_errors': replay,
        'neutral_held_reference_replay_max_error': neutral_error, 'old_base_replay_max_error': base_error,
        'zero_semantic_identity_max_error': identity_error, 'held_family_independence_max_error': independence_error,
        'semantic_weight_spectrum_errors': fields['semantic_spectrum_errors'],
        'mean_reference_admissibility': float(fields['admissible'].mean()),
        'weighted_class_base_mean_risk': float(fields['base'][valid].mean()),
        'weighted_attachment_mean_risk': float(fields['extra'][valid].mean()),
        'extra_canonical_risk_max': float(fields['extra'][:, canonical].abs().max()),
        'unknown_full_reference_fraction': float((~fields['full_known']).double().mean()),
        'unknown_held_reference_fraction': float((~fields['held_known']).double().mean()), 'sources': {}}
    for name, risk in risks.items():
        action, cap, details = coherent_observation(*args, risk, valid, CONFIG)
        values[name] = baseline+operator @ action
        details['incremental_suppression_over_old_class_base'] = float(-(action-base_action)[valid].mean())
        diag['sources'][name] = details
        if name == PRIMARY:
            values['SemanticReferenceMeanLogit'] = .5*(local.double()+broad.double()+action)
            frozen[key+'__semantic_reference_risk'] = risk.cpu().numpy()
            frozen[key+'__semantic_reference_action'] = action.cpu().numpy()
            frozen[key+'__semantic_reference_capacity'] = cap.cpu().numpy()
    frozen[key+'__weighted_full_reference'] = fields['full_reference'].cpu().numpy()
    frozen[key+'__weighted_full_known'] = fields['full_known'].cpu().numpy()
    frozen[key+'__weighted_held_reference'] = fields['held_reference'].cpu().numpy()
    frozen[key+'__weighted_held_known'] = fields['held_known'].cpu().numpy()
    stats['semantic_reference'] = diag


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset = relative.parts[0]
    prior = json.loads((PREVIOUS_NOISE/dataset/'merged.json').read_text())
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    semantic = json.loads((SEMANTIC/(dataset+'.json')).read_text())
    if not semantic['sentinel_contract_passed']:
        raise RuntimeError('Frozen semantic source must retain its contract.')
    text_features = {}
    for key in metadata:
        with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as arrays:
            text_features[key] = arrays['mean_template_features']
    positions = prior['diagnostics'][prior['sample_keys'][int(relative.stem)]]['native_crop_positions']
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached, np.load(
            SOURCE/dataset/'fields'/(relative.stem+'.npz'), allow_pickle=False) as fields, np.load(
            OWNERSHIP/relative, allow_pickle=False) as ownership, np.load(
            COHERENT/relative, allow_pickle=False) as coherent, np.load(ATTACHMENT/relative, allow_pickle=False) as attachment, np.load(
            PREVIOUS/relative, allow_pickle=False) as historical:
        def extension(**kwargs):
            extend(cached, fields, metadata, cached, positions, text_features, ownership,
                coherent, attachment, historical, semantic, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Completed previous69 prediction suite required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
