"""Actual predictions from frozen ownership fields; all prior arms replay."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.alias_action_capacity import suppression_caps
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.native_alias_noise import hard_pair_observation, signed_potential
from dinotool.native_class_ownership import family_class_field, ownership_risk
from dinotool.native_ownership_reader import (IMPLEMENTATION, CONFIG, METHODS, LEGACY, PRIMARY,
    ALIAS_SHUFFLES, source_controls)
from dinotool.pair_context_reader import pair_observation
from dinotool.rival_preserving_alias import competitive_potential, directional_controls
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_geometry_semantic_innovation import parse_args
from run_native_ownership_feasibility import native_crops
from run_region_semantic_suite_a800 import TOOL


PREVIOUS_NOISE = TOOL/'results/native_alias_noise_screen_r2_20261003'
SOURCE = TOOL/'results/native_class_ownership_feasibility_r2_20261003'


def extend(prior, sources, metadata, cached, positions, text_features, *, values, stats,
           frozen, key, crops, count, coordinates, image_size, members, risks, valid,
           baseline, operator, local, broad):
    device = baseline.device
    replay = {m: float((values[m]-torch.from_numpy(prior[key+'__'+m+'__scores']).to(device)).abs().max()) for m in LEGACY}
    if any(error != 0 for error in replay.values()):
        raise RuntimeError('Previous17 numerical endpoints changed.')
    families = metadata[key]['family_indices']
    parents = torch.tensor(metadata[key]['parents'], device=device)
    canonical = torch.tensor(metadata[key]['canonical'], device=device)
    assignments = torch.empty(len(parents), dtype=torch.long, device=device)
    for f, group in enumerate(families):
        assignments[group] = f
    features = torch.from_numpy(text_features[key]).to(device)
    conflict = semantic_contradictions(features, parents, canonical)
    field, known, attachment = [torch.from_numpy(sources[key+'__'+name]).to(device)
        for name in ('reference', 'known', 'attachment_risk')]
    replay_attachment = ownership_risk(field, known, assignments, parents, canonical, conflict, valid)
    attachment_error = float((replay_attachment.float()-attachment).abs().max())
    if attachment_error != 0:
        raise RuntimeError('Frozen attachment source replay differs.')
    gamma = risks['NativeAlias_Exact']
    if not torch.equal(gamma, torch.from_numpy(prior[key+'__risk']).to(device)):
        raise RuntimeError('Previous view source differs.')
    native, native_count = native_crops(cached, key, positions, device)
    no_holdout, no_holdout_known = family_class_field(native, native_count, coordinates, members,
        torch.zeros(1, len(parents), dtype=torch.bool, device=device), valid)
    changed = source_controls(gamma, replay_attachment, field, known, assignments, parents, canonical,
        conflict, valid, members, no_holdout, no_holdout_known)
    if not torch.equal(changed[PRIMARY], torch.from_numpy(sources[key+'__joint_risk']).to(device)):
        raise RuntimeError('Persisted joint source differs.')
    cap = suppression_caps(crops, count, coordinates, image_size, members,
        members == canonical[:, None], valid, CONFIG.beta, CONFIG.query_chunk)[1]
    zero = pair_observation(crops, count, coordinates, image_size, members,
        torch.zeros_like(gamma), valid, CONFIG.beta, CONFIG.query_chunk)[2]
    if bool((zero != 0).any()):
        raise RuntimeError('Unchanged zero-risk writer is not identity.')
    stats['ownership'] = {'legacy_score_replay_max_errors': replay,
        'attachment_risk_replay_max_error': attachment_error, 'zero_risk_writer_max': float(zero.abs().max()),
        'sources': {}, 'controls': {}}
    primary_directed = None
    for method, risk in changed.items():
        directed = pair_observation(crops, count, coordinates, image_size, members,
            risk, valid, CONFIG.beta, CONFIG.query_chunk)[2]
        potential, _, consistency = competitive_potential(directed, valid)
        excess = float((-directed-cap[..., None]).clamp_min(0).max())
        if excess > 1e-6 or float(risk[:, canonical].abs().max()) != 0:
            raise RuntimeError('Unchanged protected writer bounds differ.')
        values[method] = baseline+operator @ potential
        stats['ownership']['sources'][method] = {**consistency,
            'capacity_excess_max': excess, 'canonical_risk_max': float(risk[:, canonical].abs().max()),
            'mean_risk': float(risk[valid].double().mean()),
            'alias_spectrum_max_error': float((risk[:, members].sort(2).values-changed[PRIMARY][:, members].sort(2).values).abs().max())
                if method in ALIAS_SHUFFLES else None}
        if method == PRIMARY:
            primary_directed = directed
            values['OwnershipMeanLogit'] = .5*(local.double()+broad.double()+potential)
            frozen[key+'__ownership_joint_risk'] = risk.cpu().numpy()
            frozen[key+'__ownership_attachment_risk'] = attachment.cpu().numpy()
            frozen[key+'__ownership_directed'] = directed.cpu().numpy()
    hard = hard_pair_observation(crops, count, coordinates, image_size, members,
        changed[PRIMARY], valid, CONFIG.beta, CONFIG.query_chunk)
    potential, consistency = signed_potential(hard, valid)
    values['OwnershipHardDelete_Exact'] = baseline+operator @ potential
    stats['ownership']['hard'] = consistency
    for old, array in directional_controls(primary_directed.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
        method = 'Ownership'+old
        directed = torch.from_numpy(array).to(device)
        potential, _, consistency = competitive_potential(directed, valid)
        values[method] = baseline+operator @ potential
        stats['ownership']['controls'][method] = {**consistency,
            'pair_budget_max_error': float((directed.sum(0)-primary_directed.double().sum(0)).abs().max()),
            'capacity_excess_max': float((-directed-cap.double()[..., None]).clamp_min(0).max()),
            'pair_spectrum_max_error': float((directed[valid].sort(0).values-primary_directed.double()[valid].sort(0).values).abs().max())}


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset, index = relative.parts[0], relative.stem
    row = json.loads((SOURCE/dataset/'results.json').read_text())
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    text_features = {}
    for key in metadata:
        with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as arrays:
            text_features[key] = arrays['mean_template_features']
    prior = json.loads((PREVIOUS_NOISE/dataset/'merged.json').read_text())
    if row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature'] or row['target_masks_loaded']:
        raise RuntimeError('Frozen mask-free source contract changed.')
    positions = prior['diagnostics'][prior['sample_keys'][int(index)]]['native_crop_positions']
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached, np.load(
            SOURCE/dataset/'fields'/(index+'.npz'), allow_pickle=False) as fields:
        def extension(**kwargs):
            extend(cached, fields, metadata, cached, positions, text_features, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    decision = json.loads((SOURCE/'suite_results.json').read_text())['decision']
    if not decision['passed']:
        raise RuntimeError('Source feasibility gate failed; no prediction evaluation.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
