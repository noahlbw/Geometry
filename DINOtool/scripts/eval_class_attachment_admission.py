"""Actual predictions with separated class and attachment evidence sources."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.class_attachment_admission import (IMPLEMENTATION, CONFIG, METHODS, LEGACY,
    PRIMARY, ALIAS_SHUFFLES, source_controls)
from dinotool.coherent_native_admission import coherent_observation
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.native_class_ownership import family_class_field
from eval_coherent_native_admission import extend as coherent_extend, PREVIOUS as OWNERSHIP
from eval_native_alias_noise import main, predict as original_predict, ORIGINAL
from eval_native_ownership_reader import SOURCE, PREVIOUS_NOISE
from eval_geometry_semantic_innovation import parse_args
from run_native_ownership_feasibility import native_crops
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/coherent_native_admission_20261003'


def extend(prior, sources, metadata, cached, positions, text_features, ownership, historical, **kwargs):
    coherent_extend(prior, sources, metadata, cached, positions, text_features, ownership, **kwargs)
    values, stats, frozen = [kwargs[n] for n in ('values', 'stats', 'frozen')]
    key, members, valid = [kwargs[n] for n in ('key', 'members', 'valid')]
    operator, baseline, local, broad = [kwargs[n] for n in ('operator', 'baseline', 'local', 'broad')]
    device = baseline.device
    replay = {m: float((values[m]-torch.from_numpy(historical[key+'__'+m+'__scores']).to(device)).abs().max()) for m in LEGACY}
    if max(replay.values()) != 0:
        raise RuntimeError('Previous50 numerical endpoints changed.')
    assignments = torch.empty(members.numel(), dtype=torch.long, device=device)
    for f, group in enumerate(metadata[key]['family_indices']):
        assignments[group] = f
    parents = torch.tensor(metadata[key]['parents'], device=device)
    canonical = torch.tensor(metadata[key]['canonical'], device=device)
    field, known = [torch.from_numpy(sources[key+'__'+n]).to(device) for n in ('reference', 'known')]
    features = torch.from_numpy(text_features[key]).to(device)
    conflict = semantic_contradictions(features, parents, canonical)
    native, native_count = native_crops(cached, key, positions, device)
    full, full_known = family_class_field(native, native_count, kwargs['coordinates'], members,
        torch.zeros(1, members.numel(), dtype=torch.bool, device=device), valid)
    risks, base, extra = source_controls(kwargs['risks']['NativeAlias_Exact'], field, known,
        assignments, parents, canonical, conflict, valid, members, full, full_known)
    args = (kwargs['crops'], kwargs['count'], kwargs['coordinates'], kwargs['image_size'], members)
    base_action, _, _ = coherent_observation(*args, base, valid, CONFIG)
    base_error = float((baseline+operator @ base_action-values['CoherentNoHoldout_Exact']).abs().max())
    zero, _, _ = coherent_observation(*args, torch.zeros_like(base), valid, CONFIG)
    if base_error != 0 or bool((zero != 0).any()):
        raise RuntimeError('Class/view base or zero-risk identity changed.')
    diag = {'legacy50_score_replay_max_errors': replay, 'base_score_replay_max_error': base_error,
        'zero_risk_action_max': float(zero.abs().max()), 'sources': {},
        'extra_attachment_mean_risk': float(extra[valid].mean()),
        'extra_attachment_active_fraction': float((extra[valid] > 0).double().mean()),
        'extra_canonical_risk_max': float(extra[:, canonical].abs().max())}
    grouped_extra = extra[:, members]
    for name, risk in risks.items():
        action, cap, entry = coherent_observation(*args, risk, valid, CONFIG)
        values[name] = baseline+operator @ action
        entry['incremental_suppression_over_class_base'] = float(-(action-base_action)[valid].mean())
        entry['final_canonical_mean_risk'] = float(risk[valid][:, canonical].mean())
        if name in ALIAS_SHUFFLES:
            changed_extra = (risk-base)/(1-base).clamp_min(1e-30)
            resolved = (1-base) > 1e-12
            expected = grouped_extra.masked_fill(~resolved[:, members], 0.)
            # Saturated bases hide extra risks, so verify only unsaturated classes.
            class_known = resolved[:, members].all(-1)
            error = (changed_extra[:, members].sort(-1).values-expected.sort(-1).values).abs()
            entry['extra_risk_spectrum_max_error'] = float(error[class_known].max()) if bool(class_known.any()) else 0.
            entry['extra_spectrum_known_class_fraction'] = float(class_known[valid].double().mean())
        diag['sources'][name] = entry
        if name == PRIMARY:
            values['ClassAttachmentMeanLogit'] = .5*(local.double()+broad.double()+action)
            frozen[key+'__class_attachment_risk'] = risk.cpu().numpy()
            frozen[key+'__class_base_risk'] = base.cpu().numpy()
            frozen[key+'__extra_attachment_risk'] = extra.cpu().numpy()
            frozen[key+'__class_attachment_action'] = action.cpu().numpy()
            frozen[key+'__class_attachment_capacity'] = cap.cpu().numpy()
    if diag['extra_canonical_risk_max'] != 0:
        raise RuntimeError('Extra attachment modified canonical words.')
    stats['class_attachment'] = diag


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset = relative.parts[0]
    row = json.loads((PREVIOUS_NOISE/dataset/'merged.json').read_text())
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    text_features = {}
    for key in metadata:
        with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as arrays:
            text_features[key] = arrays['mean_template_features']
    positions = row['diagnostics'][row['sample_keys'][int(relative.stem)]]['native_crop_positions']
    with np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached, np.load(
            SOURCE/dataset/'fields'/(relative.stem+'.npz'), allow_pickle=False) as fields, np.load(
            OWNERSHIP/relative, allow_pickle=False) as ownership, np.load(
            PREVIOUS/relative, allow_pickle=False) as historical:
        def extension(**kwargs):
            extend(cached, fields, metadata, cached, positions, text_features, ownership, historical, **kwargs)
        return original_predict(image, geometry, banks, vip, variants, original, previous, extension, METHODS)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Completed previous50 prediction suite required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
