"""Fixed-window prediction with physical8 references; no historical arm expansion."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.coherent_native_admission import coherent_observation, hard_observation
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.fine_reference_admission import (IMPLEMENTATION, CONFIG, METHODS, PRIMARY,
    REPLAY, fine_crop_positions, fine_patch_features, fine_source_controls)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.semantic_reference_admission import weighted_family_field
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_matched_contribution_alias import crop_from_features
from eval_native_alias_noise import main, ORIGINAL
from eval_native_ownership_reader import SOURCE, PREVIOUS_NOISE
from eval_semantic_role_admission import SEMANTIC
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/semantic_reference_admission_20261003'


@torch.inference_mode()
def fine_observations(image, vip, queries, coordinates, valid):
    h, w = min(512, image.shape[-2]), min(512, image.shape[-1])
    positions = fine_crop_positions(h, w)
    count = torch.zeros(512, 512, device=vip.device)
    crops = {key: [] for key in queries}
    frozen, coverage = {}, torch.zeros(len(valid), device=vip.device)
    for t, l, ch, cw in positions:
        count[t:t+ch, l:l+cw] += 1
    count.clamp_min_(1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    replay_error = None
    head_grid = vip.backbone.model.visual_model.head.patch_size
    for i, (t, l, ch, cw) in enumerate(positions):
        rgb = F.pad(image[:, t:t+ch, l:l+cw].to(vip.device), (0, 256-cw, 0, 256-ch))
        resized = F.interpolate(rgb[None], (512, 512), mode='bilinear', align_corners=False)[0]
        if i == 0:
            replay_rgb = F.interpolate(rgb[None], (336, 336), mode='bilinear', align_corners=False)[0]
            replay_error = float((fine_patch_features(vip, replay_rgb)-vip.crop_patch_features(replay_rgb)).abs().max())
            if replay_error != 0:
                raise RuntimeError('Resolution-generalized observer does not exactly replay336.')
        features = fine_patch_features(vip, resized)
        if vip.backbone.model.visual_model.head.patch_size != head_grid:
            raise RuntimeError('Fine reference mutated original head layout.')
        blank = WideCrop(features.new_empty(1024, 1), features.new_empty(1), t, l, ch, cw, 32, 256)
        ids, coeff = crop_stencil(blank, count, coordinates, (512, 512))
        coverage += coeff.sum(-1)
        for key, query in queries.items():
            crop = replace(crop_from_features(features, query, blank), grid_side=32, crop_side=256)
            crops[key].append(crop)
            prefix = key+f'__fine_crop{i}__'
            frozen[prefix+'mean_template_logits'] = crop.alias_logits.cpu().numpy()
            frozen[prefix+'salience'] = crop.salience.cpu().numpy()
    torch.cuda.synchronize()
    error = float((coverage[valid]-1).abs().max()) if bool(valid.any()) else 0.
    if error > 1e-6:
        raise RuntimeError('Fine physical footprint coverage differs.')
    return crops, count, frozen, {'fine_forwards': len(positions), 'observer336_replay_max_error': replay_error,
        'observer336_diagnostic_forwards': 2, 'fine_coverage_max_error': error,
        'fine_crop_positions': positions, 'fine_observation_seconds': time.perf_counter()-started,
        'fine_token_original_pixels': 8., 'original_head_layout_unchanged': True}


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset, device = relative.parts[0], geometry.device
    metadata = json.loads((SOURCE/dataset/'families.json').read_text())
    semantic = json.loads((SEMANTIC/(dataset+'.json')).read_text())
    if not semantic['sentinel_contract_passed']:
        raise RuntimeError('Changed frozen semantic source contract.')
    coordinates, valid, operator = [torch.from_numpy(original[k]).to(device) for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Original Geometry operator differs.')
    broad, wide, _, wide_count = prepare_wide(image, vip, variants)
    queries = {s+'__'+p: q for s, (_, qs) in variants.items() for p, q in qs.items()}
    fine, fine_count, frozen, costs = fine_observations(image, vip, queries, coordinates, valid)
    output, diag = {}, {'operator_replay_max_error': operator_error, **costs,
        'wide_forwards': len(next(iter(wide.values()))), 'new_intervention_forwards': 0,
        'local_replay_max_error': {}, 'scenarios': {}}
    local = {}
    for p, bank in banks.items():
        actual = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
            bank.parent_indices, bank.class_count)/.07
        local[p] = torch.from_numpy(original[p+'__local']).to(device)
        error = float((actual-local[p]).abs().max())
        diag['local_replay_max_error'][p] = error
        if error > 1e-4:
            raise RuntimeError('Original local Geometry differs.')
    with np.load(PREVIOUS/relative, allow_pickle=False) as history, np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as cached:
        for scene, (vbanks, _) in variants.items():
            output[scene], diag['scenarios'][scene] = {}, {}
            for p, bank in vbanks.items():
                key = scene+'__'+p
                values = {name: torch.from_numpy(history[key+'__'+old+'__scores']).to(device) for name, old in REPLAY.items()}
                baseline = values['NoAdmission_Exact']
                b = sample_broad(broad[scene, p], 0, 0, *image.shape[-2:]).reshape_as(local[p])
                error = float((baseline-(local[p].double()+operator @ (b.double()-local[p].double()))).abs().max())
                if error > 1e-4:
                    raise RuntimeError('Original wide/local reconstruction differs.')
                members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
                parents = torch.tensor(metadata[key]['parents'], device=device)
                canonical = torch.tensor(metadata[key]['canonical'], device=device)
                assignments = torch.empty(members.numel(), dtype=torch.long, device=device)
                excluded = torch.zeros(len(metadata[key]['family_indices']), members.numel(), dtype=torch.bool, device=device)
                for family, group in enumerate(metadata[key]['family_indices']):
                    assignments[group], excluded[family, group] = family, True
                conflict = torch.tensor(semantic['sources'][key]['semantic_conflict'], dtype=torch.float64, device=device)
                with np.load(SOURCE/dataset/(key+'__text.npz'), allow_pickle=False) as text:
                    cosine = semantic_contradictions(torch.from_numpy(text['mean_template_features']).to(device), parents, canonical)
                view = torch.from_numpy(cached[key+'__risk']).to(device)
                arguments = (fine[key], fine_count, coordinates, members, excluded, valid)
                risks, fields = fine_source_controls(*arguments, assignments, parents, canonical, conflict, cosine, view)
                selected = excluded[0].nonzero().flatten()
                changed = [replace(c, alias_logits=c.alias_logits.clone(), salience=c.salience.clone()) for c in fine[key]]
                for crop in changed:
                    crop.alias_logits[:, selected], crop.salience[selected] = 10000, 10000
                checked = weighted_family_field(changed, fine_count, coordinates, members, excluded[:1], valid, fields['admissible'], CONFIG)[0]
                independence = float((checked[:, 0]-fields['held_reference'][:, 0]).abs().max())
                if independence != 0:
                    raise RuntimeError('Fine held-out family confirms itself.')
                writer_args = (wide[scene, p], wide_count, coordinates, tuple(image.shape[-2:]), members)
                zero = coherent_observation(*writer_args, torch.zeros_like(risks[PRIMARY]), valid, CONFIG)[0]
                if bool((zero != 0).any()):
                    raise RuntimeError('Unchanged writer zero-risk identity differs.')
                details = {'baseline_readout_replay_max_error': error, 'held_family_independence_max_error': independence,
                    'semantic_weight_spectrum_errors': fields['semantic_spectrum_errors'],
                    'spatial_risk_spectrum_error': fields['spatial_risk_spectrum_error'],
                    'zero_semantic_identity_max_error': float((risks[PRIMARY]-risks['FineReferenceOnly_Exact']).abs().max())
                        if bool((conflict == 0).all()) else None,
                    'zero_risk_writer_max_error': float(zero.abs().max()),
                    'extra_canonical_risk_max': float(fields['extra'][:, canonical].abs().max()),
                    'mean_reference_admissibility': float(fields['admissible'].mean()), 'sources': {}}
                for name, risk in risks.items():
                    action, cap, stats = coherent_observation(*writer_args, risk, valid, CONFIG)
                    values[name] = baseline+operator @ action
                    details['sources'][name] = stats
                    frozen[key+'__'+name+'__risk'] = risk.cpu().numpy()
                    frozen[key+'__'+name+'__action'] = action.cpu().numpy()
                hard, hard_stats = hard_observation(*writer_args, risks[PRIMARY], valid, CONFIG)
                values['FineReferenceHardDelete_Exact'] = baseline+operator @ hard
                details['hard'] = hard_stats
                if set(values) != set(METHODS):
                    raise RuntimeError('Missing fixed fine-reference arms.')
                for name, value in values.items():
                    frozen[key+'__'+name+'__scores'] = value.cpu().numpy()
                frozen[key+'__fine_full_reference'] = fields['full_reference'].cpu().numpy()
                frozen[key+'__fine_full_known'] = fields['full_known'].cpu().numpy()
                frozen[key+'__fine_attachment_risk'] = fields['extra'].cpu().numpy()
                output[scene][p], diag['scenarios'][scene][p] = values, details
    if not all(np.isfinite(v).all() for v in frozen.values()):
        raise RuntimeError('Nonfinite fine-reference pre-mask cache.')
    return output, frozen, diag


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Complete previous prediction suite required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
