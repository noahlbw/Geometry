"""Actual predictions using frozen physical8 alias evidence and original writer."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.coherent_native_admission import coherent_observation, hard_observation, native_risk, union_risk
from dinotool.fine_alias_view import (IMPLEMENTATION, CONFIG, METHODS, REPLAY, PRIMARY,
    independent_observation, physical_margins, view_sources)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.stratified_soft_alias import WideCrop
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_native_alias_noise import main, ORIGINAL
from eval_native_ownership_reader import PREVIOUS_NOISE
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/fine_reference_admission_20261003'


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    dataset, device = relative.parts[0], geometry.device
    prior = json.loads((PREVIOUS/dataset/'merged.json').read_text())
    positions = prior['diagnostics'][prior['sample_keys'][int(relative.stem)]]['fine_crop_positions']
    coordinates, valid, operator = [torch.from_numpy(original[k]).to(device) for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Original Geometry operator changed.')
    broad, wide, _, count = prepare_wide(image, vip, variants)
    fine_count = torch.zeros(512, 512, device=device)
    for t, l, h, w in positions:
        fine_count[t:t+h, l:l+w] += 1
    fine_count.clamp_min_(1)
    local = {}
    diag = {'operator_replay_max_error': operator_error, 'local_replay_max_error': {},
        'new_image_observation_forwards': 0, 'new_intervention_forwards': 0,
        'wide_replay_forwards': len(next(iter(wide.values()))), 'fine_token_original_pixels': 8.,
        'cached_fine_source_precedes_masks': True, 'scenarios': {}}
    for p, bank in banks.items():
        actual = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
            bank.parent_indices, bank.class_count)/.07
        local[p] = torch.from_numpy(original[p+'__local']).to(device)
        error = float((actual-local[p]).abs().max())
        diag['local_replay_max_error'][p] = error
        if error > 1e-4:
            raise RuntimeError('Original local scores changed.')
    output, frozen = {}, {}
    with np.load(PREVIOUS/relative, allow_pickle=False) as cached, np.load(PREVIOUS_NOISE/relative, allow_pickle=False) as native:
        for scene, (vbanks, _) in variants.items():
            output[scene], diag['scenarios'][scene] = {}, {}
            for p, bank in vbanks.items():
                key = scene+'__'+p
                values = {name: torch.from_numpy(cached[key+'__'+old+'__scores']).to(device) for name, old in REPLAY.items()}
                baseline = values['NoAdmission_Exact']
                b = sample_broad(broad[scene, p], 0, 0, *image.shape[-2:]).reshape_as(local[p])
                replay = float((baseline-(local[p].double()+operator @ (b.double()-local[p].double()))).abs().max())
                if replay > 1e-4:
                    raise RuntimeError('Wide/local reconstruction changed.')
                members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
                canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                    for c in range(bank.class_count)])
                fine_crops = [WideCrop(torch.from_numpy(cached[key+f'__fine_crop{i}__mean_template_logits']).to(device),
                    torch.from_numpy(cached[key+f'__fine_crop{i}__salience']).to(device), *pos, 32, 256)
                    for i, pos in enumerate(positions)]
                full = torch.from_numpy(cached[key+'__fine_full_reference']).to(device)
                known = torch.from_numpy(cached[key+'__fine_full_known']).to(device)
                class_risk = native_risk(full, known, torch.zeros(members.numel(), dtype=torch.long, device=device),
                    bank.parent_indices, valid)
                fine = physical_margins(fine_crops, fine_count, coordinates, members, valid, CONFIG)
                wide_margin = torch.from_numpy(native[key+'__broad_margin']).to(device)
                sources, fields = view_sources(wide_margin, fine, bank.parent_indices, canonical, valid, members, class_risk, CONFIG)
                old_view = torch.from_numpy(native[key+'__risk']).to(device).double().amax(-1)
                args = (wide[scene, p], count, coordinates, tuple(image.shape[-2:]), members)
                old_risk = union_risk(old_view, class_risk)
                saved_old_risk = torch.from_numpy(cached[key+'__FineReferenceOnly_Exact__risk']).to(device)
                old_risk_replay = float((old_risk-saved_old_risk).abs().max())
                if old_risk_replay != 0:
                    raise RuntimeError('Original native16/class risk changed: '+str(old_risk_replay))
                old_action = coherent_observation(*args, old_risk, valid, CONFIG)[0]
                old_replay = float((baseline+operator @ old_action-values['FineNative16_Exact']).abs().max())
                if old_replay != 0:
                    raise RuntimeError('Original physical8 reference/native16 view candidate changed: '+str(old_replay))
                zero = coherent_observation(*args, torch.zeros_like(class_risk), valid, CONFIG)[0]
                details = {'baseline_readout_replay_max_error': replay, 'native16_control_score_replay_max_error': old_replay,
                    'native16_control_risk_replay_max_error': old_risk_replay,
                    'zero_risk_writer_max_error': float(zero.abs().max()),
                    **{k: v for k, v in fields.items() if k not in ('pair', 'view')}, 'sources': {}}
                if bool((zero != 0).any()):
                    raise RuntimeError('Zero-risk original writer identity changed.')
                for name, risk in sources.items():
                    action, cap, stats = coherent_observation(*args, risk, valid, CONFIG)
                    values[name] = baseline+operator @ action
                    details['sources'][name] = stats
                    frozen[key+'__'+name+'__action'] = action.cpu().numpy()
                hard, hard_stats = hard_observation(*args, sources[PRIMARY], valid, CONFIG)
                values['FineAliasViewHardDelete_Exact'] = baseline+operator @ hard
                details['hard'] = hard_stats
                observer, observer_stats = independent_observation(b, *args, fields['pair'],
                    valid, canonical, torch.from_numpy(native[key+'__risk']).to(device), CONFIG)
                values.update(observer)
                details['independent_observer'] = observer_stats
                if set(values) != set(METHODS):
                    raise RuntimeError('Missing declared physical8 alias-view endpoint.')
                for name, value in values.items():
                    frozen[key+'__'+name+'__scores'] = value.cpu().numpy()
                frozen[key+'__fine_margin'] = fine.cpu().numpy()
                frozen[key+'__fine_view_pair_risk'] = fields['pair'].cpu().numpy()
                frozen[key+'__fine_class_risk'] = class_risk.cpu().numpy()
                frozen[key+'__old_view_pair_risk'] = native[key+'__risk']
                output[scene][p], diag['scenarios'][scene][p] = values, details
    if not all(np.isfinite(v).all() for v in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask fine-view fields.')
    return output, frozen, diag


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Complete frozen physical8 source required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
