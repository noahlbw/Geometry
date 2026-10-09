"""Actual-profile reliability observations with the unchanged competitive writer."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.alias_spatial_allocation import ROUNDING_GUARD
from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.matched_contribution_alias import (IMPLEMENTATION, CONFIG, PRIMARY, METHODS, LEGACY,
    ALIAS_SHUFFLES, matched_reversal, stencil_margins)
from dinotool.pair_context_reader import pair_observation
from dinotool.rival_preserving_alias import competitive_potential, directional_controls
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from dinotool.target_context_alias import (TargetContextConfig, alias_permutations, crop_masks,
    geometry_masks, shuffled_crop_masks)
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_matched_head_fov import resize_rgb
from eval_pair_context_reader import main, ORIGINAL
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/rival_preserving_alias_reader_screen_20261003'
SUPPORT_CONFIG = TargetContextConfig()


def crop_from_features(features, query, original_crop):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        similarity = torch.einsum('bnd,mtd->bnmt', features, query.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(query.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0]/SETTINGS.tem).float()
        logits = (similarity*SETTINGS.logit_scale).float()
    return WideCrop(logits, salience, original_crop.top, original_crop.left,
                    original_crop.actual_height, original_crop.actual_width)


@torch.inference_mode()
def observations(image, vip, queries, crops, count, coordinates, valid, masks, assignments, supported, members):
    n = len(valid)
    names = ('full', 'kept', 'removed', 'shuffle_kept', 'shuffle_removed')
    measurements = {p: {name: torch.zeros((1 if name == 'full' else 2, n, len(q.aliases), len(q.class_names)), device=vip.device)
                        for name in names} for p, q in queries.items()}
    class_fields = {p: torch.zeros(n, len(q.class_names), device=vip.device) for p, q in queries.items()}
    frozen, forwards, seconds = {}, {'true': 0, 'shuffle': 0}, {'true': 0., 'shuffle': 0.}
    resized = resize_rgb(image, 448).to(vip.device)
    spectrum_error = 0.
    for number, crop in enumerate(next(iter(crops.values()))):
        indices, coefficients = crop_stencil(crop, count, coordinates, tuple(image.shape[-2:]))
        active = valid & (coefficients.sum(-1) > 0)
        if not bool(active.any()):
            continue
        raw_full = {}
        for p in queries:
            actual = crops[p][number]
            evidence = profiled_logits(actual, members[p])
            raw_full[p] = stencil_margins(evidence, indices, coefficients, CONFIG.beta)
            measurements[p]['full'][0] += raw_full[p]
            score = (CONFIG.beta*evidence).logsumexp(-1)/CONFIG.beta
            class_fields[p] += (score[indices]*coefficients[..., None]).sum(1)
            frozen[p+f'__crop{number}__full_mean_template_logits'] = actual.alias_logits.cpu().numpy()
            frozen[p+f'__crop{number}__full_salience'] = actual.salience.cpu().numpy()
        rgb = F.pad(resized[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width],
                    (0, 336-crop.actual_width, 0, 336-crop.actual_height))
        mapped = crop_masks(masks, crop, count.shape, tuple(image.shape[-2:]))
        shuffled = shuffled_crop_masks(mapped, crop.actual_height, crop.actual_width, SUPPORT_CONFIG.random_seed+number)
        spectrum_error = max(spectrum_error, float((mapped[:, :crop.actual_height, :crop.actual_width].flatten(1).sort(-1).values
            -shuffled[:, :crop.actual_height, :crop.actual_width].flatten(1).sort(-1).values).abs().max()))
        visible = torch.zeros((336, 336), device=vip.device, dtype=torch.bool)
        visible[:crop.actual_height, :crop.actual_width] = True
        fills = (rgb.new_tensor([.485, .456, .406])[:, None, None], rgb[:, visible].mean(-1)[:, None, None])
        for cell in assignments[active].unique().tolist():
            use = active & (assignments == cell)
            for kind, support in (('true', mapped[cell]), ('shuffle', shuffled[cell])):
                for fill_index, fill in enumerate(fills):
                    for suffix, mask in (('kept', support), ('removed', 1-support)):
                        name = suffix if kind == 'true' else 'shuffle_'+suffix
                        if not bool(supported[cell]):
                            for p in queries:
                                measurements[p][name][fill_index, use] += raw_full[p][use]
                            continue
                        changed = (mask[None]*rgb+(1-mask[None])*fill).masked_fill(~visible[None], 0.)
                        torch.cuda.synchronize()
                        begun = time.perf_counter()
                        features = vip.crop_patch_features(changed)
                        for p, query in queries.items():
                            observed = crop_from_features(features, query, crops[p][number])
                            measured = stencil_margins(profiled_logits(observed, members[p]), indices[use], coefficients[use], CONFIG.beta)
                            measurements[p][name][fill_index, use] += measured
                            prefix = p+f'__crop{number}_cell{cell}_{name}_fill{fill_index}__'
                            frozen[prefix+'mean_template_logits'] = observed.alias_logits.cpu().numpy()
                            frozen[prefix+'salience'] = observed.salience.cpu().numpy()
                        torch.cuda.synchronize()
                        forwards[kind] += 1
                        seconds[kind] += time.perf_counter()-begun
    return measurements, class_fields, frozen, {'source_forwards': forwards, 'source_seconds': seconds,
        'support_shuffle_spectrum_error': spectrum_error, 'new_intervention_forwards': sum(forwards.values())}


@torch.inference_mode()
def predict(image, geometry, banks, vip, queries, original, source):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    if len(relative.parts) != 3 or relative.parts[1] != 'numerical_cache':
        raise ValueError('Verified source sequence required.')
    device, started = geometry.device, time.perf_counter()
    coordinates = torch.from_numpy(original['coordinates']).to(device)
    valid = torch.from_numpy(original['valid']).to(device)
    operator = torch.from_numpy(original['operator']).to(device)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    masks, assignments, supported = geometry_masks(prepared.geometry_patch_conditional[0], coordinates, valid, SUPPORT_CONFIG)
    if (operator_error > 1e-10 or not np.array_equal(masks.cpu().numpy(), source['support_masks'])
            or not np.array_equal(assignments.cpu().numpy(), source['assignments'])
            or not np.array_equal(supported.cpu().numpy(), source['supported_units'])):
        raise RuntimeError('Changed Geometry operator/shared support.')
    broad, all_crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    crops = {p: all_crops['clean', p] for p in banks}
    members = {p: torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
               for p, bank in banks.items()}
    canonical = {p: torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                    for c in range(bank.class_count)]) for p, bank in banks.items()}
    endpoints, frozen, diagnostics = {}, {}, {'operator_replay_max_error': operator_error,
        'shared_support_replay_exact': True, 'unmasked_wide_forwards': len(next(iter(crops.values())))}
    with np.load(PREVIOUS/relative, allow_pickle=False) as prior:
        for p, bank in banks.items():
            aliases = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
            local = alias_class_scores(aliases, bank.parent_indices, bank.class_count)/.07
            actual_b = sample_broad(broad['clean', p], 0, 0, *image.shape[-2:]).reshape_as(local)
            errors = {'local': float((local-torch.from_numpy(original[p+'__local']).to(device)).abs().max()),
                'broad': float((actual_b-torch.from_numpy(original['clean__'+p+'__broad']).to(device)).abs().max())}
            if max(errors.values()) > 1e-4:
                raise RuntimeError('Changed original readout: '+str(errors))
            endpoints[p] = {m: torch.from_numpy(prior[p+'__'+m+'__scores']).to(device) for m in LEGACY}
            diagnostics[p] = {'replay_max_errors': errors, 'sources': {}, 'controls': {}}
    observed, profiled_class_fields, source_fields, costs = observations(image, vip, queries, crops, count,
        coordinates, valid, masks, assignments, supported, members)
    frozen.update(source_fields)
    frozen.update({'support_masks': masks.cpu().numpy(), 'assignments': assignments.cpu().numpy(),
                   'supported_units': supported.cpu().numpy()})
    diagnostics.update(costs)
    known = supported[assignments] & valid
    for p, bank in banks.items():
        row = observed[p]
        baseline = endpoints[p]['Anchored_Exact']
        local, b = [torch.from_numpy(original[name]).to(device) for name in (p+'__local', 'clean__'+p+'__broad')]
        profile_error = float((profiled_class_fields[p]-b).abs().max())
        if profile_error > 1e-4:
            raise RuntimeError('Actual profiled class field failed original broad-score replay: '+str(profile_error))
        diagnostics[p]['replay_max_errors']['profiled_broad'] = profile_error
        primary_risk = matched_reversal(row['full'][0], row['kept'], row['removed'], bank.parent_indices, canonical[p], known)
        shuffled = matched_reversal(row['full'][0], row['shuffle_kept'], row['shuffle_removed'], bank.parent_indices, canonical[p], known)
        text = semantic_contradictions(queries[p].features.float().mean(1), bank.parent_indices, canonical[p], CONFIG.epsilon)
        text = text[None].expand_as(primary_risk).masked_fill(~valid[:, None, None], 0.).clone()
        text.scatter_(-1, bank.parent_indices[None, :, None].expand(len(valid), -1, 1), 0)
        risks = {PRIMARY: primary_risk, 'MatchedShuffledSupport_Exact': shuffled, 'MatchedTextOnly_Exact': text}
        for name, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members[p], canonical[p], CONFIG.random_seed)):
            changed = torch.empty_like(primary_risk)
            grouped = primary_risk[:, members[p]]
            changed[:, members[p]] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
            risks[name] = changed
        cap = torch.from_numpy(original['clean__'+p+'__protected_cap']).to(device)
        identity = pair_observation(crops[p], count, coordinates, tuple(image.shape[-2:]), members[p],
            torch.zeros_like(primary_risk), valid, CONFIG.beta, CONFIG.query_chunk)[2]
        if bool((identity != 0).any()):
            raise RuntimeError('Zero-risk original writer identity failed.')
        primary_directed = None
        for method, gamma in risks.items():
            directed = pair_observation(crops[p], count, coordinates, tuple(image.shape[-2:]), members[p],
                gamma, valid, CONFIG.beta, CONFIG.query_chunk)[2]
            excess = float((-directed-cap[..., None]).clamp_min(0).max())
            if excess > ROUNDING_GUARD or float(gamma[:, canonical[p]].abs().max()) != 0:
                raise RuntimeError('Original protected directed capacity failed.')
            potential, margin, stats = competitive_potential(directed, valid)
            endpoints[p][method] = baseline+operator @ potential
            for name, value in (('risk', gamma), ('directed', directed), ('potential', potential), ('requested', margin)):
                frozen[p+'__'+method+'__'+name] = value.cpu().numpy()
            diagnostics[p]['sources'][method] = {**stats, 'directed_capacity_excess_max': excess,
                'canonical_risk_max': float(gamma[:, canonical[p]].abs().max()), 'zero_risk_writer_max': float(identity.abs().max()),
                'mean_risk': float(gamma[valid].mean()), 'active_risk_fraction': float((gamma[valid] > 0).float().mean()),
                'unknown_fraction': float((gamma[valid] == 0).float().mean()),
                'alias_spectrum_max_error': float((gamma[:, members[p]].sort(2).values-primary_risk[:, members[p]].sort(2).values).abs().max())
                    if method in ALIAS_SHUFFLES else None}
            if method == PRIMARY:
                primary_directed = directed
                endpoints[p]['MatchedMeanLogit'] = .5*(local.double()+b.double()+potential)
        for old_name, changed in directional_controls(primary_directed.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
            method = 'Matched'+old_name
            directed = torch.from_numpy(changed).to(device)
            potential, _, stats = competitive_potential(directed, valid)
            endpoints[p][method] = baseline+operator @ potential
            frozen[p+'__'+method+'__directed'] = changed
            frozen[p+'__'+method+'__potential'] = potential.cpu().numpy()
            diagnostics[p]['controls'][method] = {**stats,
                'pair_budget_max_error': float((directed.sum(0)-primary_directed.double().sum(0)).abs().max()),
                'capacity_excess_max': float((-directed-cap.double()[..., None]).clamp_min(0).max()),
                'pair_spectrum_max_error': float((directed[valid].sort(0).values-primary_directed.double()[valid].sort(0).values).abs().max())}
        for name, values in row.items():
            frozen[p+'__matched_'+name+'_margin'] = values.cpu().numpy()
        for method, value in endpoints[p].items():
            frozen[p+'__'+method+'__scores'] = value.cpu().numpy()
        diagnostics[p]['supported_query_fraction'] = float(known[valid].float().mean())
        if set(endpoints[p]) != set(METHODS):
            raise RuntimeError('Missing matched/historical arms.')
    torch.cuda.synchronize()
    diagnostics['all_observation_reader_seconds'] = time.perf_counter()-started
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask source fields.')
    return endpoints, frozen, diagnostics


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION, config=CONFIG)
