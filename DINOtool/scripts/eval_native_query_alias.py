"""Query-owned unmasked native witnesses; original Geometry and writer unchanged."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.matched_contribution_alias import stencil_margins
from dinotool.native_query_alias import (IMPLEMENTATION, CONFIG, PRIMARY, METHODS, LEGACY,
    ALIAS_SHUFFLES, native_crop_positions, native_risk, query_relation, support_controls, support_margins)
from dinotool.pair_context_reader import pair_observation
from dinotool.rival_preserving_alias import competitive_potential, directional_controls
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from dinotool.target_context_alias import alias_permutations
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_matched_contribution_alias import crop_from_features
from eval_pair_context_reader import main, ORIGINAL
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/matched_contribution_alias_screen_20261003'


@torch.inference_mode()
def native_observations(image, vip, queries, coordinates, valid, members):
    ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
    count = torch.zeros(512, 512, device=vip.device)
    positions = native_crop_positions(ah, aw)
    for t, l, h, w in positions:
        count[t:t+h, l:l+w] += 1
    if not bool((count[:ah, :aw] > 0).all()):
        raise RuntimeError('Native observation leaves uncovered image pixels.')
    count = count.clamp_min(1)
    margins = {p: torch.zeros(len(valid), len(q.aliases), len(q.class_names), device=vip.device)
               for p, q in queries.items()}
    classes = {p: torch.zeros(len(valid), len(q.class_names), device=vip.device) for p, q in queries.items()}
    coverage, frozen = torch.zeros(len(valid), device=vip.device), {}
    torch.cuda.synchronize()
    begun = time.perf_counter()
    for number, (top, left, h, w) in enumerate(positions):
        rgb = F.pad(image[:, top:top+h, left:left+w].to(vip.device), (0, 336-w, 0, 336-h))
        features = vip.crop_patch_features(rgb)
        blank = WideCrop(features.new_empty(441, 1), features.new_empty(1), top, left, h, w)
        indices, coefficients = crop_stencil(blank, count, coordinates, (512, 512))
        coverage += coefficients.sum(-1)
        for p, q in queries.items():
            crop = crop_from_features(features, q, blank)
            evidence = profiled_logits(crop, members[p])
            margins[p] += stencil_margins(evidence, indices, coefficients, CONFIG.beta)
            score = (CONFIG.beta*evidence).logsumexp(-1)/CONFIG.beta
            classes[p] += (score[indices]*coefficients[..., None]).sum(1)
            prefix = p+f'__native_crop{number}__'
            frozen[prefix+'mean_template_logits'] = crop.alias_logits.cpu().numpy()
            frozen[prefix+'salience'] = crop.salience.cpu().numpy()
    torch.cuda.synchronize()
    error = float((coverage[valid]-1).abs().max())
    if error > 1e-6:
        raise RuntimeError('Native query coordinates/coverage differ: '+str(error))
    return margins, classes, frozen, {'native_forwards': len(positions),
        'native_observation_seconds': time.perf_counter()-begun, 'native_coverage_max_error': error,
        'native_token_original_pixels': 16., 'native_crop_positions': positions}


@torch.inference_mode()
def predict(image, geometry, banks, vip, queries, original, source):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    device, begun = geometry.device, time.perf_counter()
    coordinates = torch.from_numpy(original['coordinates']).to(device)
    valid = torch.from_numpy(original['valid']).to(device)
    operator = torch.from_numpy(original['operator']).to(device)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    relation = prepared.geometry_patch_conditional[0]
    actual_operator, _ = reconstruction_operator(relation, valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Changed Geometry reconstruction operator.')
    weights, known = query_relation(relation, valid)
    controls = support_controls(weights, coordinates, valid)
    support_spectrum = float((weights.sort(-1).values-controls['shuffle'].sort(-1).values).abs().max())
    broad, all_crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    crops = {p: all_crops['clean', p] for p in banks}
    members = {p: torch.stack([(b.parent_indices == c).nonzero().flatten() for c in range(b.class_count)]) for p, b in banks.items()}
    canonical = {p: torch.stack([(b.canonical_mask & (b.parent_indices == c)).nonzero().flatten()[0]
        for c in range(b.class_count)]) for p, b in banks.items()}
    native, native_classes, frozen, costs = native_observations(image, vip, queries, coordinates, valid, members)
    frozen.update({'query_support': weights.cpu().numpy(), 'valid': valid.cpu().numpy()})
    diagnostics = {'operator_replay_max_error': operator_error, 'support_shuffle_spectrum_error': support_spectrum,
        'unmasked_wide_forwards': len(next(iter(crops.values()))), 'new_intervention_forwards': 0, **costs}
    endpoints = {}
    with np.load(PREVIOUS/relative, allow_pickle=False) as prior:
        for p, bank in banks.items():
            local = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
                bank.parent_indices, bank.class_count)/.07
            b = torch.from_numpy(original['clean__'+p+'__broad']).to(device)
            actual_b = sample_broad(broad['clean', p], 0, 0, *image.shape[-2:]).reshape_as(local)
            errors = {'local': float((local-torch.from_numpy(original[p+'__local']).to(device)).abs().max()),
                'broad': float((actual_b-b).abs().max())}
            if max(errors.values()) > 1e-4:
                raise RuntimeError('Original readout replay differs: '+str(errors))
            endpoints[p] = {m: torch.from_numpy(prior[p+'__'+m+'__scores']).to(device) for m in LEGACY}
            baseline = endpoints[p]['Anchored_Exact']
            full, profiled = torch.zeros_like(native[p]), torch.zeros_like(b)
            for crop in crops[p]:
                ids, coefficients = crop_stencil(crop, count, coordinates, tuple(image.shape[-2:]))
                evidence = profiled_logits(crop, members[p])
                full += stencil_margins(evidence, ids, coefficients, CONFIG.beta)
                profiled += ((CONFIG.beta*evidence).logsumexp(-1)[ids]/CONFIG.beta*coefficients[..., None]).sum(1)
            errors['profiled_broad'] = float((profiled-b).abs().max())
            errors['cached_full_margin'] = float((full-torch.from_numpy(prior[p+'__matched_full_margin'][0]).to(device)).abs().max())
            if max(errors.values()) > 1e-4:
                raise RuntimeError('Changed actual broad contribution source.')
            supported = support_margins(weights, native[p])
            gamma = native_risk(full, native[p], supported, bank.parent_indices, canonical[p], known)
            risks = {PRIMARY: gamma, 'NativeQueryOnly_Exact': native_risk(full, native[p], native[p], bank.parent_indices, canonical[p], known)}
            for kind, name in (('shared', 'NativeSharedSupport_Exact'), ('shuffle', 'NativeShuffledSupport_Exact')):
                risks[name] = native_risk(full, native[p], support_margins(controls[kind], native[p]),
                    bank.parent_indices, canonical[p], known)
            for name, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members[p], canonical[p], CONFIG.random_seed)):
                changed = torch.empty_like(gamma)
                grouped = gamma[:, members[p]]
                changed[:, members[p]] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
                risks[name] = changed
            cap = torch.from_numpy(original['clean__'+p+'__protected_cap']).to(device)
            zero = pair_observation(crops[p], count, coordinates, tuple(image.shape[-2:]), members[p],
                torch.zeros_like(gamma), valid, CONFIG.beta, CONFIG.query_chunk)[2]
            if bool((zero != 0).any()):
                raise RuntimeError('Zero-risk writer identity failed.')
            diagnostics[p] = {'replay_max_errors': errors, 'sources': {}, 'controls': {}}
            directed_primary = None
            for method, risk in risks.items():
                directed = pair_observation(crops[p], count, coordinates, tuple(image.shape[-2:]), members[p],
                    risk, valid, CONFIG.beta, CONFIG.query_chunk)[2]
                excess = float((-directed-cap[..., None]).clamp_min(0).max())
                if excess > 1e-6 or float(risk[:, canonical[p]].abs().max()) != 0:
                    raise RuntimeError('Original bounded writer constraint failed.')
                potential, _, stats = competitive_potential(directed, valid)
                endpoints[p][method] = baseline+operator @ potential
                for name, value in (('risk', risk), ('directed', directed), ('potential', potential)):
                    frozen[p+'__'+method+'__'+name] = value.cpu().numpy()
                diagnostics[p]['sources'][method] = {**stats, 'directed_capacity_excess_max': excess,
                    'canonical_risk_max': float(risk[:, canonical[p]].abs().max()), 'zero_risk_writer_max': float(zero.abs().max()),
                    'mean_risk': float(risk[valid].mean()), 'active_risk_fraction': float((risk[valid] > 0).float().mean()),
                    'unknown_fraction': float((risk[valid] == 0).float().mean()),
                    'alias_spectrum_max_error': float((risk[:, members[p]].sort(2).values-gamma[:, members[p]].sort(2).values).abs().max())
                        if method in ALIAS_SHUFFLES else None}
                if method == PRIMARY:
                    directed_primary = directed
                    endpoints[p]['NativeMeanLogit'] = .5*(local.double()+b.double()+potential)
            endpoints[p]['NativeObservationMeanLogit'] = .5*(local.double()+native_classes[p].double())
            endpoints[p]['NativeObservationAnchored'] = local.double()+operator @ (native_classes[p].double()-local.double())
            for name, values in directional_controls(directed_primary.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
                method = 'Native'+name
                directed = torch.from_numpy(values).to(device)
                potential, _, stats = competitive_potential(directed, valid)
                endpoints[p][method] = baseline+operator @ potential
                frozen[p+'__'+method+'__directed'] = values
                frozen[p+'__'+method+'__potential'] = potential.cpu().numpy()
                diagnostics[p]['controls'][method] = {**stats,
                    'pair_budget_max_error': float((directed.sum(0)-directed_primary.double().sum(0)).abs().max()),
                    'capacity_excess_max': float((-directed-cap.double()[..., None]).clamp_min(0).max()),
                    'pair_spectrum_max_error': float((directed[valid].sort(0).values-directed_primary.double()[valid].sort(0).values).abs().max())}
            for name, value in (('broad_margin', full), ('native_margin', native[p]), ('supported_margin', supported),
                                ('native_class', native_classes[p])):
                frozen[p+'__'+name] = value.cpu().numpy()
            for method, value in endpoints[p].items():
                frozen[p+'__'+method+'__scores'] = value.cpu().numpy()
            if set(endpoints[p]) != set(METHODS):
                raise RuntimeError('Missing primary or controls.')
    torch.cuda.synchronize()
    diagnostics['all_observation_reader_seconds'] = time.perf_counter()-begun
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask source fields.')
    return endpoints, frozen, diagnostics


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION, config=CONFIG)
