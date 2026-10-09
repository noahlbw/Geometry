"""Frozen contrastive-reversal source with the original bounded alias writer."""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.alias_spatial_allocation import matched_controls, ROUNDING_GUARD
from dinotool.contrastive_reversal_alias import IMPLEMENTATION, METHODS, PRIMARY, ALIAS_SHUFFLES, CONFIG, reversal_risk
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.target_context_alias import intervention_retention, observation_delta, alias_permutations
from audit_alias_action_capacity import dense
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_semantic_innovation import parse_args, SETTINGS
from eval_geometry_vip_reliability import sample_broad
from eval_pair_context_reader import main, RETAINED
from eval_stratified_soft_alias import tile_coordinates


@torch.inference_mode()
def predict(image, geometry, banks, vip, queries, original, source, *, extension=None):
    device = geometry.device
    coordinates = tile_coordinates(0, 0, device)
    valid = torch.from_numpy(original['valid']).to(device)
    operator = torch.from_numpy(original['operator']).to(device)
    started = time.perf_counter()
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Changed original Geometry reconstruction.')
    broad, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    torch.cuda.synchronize()
    supported = torch.from_numpy(source['supported_units']).to(device)[torch.from_numpy(source['assignments']).to(device)] & valid
    endpoints, frozen = {}, {}
    diagnostics = {'base_replay_seconds': time.perf_counter()-started, 'operator_replay_max_error': operator_error,
        'unmasked_wide_forwards': len(next(iter(crops.values()))), 'new_intervention_forwards': 0}
    for key, bank in banks.items():
        begun = time.perf_counter()
        alias = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
        local = alias_class_scores(alias, bank.parent_indices, bank.class_count)/.07
        g = torch.from_numpy(original[key+'__local']).to(device)
        b = torch.from_numpy(original['clean__'+key+'__broad']).to(device)
        actual_b = sample_broad(broad['clean', key], 0, 0, *image.shape[-2:]).reshape_as(local)
        errors = {'local': float((local-g).abs().max()), 'broad': float((actual_b-b).abs().max())}
        members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0] for c in range(bank.class_count)])
        full, kept, removed = [torch.from_numpy(source[key+'__source_'+name]).to(device) for name in ('full', 'kept', 'removed')]
        old_rho, _ = intervention_retention(full[0], kept, removed, bank.parent_indices, canonical, supported, CONFIG)
        if not torch.equal(old_rho, torch.from_numpy(source[key+'__TargetContext_Exact__retention']).to(device)):
            raise RuntimeError('Changed old target/context source.')
        old_action = observation_delta(crops['clean', key], count, coordinates, tuple(image.shape[-2:]), members, old_rho, valid, SETTINGS.tau)
        baseline = torch.from_numpy(source[key+'__Anchored_Exact__scores']).to(device)
        old_endpoint = baseline+operator @ old_action.double()
        previous = torch.from_numpy(source[key+'__TargetContext_Exact__scores']).to(device)
        errors['independent_endpoint'] = float((old_endpoint-previous).abs().max())
        if max(errors.values()) > 1e-4 or not torch.equal(dense(old_endpoint).argmax(-1), dense(previous).argmax(-1)):
            raise RuntimeError('Original controls failed actual readout replay.')
        risk = reversal_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, CONFIG)
        rho = 1-risk['reversal'].amax(-1)
        text = semantic_contradictions(bank.features, bank.parent_indices, canonical, CONFIG.epsilon)
        allocations = {PRIMARY: rho, 'RelativeDependence_Exact': 1-risk['relative_dependence'].amax(-1),
            'ShuffledSupport_Reversal': 1-reversal_risk(full[0], *[torch.from_numpy(source[key+'__source_shuffle_'+name]).to(device)
                for name in ('kept', 'removed')], bank.parent_indices, canonical, supported, CONFIG)['reversal'].amax(-1),
            'SingleFill0_Reversal': 1-reversal_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, CONFIG, 0)['reversal'].amax(-1),
            'SingleFill1_Reversal': 1-reversal_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, CONFIG, 1)['reversal'].amax(-1),
            'TextOnly_Exact': (1-text.amax(-1))[None].expand_as(rho).masked_fill(~valid[:, None], 1)}
        for method, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members, canonical, CONFIG.random_seed)):
            changed = torch.empty_like(rho)
            changed[:, members] = rho[:, members].gather(-1, permutation[None].expand(len(rho), -1, -1))
            allocations[method] = changed
        endpoints[key] = {m: torch.from_numpy(source[key+'__'+m+'__scores']).to(device) for m in RETAINED}
        diagnostics[key] = {'replay_max_errors': errors, 'source': {}, 'allocation_controls': {}}
        cap = torch.from_numpy(original['clean__'+key+'__protected_cap']).to(device)
        single_cap = torch.from_numpy(original['clean__'+key+'__single_cap']).to(device)
        primary_delta = None
        for method, retention in allocations.items():
            delta = observation_delta(crops['clean', key], count, coordinates, tuple(image.shape[-2:]), members, retention, valid, SETTINGS.tau)
            excess = float((-delta-cap).clamp_min(0).max())
            if bool((delta > 0).any()) or excess > ROUNDING_GUARD or bool((delta[~valid] != 0).any()):
                raise RuntimeError('Original bounded writer capacity failed.')
            endpoints[key][method] = baseline+operator @ delta.double()
            grouped = retention[:, members]
            score = torch.zeros(len(bank.alias_names), device=device)
            score[members] = ((1-grouped)*single_cap).sum(0)/single_cap.sum(0).clamp_min(CONFIG.epsilon)
            frozen[key+'__'+method+'__action_score'] = score.cpu().numpy()
            frozen[key+'__'+method+'__retention'] = retention.cpu().numpy()
            frozen[key+'__'+method+'__delta'] = delta.cpu().numpy()
            diagnostics[key]['source'][method] = {'mean_rejection': float((1-retention[valid]).mean()),
                'active_alias_fraction': float(((1-retention[valid]) > 1e-6).float().mean()),
                'unknown_alias_fraction': float((retention[valid] == 1).float().mean()),
                'mean_suppression': float(delta[valid].abs().mean()), 'capacity_excess_max': excess,
                'canonical_rejection_max': float((1-retention[:, canonical]).abs().max())}
            if method in ALIAS_SHUFFLES:
                diagnostics[key]['source'][method]['alias_spectrum_max_error'] = float((grouped.sort(-1).values-rho[:, members].sort(-1).values).abs().max())
            if method == PRIMARY:
                primary_delta = delta
                frozen[key+'__pair_reversal_risk'] = risk['reversal'].cpu().numpy()
        endpoints[key]['Reversal_MeanLogit'] = .5*(g.double()+b.double()+primary_delta.double())
        for method, action in matched_controls(primary_delta.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
            if method == 'RawMean_Exact':
                continue
            delta = torch.from_numpy(action).to(device)
            endpoints[key][method] = baseline+operator @ delta
            frozen[key+'__'+method+'__delta'] = action
            diagnostics[key]['allocation_controls'][method] = {'budget_max_error': float((delta.sum(0)-primary_delta.double().sum(0)).abs().max()),
                'spectrum_max_error': float((delta[valid].sort(0).values-primary_delta.double()[valid].sort(0).values).abs().max()),
                'capacity_excess_max': float((-delta-cap.double()).clamp_min(0).max())}
        for method, scores in endpoints[key].items():
            frozen[key+'__'+method+'__scores'] = scores.cpu().numpy()
        if extension is not None:
            extra, fields, stats = extension(bank=bank, crops=crops['clean', key], count=count,
                coordinates=coordinates, image_size=tuple(image.shape[-2:]), members=members,
                canonical=canonical, valid=valid, cap=cap, operator=operator, local=g, broad=b,
                baseline=baseline, risk=risk['reversal'], full=full[0], kept=kept, removed=removed,
                text=text, source=source, protocol=key, supported=supported)
            if set(extra) & set(endpoints[key]) or any(key+'__'+name in frozen for name in fields):
                raise RuntimeError('Extended reader overwrote historical fields.')
            endpoints[key].update(extra)
            frozen.update({key+'__'+name: value for name, value in fields.items()})
            diagnostics[key]['rival_reader'] = stats
        torch.cuda.synchronize()
        diagnostics[key]['reader_shared_seconds'] = time.perf_counter()-begun
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask source or action.')
    return endpoints, frozen, diagnostics


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION, config=CONFIG)
