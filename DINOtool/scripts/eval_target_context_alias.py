"""Frozen target/context source pilot on the prior64 fixed audit windows."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.stratified_soft_alias import crop_stencil
from dinotool.target_context_alias import (IMPLEMENTATION, METHODS, PRIMARY, SHUFFLED, TargetContextConfig,
    alias_permutations, crop_masks, geometry_masks, intervention_retention, observation_delta, shuffled_crop_masks)
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_stratified_soft_alias import tile_coordinates


CONFIG = TargetContextConfig()
SOURCE = Path('/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/alias_action_capacity_audit_20261003')


@torch.inference_mode()
def observations(image, vip, queries, crops, count, coordinates, valid, masks, assignments, supported):
    resized = resize_rgb(image, 448).to(vip.device)
    measurements = {key: {name: torch.zeros((2 if name != 'full' else 1, len(valid), len(query.aliases)), device=vip.device)
                          for name in ('full', 'kept', 'removed', 'shuffle_kept', 'shuffle_removed')}
                    for key, query in queries.items()}
    first = next(iter(crops.values()))
    means = {key: query.features.float().mean(1) for key, query in queries.items()}
    forwards, seconds = {'unmasked': 0, 'true': 0, 'shuffle': 0}, {'unmasked': 0., 'true': 0., 'shuffle': 0.}
    spectrum_error = 0.
    for number, crop in enumerate(first):
        rgb = F.pad(resized[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width],
                    (0, 336-crop.actual_width, 0, 336-crop.actual_height))
        indices, coefficients = crop_stencil(crop, count, coordinates, tuple(image.shape[-2:]))
        active = valid & (coefficients.sum(-1) > 0)
        if not bool(active.any()):
            continue
        torch.cuda.synchronize()
        started = time.perf_counter()
        features = vip.crop_patch_features(rgb)
        raw = {key: (features[0].float() @ text.T) for key, text in means.items()}
        torch.cuda.synchronize()
        seconds['unmasked'] += time.perf_counter()-started
        forwards['unmasked'] += 1
        for key in queries:
            measurements[key]['full'][0] += (raw[key][indices]*coefficients[..., None]).sum(1)
        mapped = crop_masks(masks, crop, count.shape, tuple(image.shape[-2:]))
        shuffled = shuffled_crop_masks(mapped, crop.actual_height, crop.actual_width, CONFIG.random_seed+number)
        spectrum_error = max(spectrum_error, float((mapped[:, :crop.actual_height, :crop.actual_width].flatten(1).sort(-1).values
                            -shuffled[:, :crop.actual_height, :crop.actual_width].flatten(1).sort(-1).values).abs().max()))
        visible = torch.zeros((336, 336), device=vip.device, dtype=torch.bool)
        visible[:crop.actual_height, :crop.actual_width] = True
        fills = (rgb.new_tensor([.485, .456, .406])[:, None, None], rgb[:, visible].mean(-1)[:, None, None])
        for cell in assignments[active].unique().tolist():
            use = active & (assignments == cell)
            for kind, support in (('true', mapped[cell]), ('shuffle', shuffled[cell])):
                for fill_index, fill in enumerate(fills):
                    for suffix, m in (('kept', support), ('removed', 1-support)):
                        name = suffix if kind == 'true' else 'shuffle_'+suffix
                        if not bool(supported[cell]):
                            for key in queries:
                                measurements[key][name][fill_index, use] += (raw[key][indices[use]]*coefficients[use, :, None]).sum(1)
                            continue
                        changed = (m[None]*rgb+(1-m[None])*fill).masked_fill(~visible[None], 0.)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        features = vip.crop_patch_features(changed)
                        for key, text in means.items():
                            field = features[0].float() @ text.T
                            measurements[key][name][fill_index, use] += (field[indices[use]]*coefficients[use, :, None]).sum(1)
                        torch.cuda.synchronize()
                        seconds[kind] += time.perf_counter()-started
                        forwards[kind] += 1
    return measurements, {'source_forwards': forwards, 'source_seconds': seconds, 'support_shuffle_spectrum_error': spectrum_error}


@torch.inference_mode()
def predict(image, geometry, banks, vip, queries, cache):
    device = geometry.device
    coords = tile_coordinates(0, 0, device)
    valid = (coords[:, 0] < image.shape[-2]) & (coords[:, 1] < image.shape[-1])
    started = time.perf_counter()
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    relation = prepared.geometry_patch_conditional[0]
    masks, assignment, supported = geometry_masks(relation, coords, valid, CONFIG)
    operator, _ = reconstruction_operator(relation, valid)
    error = float((operator-torch.from_numpy(cache['operator']).to(device)).abs().max())
    if error > 1e-10 or not np.array_equal(valid.cpu().numpy(), cache['valid']):
        raise RuntimeError('Changed exact-system relation or validity.')
    broad, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    replay = {'operator': error}
    for key, bank in banks.items():
        aliases = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
        local = alias_class_scores(aliases, bank.parent_indices, bank.class_count)/.07
        b = sample_broad(broad['clean', key], 0, 0, *image.shape[-2:]).reshape_as(local)
        for name, value, stored in (('local', local, key+'__local'), ('broad', b, 'clean__'+key+'__broad')):
            replay[key+'_'+name] = float((value.double()-torch.from_numpy(cache[stored]).to(device)).abs().max())
    if any(value > 1e-4 for value in replay.values()):
        raise RuntimeError('Original window control changed: '+json.dumps(replay))
    torch.cuda.synchronize()
    base_seconds = time.perf_counter()-started
    cropped = {key: crops['clean', key] for key in banks}
    measurement, costs = observations(image, vip, queries, cropped, count, coords, valid, masks, assignment, supported)
    endpoints, frozen, diagnostics = {}, {'support_masks': masks.cpu().numpy(), 'assignments': assignment.cpu().numpy(),
        'supported_units': supported.cpu().numpy()}, {'replay_max_errors': replay, 'base_shared_seconds': base_seconds, **costs}
    for key, bank in banks.items():
        prefix = 'clean__'+key+'__'
        original = torch.from_numpy(cache[prefix+'baseline_exact']).to(device)
        g, b = torch.from_numpy(cache[key+'__local']).to(device), torch.from_numpy(cache[prefix+'broad']).to(device)
        members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0] for c in range(bank.class_count)])
        row = measurement[key]
        query_known = supported[assignment] & valid
        rho, context = intervention_retention(row['full'][0], row['kept'], row['removed'], bank.parent_indices, canonical, query_known, CONFIG)
        shuffle, _ = intervention_retention(row['full'][0], row['shuffle_kept'], row['shuffle_removed'], bank.parent_indices, canonical, query_known, CONFIG)
        allocations = {PRIMARY: rho, 'ContextOnly_Exact': context, 'ShuffledSupport_Exact': shuffle}
        for name, permutation in zip(SHUFFLED, alias_permutations(members, canonical, CONFIG.random_seed)):
            allocations[name] = torch.empty_like(rho)
            allocations[name][:, members] = rho[:, members].gather(-1, permutation[None].expand(len(rho), -1, -1))
        variants = {'Geometry': g, 'Anchored_Exact': original,
                    'ClassRelativeReject_CG': torch.from_numpy(cache[prefix+'source_cg']).to(device),
                    'MeanLogit_Original': .5*(g.double()+b.double())}
        primary_delta = None
        capacity = torch.zeros_like(rho)
        capacity[:, members] = torch.from_numpy(cache[prefix+'single_cap']).to(device)
        identity = observation_delta(cropped[key], count, coords, tuple(image.shape[-2:]), members, torch.ones_like(rho), valid, SETTINGS.tau)
        if bool((identity != 0).any()):
            raise RuntimeError('Identity retention failed original writer recovery.')
        for name, retention in allocations.items():
            delta = observation_delta(cropped[key], count, coords, tuple(image.shape[-2:]), members, retention, valid, SETTINGS.tau)
            if bool((delta > 0).any()) or bool(((retention < 0) | (retention > 1)).any()):
                raise RuntimeError('Bounded suppression invariant failed.')
            variants[name] = original+operator @ delta.double()
            if name == PRIMARY:
                primary_delta = delta
            frozen[key+'__'+name+'__retention'] = retention.cpu().numpy()
            frozen[key+'__'+name+'__action_score'] = (((1-retention)*capacity).sum(0)/capacity.sum(0).clamp_min(1e-12)).cpu().numpy()
        variants['MeanLogit_TargetContext'] = .5*(g.double()+b.double()+primary_delta.double())
        endpoints[key] = variants
        for name, value in row.items():
            frozen[key+'__source_'+name] = value.cpu().numpy()
        for name, value in variants.items():
            frozen[key+'__'+name+'__scores'] = value.cpu().numpy()
        grouped = rho[:, members]
        shuffle_error = max(float((allocations[name][:, members].sort(-1).values-grouped.sort(-1).values).abs().max()) for name in SHUFFLED)
        diagnostics[key] = {'supported_query_fraction': float(query_known[valid].float().mean()),
            'mean_rejection': float((1-rho[valid]).mean()), 'positive_rejection_fraction': float(((1-rho[valid]) > 1e-6).float().mean()),
            'mean_score_suppression': float(primary_delta[valid].abs().mean()), 'alias_shuffle_spectrum_error': shuffle_error,
            'canonical_rejection': float((1-rho[:, canonical]).abs().max()), 'identity_writer_error': float(identity.abs().max())}
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask observation or endpoint.')
    return endpoints, frozen, diagnostics


def main(args, smoke=False):
    output = Path(args.output_dir)
    prior = json.loads((SOURCE/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if output.exists() or len(keys) != 8 or len(set(keys)) != 8:
        raise ValueError('New output and prior verified8-image sequence required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise ValueError('Changed checkpoints.')
    states = {prefix+key: value.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
              for key, value in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    matrices = {key: {method: [] for method in METHODS} for key in banks}
    transitions = {key: {method: [] for method in METHODS if method != 'Anchored_Exact'} for key in banks}
    diagnostics, ignored, source_scores = {}, dict.fromkeys(banks, 0), {}
    selected = keys[:1] if smoke else keys
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, key in enumerate(selected):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        with np.load(SOURCE/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as cache:
            endpoints, frozen, diagnostics[key] = predict(image, geometry, banks, vip, queries, cache)
        np.savez_compressed(output/'numerical_cache'/f'{index}.npz', **frozen)
        for name, value in frozen.items():
            if name.endswith('__action_score'):
                source_scores.setdefault(name, []).append(value)
        if not smoke:
            # The frozen source and all predictions are persisted before labels are opened.
            for protocol_key, bank in banks.items():
                full = load_mask(sample, protocol_key, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, dtype=np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                ignored[protocol_key] += int(((target < 0) | (target >= bank.class_count)).sum())
                predictions = {method: dense(scores).argmax(-1).cpu().numpy() for method, scores in endpoints[protocol_key].items()}
                for method, pred in predictions.items():
                    cm = confusion_batch(torch.from_numpy(pred).to(geometry.device), torch.from_numpy(target).to(geometry.device), bank.class_count)[0].cpu().numpy()
                    matrices[protocol_key][method].append(cm)
                    if method != 'Anchored_Exact':
                        transitions[protocol_key][method].append(transition_counts(predictions['Anchored_Exact'], pred, target, bank.class_count))
        result = {'status': 'complete' if index+1 == len(selected) else 'running', 'implementation': IMPLEMENTATION,
            'source_pilot_only': True, 'config': asdict(CONFIG), 'processed_images': index+1, 'total_images': len(selected),
            'sample_keys': selected, 'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True,
            'signature': prior['signature'], 'diagnostics': diagnostics, 'metrics': {protocol_key: {method: summary(np.stack(values).sum(0),
                banks[protocol_key].class_names, ignored[protocol_key]) for method, values in group.items()} for protocol_key, group in matrices.items()} if not smoke else {},
            'transitions': {protocol_key: {method: {'counts': np.stack(values).sum(0).tolist(), **transition_summary(np.stack(values).sum(0))}
                for method, values in group.items()} for protocol_key, group in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(selected)}), flush=True)
    if not smoke:
        np.savez_compressed(output/'per_image_source_scores.npz', sample_keys=np.asarray(selected),
                            **{name: np.stack(values) for name, values in source_scores.items()})
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(selected),
            **{key+'__'+method: np.stack(values) for key, group in matrices.items() for method, values in group.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(selected),
            **{key+'__'+method: np.stack(values) for key, group in transitions.items() for method, values in group.items()})
    result['weights_frozen'] = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(states[prefix+key], value) for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
        for key, value in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Frozen model changed.')
    save(output/'results.json', result)


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), is_smoke)
