"""Fixed-window pair-conditioned reader with reused frozen intervention observations."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_spatial_allocation import matched_controls
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.pair_context_reader import (IMPLEMENTATION, METHODS, PRIMARY, PairContextConfig,
    pair_risk, pair_observation, reconcile)
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.target_context_alias import TargetContextConfig, observation_delta
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args, SETTINGS
from eval_geometry_vip_reliability import sample_broad, summary
from eval_stratified_soft_alias import tile_coordinates
from run_region_semantic_suite_a800 import TOOL


CONFIG, SOURCE_CONFIG = PairContextConfig(), TargetContextConfig()
ORIGINAL = TOOL/'results/alias_action_capacity_audit_20261003'
SOURCE = TOOL/'results/target_context_alias_source_r2_20261003'
RETAINED = ('Geometry', 'Anchored_Exact', 'TargetContext_Exact', 'MeanLogit_Original')


@torch.inference_mode()
def predict(image, geometry, banks, vip, queries, original, source):
    device = geometry.device
    coordinates = tile_coordinates(0, 0, device)
    valid = torch.from_numpy(original['valid']).to(device)
    operator = torch.from_numpy(original['operator']).to(device)
    started = time.perf_counter()
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Changed Geometry reconstruction operator.')
    broad, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    torch.cuda.synchronize()
    base_seconds = time.perf_counter()-started
    supported = torch.from_numpy(source['supported_units']).to(device)[torch.from_numpy(source['assignments']).to(device)] & valid
    endpoints, frozen, diagnostics = {}, {}, {'base_replay_seconds': base_seconds, 'operator_replay_max_error': operator_error,
        'unmasked_wide_forwards': len(next(iter(crops.values()))), 'new_intervention_forwards': 0}
    for key, bank in banks.items():
        begun = time.perf_counter()
        aliases = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
        local = alias_class_scores(aliases, bank.parent_indices, bank.class_count)/.07
        stored_local = torch.from_numpy(original[key+'__local']).to(device)
        b = torch.from_numpy(original['clean__'+key+'__broad']).to(device)
        actual_b = sample_broad(broad['clean', key], 0, 0, *image.shape[-2:]).reshape_as(local)
        errors = {'local': float((local-stored_local).abs().max()), 'broad': float((actual_b-b).abs().max())}
        if max(errors.values()) > 1e-4:
            raise RuntimeError('Changed original readout: '+json.dumps(errors))
        members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0] for c in range(bank.class_count)])
        full, kept, removed = [torch.from_numpy(source[key+'__source_'+name]).to(device) for name in ('full', 'kept', 'removed')]
        risk = pair_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, SOURCE_CONFIG)
        rho = 1-risk.amax(-1)
        old_rho = torch.from_numpy(source[key+'__TargetContext_Exact__retention']).to(device)
        if not torch.equal(rho, old_rho):
            raise RuntimeError('Collapsed rival source did not exactly recover prior retention.')
        delta_old = observation_delta(crops['clean', key], count, coordinates, tuple(image.shape[-2:]), members, rho, valid, SETTINGS.tau)
        old_delta = 2*(torch.from_numpy(source[key+'__MeanLogit_TargetContext__scores']).to(device)
            -torch.from_numpy(source[key+'__MeanLogit_Original__scores']).to(device))
        errors['independent_action'] = float((delta_old.double()-old_delta).abs().max())
        baseline = torch.from_numpy(source[key+'__Anchored_Exact__scores']).to(device)
        replay_primary = baseline+operator @ delta_old.double()
        prior_primary = torch.from_numpy(source[key+'__TargetContext_Exact__scores']).to(device)
        errors['independent_endpoint'] = float((replay_primary-prior_primary).abs().max())
        if (max(errors.values()) > 1e-4 or not torch.equal(dense(replay_primary).argmax(-1), dense(prior_primary).argmax(-1))):
            raise RuntimeError('Prior independent reader failed matched replay: '+json.dumps(errors))
        cap = torch.from_numpy(original['clean__'+key+'__protected_cap']).to(device)
        text = semantic_contradictions(bank.features, bank.parent_indices, canonical, SOURCE_CONFIG.epsilon)
        sources = {PRIMARY: risk,
            'ShuffledSupport_PairExact': pair_risk(full[0], *[torch.from_numpy(source[key+'__source_shuffle_'+name]).to(device)
                for name in ('kept', 'removed')], bank.parent_indices, canonical, supported, SOURCE_CONFIG),
            'SingleFill0_PairExact': pair_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, SOURCE_CONFIG, 0),
            'SingleFill1_PairExact': pair_risk(full[0], kept, removed, bank.parent_indices, canonical, supported, SOURCE_CONFIG, 1),
            'TextOnly_PairExact': text[None].expand(len(valid), -1, -1).masked_fill(~valid[:, None, None], 0)}
        endpoints[key] = {m: torch.from_numpy(source[key+'__'+m+'__scores']).to(device) for m in RETAINED}
        diagnostics[key] = {'replay_max_errors': errors, 'canonical_max_risk': float(risk[:, canonical].abs().max()),
            'solver': {}, 'allocation_controls': {}}
        primary_delta = None
        for method, gamma in sources.items():
            margins, weights, directed = pair_observation(crops['clean', key], count, coordinates,
                tuple(image.shape[-2:]), members, gamma, valid, SETTINGS.tau, CONFIG.query_chunk)
            delta, stats = reconcile(margins, weights, cap, valid, CONFIG)
            if bool((delta > 0).any()) or bool((delta < -cap.double()).any()) or bool((delta[~valid] != 0).any()):
                raise RuntimeError('Joint suppression bounds failed.')
            endpoints[key][method] = baseline+operator @ delta
            diagnostics[key]['solver'][method] = stats
            frozen[key+'__'+method+'__delta'] = delta.cpu().numpy()
            frozen[key+'__'+method+'__margins'] = margins.cpu().numpy()
            frozen[key+'__'+method+'__weights'] = weights.cpu().numpy()
            if method == PRIMARY:
                primary_delta = delta
                frozen[key+'__directed_alias_corrections'] = directed.cpu().numpy()
                frozen[key+'__pair_alias_risk'] = gamma.cpu().numpy()
        endpoints[key]['PairContext_MeanLogit'] = .5*(stored_local.double()+b.double()+primary_delta)
        for method, action in matched_controls(primary_delta.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
            if method == 'RawMean_Exact':
                continue
            delta = torch.from_numpy(action).to(device)
            endpoints[key][method] = baseline+operator @ delta
            frozen[key+'__'+method+'__delta'] = action
            diagnostics[key]['allocation_controls'][method] = {
                'budget_max_error': float((delta.sum(0)-primary_delta.sum(0)).abs().max()),
                'spectrum_max_error': float((delta[valid].sort(0).values-primary_delta[valid].sort(0).values).abs().max()),
                'capacity_excess_max': float((-delta-cap.double()).clamp_min(0).max()),
                'changed_donor_class_fraction': float(((delta-primary_delta).abs()[valid] > 1e-12).double().mean())}
        for method, value in endpoints[key].items():
            frozen[key+'__'+method+'__scores'] = value.cpu().numpy()
        torch.cuda.synchronize()
        diagnostics[key]['reader_shared_seconds'] = time.perf_counter()-begun
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask field.')
    return endpoints, frozen, diagnostics


def main(args, smoke=False, *, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION, config=CONFIG):
    output = Path(args.output_dir)
    prior = json.loads((SOURCE/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if output.exists() or len(keys) != 8 or len(set(keys)) != 8 or not prior['coverage_verified']:
        raise ValueError('New output and verified8-window source required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise ValueError('Changed checkpoints.')
    actual_vocab = {'sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        'aliases': {key: list(bank.alias_names) for key, bank in banks.items()},
        'counts': {key: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)] for key, bank in banks.items()}}
    if actual_vocab != prior['signature']['vocabulary'] or {k: list(b.class_names) for k, b in banks.items()} != prior['signature']['classes']:
        raise ValueError('Changed actual vocabulary/groups/classes.')
    states = {prefix+key: value.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
              for key, value in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    selected = keys[:1] if smoke else keys
    matrices = {p: {m: [] for m in methods} for p in banks}
    transitions = {p: {m: [] for m in methods if m != 'Anchored_Exact'} for p in banks}
    ignored, diagnostics, source_scores = dict.fromkeys(banks, 0), {}, {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, key in enumerate(selected):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                SOURCE/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as source:
            endpoints, frozen, diagnostics[key] = predictor(image, geometry, banks, vip, queries, original, source)
        np.savez_compressed(output/'numerical_cache'/f'{index}.npz', **frozen)
        for name, value in frozen.items():
            if name.endswith('__action_score'):
                source_scores.setdefault(name, []).append(value)
        if not smoke:
            for p, bank in banks.items():
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, dtype=np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                ignored[p] += int(((target < 0) | (target >= bank.class_count)).sum())
                predictions = {m: dense(value).argmax(-1).cpu().numpy() for m, value in endpoints[p].items()}
                for method, pred in predictions.items():
                    cm = confusion_batch(torch.from_numpy(pred).to(args.device), torch.from_numpy(target).to(args.device), bank.class_count)[0].cpu().numpy()
                    matrices[p][method].append(cm)
                    if method != 'Anchored_Exact':
                        transitions[p][method].append(transition_counts(predictions['Anchored_Exact'], pred, target, bank.class_count))
        result = {'status': 'complete' if index+1 == len(selected) else 'running', 'implementation': implementation,
            'config': asdict(config), 'source_config': asdict(SOURCE_CONFIG), 'window_pilot_only': True,
            'processed_images': index+1, 'total_images': len(selected), 'sample_keys': selected, 'signature': prior['signature'],
            'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True, 'diagnostics': diagnostics,
            'metrics': {p: {m: summary(np.stack(values).sum(0), banks[p].class_names, ignored[p])
                for m, values in group.items()} for p, group in matrices.items()} if not smoke else {},
            'transitions': {p: {m: {'counts': np.stack(values).sum(0).tolist(), **transition_summary(np.stack(values).sum(0))}
                for m, values in group.items()} for p, group in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(selected)}), flush=True)
    if not smoke:
        if source_scores:
            np.savez_compressed(output/'per_image_source_scores.npz', sample_keys=np.asarray(selected),
                **{name: np.stack(values) for name, values in source_scores.items()})
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(selected),
            **{p+'__'+m: np.stack(values) for p, group in matrices.items() for m, values in group.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(selected),
            **{p+'__'+m: np.stack(values) for p, group in transitions.items() for m, values in group.items()})
    result['weights_frozen'] = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(states[prefix+key], value) for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
        for key, value in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Frozen weights changed.')
    save(output/'results.json', result)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke)
