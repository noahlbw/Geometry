"""Replay frozen endpoints with budget-matched spatial suppression controls."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dinotool.alias_spatial_allocation import (IMPLEMENTATION, METHODS, SEEDS,
    ROUNDING_GUARD, matched_controls, validated_magnitudes)
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import summary
from run_region_semantic_suite_a800 import TOOL


ORIGINAL = TOOL/'results/alias_action_capacity_audit_20261003'
SOURCE = TOOL/'results/target_context_alias_source_r2_20261003'


@torch.inference_mode()
def replay(original, source, keys, device):
    operator = torch.from_numpy(original['operator']).to(device)
    valid = original['valid']
    endpoints, frozen, diagnostics = {}, {}, {}
    for key in keys:
        baseline = original['clean__'+key+'__baseline_exact']
        primary = source[key+'__TargetContext_Exact__scores']
        delta = 2*(source[key+'__MeanLogit_TargetContext__scores']
                   -source[key+'__MeanLogit_Original__scores'])
        cap = original['clean__'+key+'__protected_cap']
        magnitude, effective = validated_magnitudes(delta, cap, valid)
        scores = torch.from_numpy(baseline).to(device)+operator @ torch.from_numpy(delta).to(device)
        error = float((scores-torch.from_numpy(primary).to(device)).abs().max())
        changed = int((dense(scores).argmax(-1) != dense(torch.from_numpy(primary).to(device)).argmax(-1)).sum())
        if error > 1e-10 or changed:
            raise RuntimeError('Primary replay failed: '+key)
        endpoints[key] = {method: torch.from_numpy(source[key+'__'+method+'__scores']).to(device)
                          for method in ('Geometry', 'Anchored_Exact', 'TargetContext_Exact')}
        if (not np.array_equal(source[key+'__Geometry__scores'], original[key+'__local'])
                or not np.array_equal(source[key+'__Anchored_Exact__scores'], baseline)):
            raise RuntimeError('Retained endpoints changed.')
        diagnostics[key] = {'primary_replay_max_error': error, 'primary_replay_changed_dense_pixels': changed,
            'capacity_rounding_enlargement_max': float((effective-cap).max()),
            'class_suppression_budget': magnitude.sum(0).tolist(), 'controls': {}}
        for method, action in matched_controls(delta, cap, valid).items():
            budget_error = float(np.abs(action.sum(0)-delta.sum(0)).max())
            capacity_excess = np.maximum(-action-effective, 0)
            spectrum_error = float(np.abs(np.sort(action[valid], axis=0)-np.sort(delta[valid], axis=0)).max())
            if (budget_error > 1e-10 or np.abs(action[~valid]).max(initial=0) > 1e-12
                    or (method != 'RawMean_Exact' and capacity_excess.max(initial=0) > 1e-12)
                    or (method.startswith('SpatialShuffle') and spectrum_error > 1e-12)):
                raise RuntimeError('Matched action invariant failed: '+method)
            endpoints[key][method] = torch.from_numpy(baseline).to(device)+operator @ torch.from_numpy(action).to(device)
            diagnostics[key]['controls'][method] = {'budget_max_error': budget_error,
                'spectrum_max_error': spectrum_error, 'capacity_excess_max': float(capacity_excess.max(initial=0)),
                'capacity_violating_donor_classes': int((capacity_excess > 1e-12).sum()),
                'changed_donor_class_fraction': float((np.abs(action[valid]-delta[valid]) > 1e-12).mean())}
            frozen[key+'__'+method+'__delta'] = action
        frozen[key+'__primary_delta'] = delta
        for method, value in endpoints[key].items():
            frozen[key+'__'+method+'__scores'] = value.cpu().numpy()
    return endpoints, frozen, diagnostics


def main(args, smoke=False):
    output = Path(args.output_dir)
    prior = json.loads((SOURCE/args.dataset/'merged.json').read_text())
    original = json.loads((ORIGINAL/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if (output.exists() or len(keys) != 8 or len(set(keys)) != 8
            or prior['signature'] != original['signature'] or not prior['coverage_verified']):
        raise ValueError('New output and matched verified8-window caches required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    vocabulary = prior['signature']['vocabulary']
    if hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != vocabulary['sha256']:
        raise ValueError('Public vocabulary bytes changed.')
    for key, group in specs.items():
        aliases = [word for spec in group for word in spec.synonyms]
        if ([spec.name for spec in group] != prior['signature']['classes'][key]
                or aliases != vocabulary['aliases'][key] or [len(spec.synonyms) for spec in group] != vocabulary['counts'][key]
                or any(len(spec.synonyms) != 20 for spec in group)):
            raise ValueError('Changed actual classes or20-alias groups: '+key)
    lookup = {sample.key: sample for sample in samples}
    selected = keys[:1] if smoke else keys
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    matrices = {key: {method: [] for method in METHODS} for key in specs}
    transitions = {key: {method: [] for method in METHODS if method != 'Anchored_Exact'} for key in specs}
    ignored, diagnostics = dict.fromkeys(specs, 0), {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, key in enumerate(selected):
        with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as cached, np.load(
                SOURCE/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as source:
            endpoints, frozen, diagnostics[key] = replay(cached, source, specs, args.device)
        np.savez_compressed(output/'numerical_cache'/f'{index}.npz', **frozen)
        if not smoke:
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            for protocol_key, group in specs.items():
                full = load_mask(sample, protocol_key, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, dtype=np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                ignored[protocol_key] += int(((target < 0) | (target >= len(group))).sum())
                predictions = {method: dense(scores).argmax(-1).cpu().numpy()
                               for method, scores in endpoints[protocol_key].items()}
                for method, pred in predictions.items():
                    cm = confusion_batch(torch.from_numpy(pred).to(args.device), torch.from_numpy(target).to(args.device), len(group))[0].cpu().numpy()
                    matrices[protocol_key][method].append(cm)
                    if method != 'Anchored_Exact':
                        transitions[protocol_key][method].append(transition_counts(predictions['Anchored_Exact'], pred, target, len(group)))
        result = {'status': 'complete' if index+1 == len(selected) else 'running', 'implementation': IMPLEMENTATION,
            'diagnostic_only': True, 'processed_images': index+1, 'total_images': len(selected), 'sample_keys': selected,
            'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True, 'encoders_loaded': False,
            'new_encoder_forwards': 0, 'signature': prior['signature'], 'seeds': SEEDS, 'rounding_guard': ROUNDING_GUARD,
            'diagnostics': diagnostics, 'metrics': {p: {m: summary(np.stack(values).sum(0), tuple(s.name for s in specs[p]), ignored[p])
                for m, values in group.items()} for p, group in matrices.items()} if not smoke else {},
            'transitions': {p: {m: {'counts': np.stack(values).sum(0).tolist(), **transition_summary(np.stack(values).sum(0))}
                for m, values in group.items()} for p, group in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(selected)}), flush=True)
    if not smoke:
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(selected),
            **{p+'__'+m: np.stack(values) for p, group in matrices.items() for m, values in group.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(selected),
            **{p+'__'+m: np.stack(values) for p, group in transitions.items() for m, values in group.items()})


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), is_smoke)
