"""Frozen canonical-witness increments on the existing nested-count windows."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geo_alias_increment import (CONFIG, COUNTS, IMPLEMENTATION, METHODS, NEW, PRIMARY,
    anchored_veto, fixed_profile, increment_predictions, sample_raw, text_discrimination,
    visual_discrimination, witness_fields)
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_count import canonical_indices
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_fine_reference_admission import fine_observations
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args, SETTINGS
from eval_geometry_vip_reliability import sample_broad, summary
from eval_native_alias_noise import ORIGINAL
from eval_rival_alias_count import VOCAB
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/rival_alias_count_203040_20261003'
PAIRS = {'vs_fixed20': ('Fixed20_Exact', PRIMARY), 'vs_fixed20hard': ('Fixed20Hard_Exact', PRIMARY),
         'vs_same_count_hard': ('RivalFineHard_Exact', PRIMARY),
         'vs_increment_all': ('IncrementAll_Exact', PRIMARY)}


def members_for(query):
    return torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])


def copy_prefix(crops, original, full_members, base_members):
    for full, base in zip(crops, original):
        full.alias_logits[:, full_members[:, :20]] = base.alias_logits[:, base_members]
        full.salience[full_members[:, :20]] = base.salience[base_members]


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    device = geometry.device
    coordinates, valid, operator = [torch.from_numpy(original[k]).to(device) for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    error = float((actual_operator-operator).abs().max())
    if error > 1e-10:
        raise RuntimeError('Original Geometry reconstruction changed.')
    broad, wide, _, count = prepare_wide(image, vip, variants)
    queries = {s+'__'+p: q for s, (_, qs) in variants.items() for p, q in qs.items()}
    fine, fine_count, frozen, costs = fine_observations(image, vip, queries, coordinates, valid)
    output = {'k'+str(k): {} for k in COUNTS}
    diagnostics = {'operator_replay_max_error': error, 'local_replay_max_error': {},
        'base_wide_replay_max_error': {}, **costs, 'scenarios': {s: {} for s in output}}
    for p, bank in banks.items():
        base_query, query = variants['k20'][1][p], variants['k40'][1][p]
        members, base_members = members_for(query), members_for(base_query)
        canonical = canonical_indices(query.class_names, query.aliases, query.parents)
        canonical_local = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
            for c in range(bank.class_count)])
        actual_local = alias_class_scores((prepared.geometry_projected.float() @
            F.normalize(bank.features.float(), dim=-1).T)[0], bank.parent_indices, bank.class_count)/.07
        local = torch.from_numpy(original[p+'__local']).to(device)
        local_error = float((actual_local-local).abs().max())
        baseline = torch.from_numpy(previous['k20__'+p+'__NoAdmission_Exact__scores']).to(device)
        b = sample_broad(broad['k20', p], 0, 0, *image.shape[-2:]).reshape_as(local)
        base_error = float((baseline-(local.double()+operator @ (b.double()-local.double()))).abs().max())
        if max(local_error, base_error) > 1e-4:
            raise RuntimeError('Fixed local20/wide20 anchor drifted.')
        diagnostics['local_replay_max_error'][p] = local_error
        diagnostics['base_wide_replay_max_error'][p] = base_error
        copy_prefix(wide['k40', p], wide['k20', p], members, base_members)
        copy_prefix(fine['k40__'+p], fine['k20__'+p], members, base_members)
        fine_raw = sample_raw(fine['k40__'+p], fine_count, coordinates, (512, 512))/SETTINGS.logit_scale
        local_canonical = (prepared.geometry_projected.float() @
            F.normalize(bank.features[canonical_local].float(), dim=-1).T)[0]
        means, mass, quality = witness_fields(prepared.geometry_patch_conditional[0], local_canonical,
            fine_raw, canonical, valid)
        visual, known = visual_discrimination(means, mass, members, canonical, valid)
        text = text_discrimination(query.features.float().mean(1), members, canonical)
        veto = anchored_veto(wide['k40', p], count, fine['k40__'+p], fine_count,
            coordinates, tuple(image.shape[-2:]), members, canonical, valid)
        frozen[p+'__canonical_witness_quality'] = quality.cpu().numpy()
        frozen[p+'__canonical_witness_mass'] = mass.cpu().numpy()
        for k in COUNTS:
            scene = 'k'+str(k)
            chosen = members[:, :k]
            new, cached, details = increment_predictions(baseline, operator, wide['k40', p], count, coordinates,
                tuple(image.shape[-2:]), chosen, canonical, text[:, :k], visual[:, :, :k], veto[:, :, :k], known, valid)
            values = {'Geometry': torch.from_numpy(previous[scene+'__'+p+'__Geometry__scores']).to(device),
                'AllCount_Exact': torch.from_numpy(previous[scene+'__'+p+'__NoAdmission_Exact__scores']).to(device),
                'RivalFineHard_Exact': torch.from_numpy(previous[scene+'__'+p+'__RivalFineHard_Exact__scores']).to(device),
                'Fixed20_Exact': baseline,
                'Fixed20Hard_Exact': torch.from_numpy(previous['k20__'+p+'__RivalFineHard_Exact__scores']).to(device), **new}
            if k == 20:
                identity = float((values['IncrementAll_Exact']-baseline).abs().max())
                if identity != 0:
                    raise RuntimeError('Uniform20 increment did not reproduce the fixed anchor.')
                details['uniform20_increment_identity_max_error'] = identity
            prefix = scene+'__'+p+'__'
            for method, value in values.items():
                frozen[prefix+method+'__scores'] = value.cpu().numpy()
            for name, value in cached.items():
                frozen[prefix+name] = value
            output[scene][p] = values
            diagnostics['scenarios'][scene][p] = details
    if not all(np.isfinite(v).all() for v in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask fields.')
    return output, frozen, diagnostics


def main(args, smoke=False):
    output = Path(args.output_dir)
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    source = json.loads(VOCAB.read_text())
    if output.exists() or not prior['coverage_verified'] or source['status'] != 'complete':
        raise RuntimeError('New output and verified count source required.')
    keys = prior['sample_keys'][:1] if smoke else prior['sample_keys']
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['base_signature']['checkpoints']:
        raise RuntimeError('Frozen checkpoints changed.')
    context = source['datasets'][args.dataset]
    expanded = {p: vip.encode_queries(bank.class_names, tuple(tuple(c['synonyms']) for c in context[p]['40']))
        for p, bank in banks.items()}
    for p, query in expanded.items():
        full_members, base_members = members_for(query), members_for(queries[p])
        if (list(query.aliases) != [a for c in context[p]['40'] for a in c['synonyms']]
                or context[p] != prior['context_vocabularies'][p]):
            raise RuntimeError('Frozen nested source changed.')
        if not torch.equal(query.features[full_members[:, :20]], queries[p].features[base_members]):
            raise RuntimeError('Per-phrase encoded historical20 features changed.')
    variants = {'k20': (banks, queries), 'k40': (banks, expanded)}
    states = {prefix+n: v.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
        for n, v in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    save(output/'vocabularies.json', context)
    matrices = {'k'+str(k): {p: {m: [] for m in METHODS} for p in banks} for k in COUNTS}
    transitions = {s: {p: {pair: [] for pair in PAIRS} for p in banks} for s in matrices}
    diagnostics, ignored = {}, dict.fromkeys(banks, 0)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, key in enumerate(keys):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                PREVIOUS/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as previous:
            scores, frozen, diagnostics[key] = predict(image, geometry, banks, vip, variants, original, previous)
        np.savez_compressed(output/'numerical_cache'/f'{index}.npz', **frozen)
        if not smoke:
            for p, bank in banks.items():
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, dtype=np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                ignored[p] += int(((target < 0) | (target >= bank.class_count)).sum())
                for scene in matrices:
                    predictions = {m: dense(v).argmax(-1).cpu().numpy() for m, v in scores[scene][p].items()}
                    for m, pred in predictions.items():
                        cm = confusion_batch(torch.from_numpy(pred).to(args.device), torch.from_numpy(target).to(args.device), bank.class_count)[0].cpu().numpy()
                        matrices[scene][p][m].append(cm)
                    for pair, (before, after) in PAIRS.items():
                        transitions[scene][p][pair].append(transition_counts(predictions[before], predictions[after], target, bank.class_count))
        result = {'status': 'complete' if index+1 == len(keys) else 'running', 'implementation': IMPLEMENTATION,
            'config': asdict(CONFIG), 'processed_images': index+1, 'total_images': len(keys), 'sample_keys': keys,
            'base_signature': prior['base_signature'], 'vocabulary_source_sha256': prior['vocabulary_source_sha256'],
            'context_alias_counts': list(COUNTS), 'local_geometry_alias_count': 20, 'context_vocabularies': context,
            'window_pilot_only': True, 'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True,
            'diagnostics': diagnostics, 'metrics': {s: {p: {m: summary(np.stack(v).sum(0), banks[p].class_names, ignored[p])
                for m, v in ms.items()} for p, ms in ps.items()} for s, ps in matrices.items()} if not smoke else {},
            'transitions': {s: {p: {pair: {'counts': np.stack(v).sum(0).tolist(), **transition_summary(np.stack(v).sum(0))}
                for pair, v in pairs.items()} for p, pairs in ps.items()} for s, ps in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(keys)}), flush=True)
    if not smoke:
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
            **{s+'__'+p+'__'+m: np.stack(v) for s, ps in matrices.items() for p, arms in ps.items() for m, v in arms.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(keys),
            **{s+'__'+p+'__'+pair: np.stack(v) for s, ps in transitions.items() for p, pairs in ps.items() for pair, v in pairs.items()})
    result['weights_frozen'] = all(not v.requires_grad for model in (geometry.backbone, vip.backbone) for v in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(states[prefix+n], v) for prefix, model in
        (('g_', geometry.backbone), ('v_', vip.backbone)) for n, v in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Frozen weights changed.')
    save(output/'results.json', result)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke)
