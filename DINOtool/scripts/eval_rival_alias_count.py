"""Fresh observations for nested20/30/40 contextual aliases; Geometry20 stays fixed."""
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
from dinotool.fine_alias_view import CONFIG
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_count import IMPLEMENTATION, COUNTS, METHODS, canonical_indices, count_predictions
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_fine_reference_admission import fine_observations
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_native_alias_noise import ORIGINAL
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/rival_fine_coupling_20261003'
VOCAB = TOOL/'results/rival_alias_count_vocabulary_r2_20261003.json'
HISTORY = {'Geometry': 'Geometry', 'NoAdmission_Exact': 'NoAdmission_Exact',
    'RivalFineHard_Exact': 'RivalFineHard_Exact', 'ObserverAll': 'ObserverAll20',
    'ObserverHard': 'ObserverFinePairHard', 'CountMatchedRandom_Exact': 'RivalFineRandom0_Exact'}


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    coordinates, valid, operator = [torch.from_numpy(original[k]).to(geometry.device)
        for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Original Geometry relation/reconstruction changed.')
    broad, crops, _, count = prepare_wide(image, vip, variants)
    flat = {s+'__'+p: query for s, (_, queries) in variants.items() for p, query in queries.items()}
    fine, fine_count, frozen, costs = fine_observations(image, vip, flat, coordinates, valid)
    output, diag = {}, {'operator_replay_max_error': operator_error, **costs,
        'wide_forwards': len(next(iter(crops.values()))), 'new_intervention_forwards': 0,
        'local_replay_max_error': {}, 'scenarios': {}}
    local = {}
    for p, bank in banks.items():
        actual = alias_class_scores((prepared.geometry_projected.float() @
            F.normalize(bank.features.float(), dim=-1).T)[0], bank.parent_indices, bank.class_count)/.07
        local[p] = torch.from_numpy(original[p+'__local']).to(geometry.device)
        error = float((actual-local[p]).abs().max())
        diag['local_replay_max_error'][p] = error
        if error > 1e-4:
            raise RuntimeError('Original local20 Geometry scores changed.')
    for scene, (_, queries) in variants.items():
        output[scene], diag['scenarios'][scene] = {}, {}
        for p, query in queries.items():
            key = scene+'__'+p
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            canonical = canonical_indices(query.class_names, query.aliases, query.parents)
            b = sample_broad(broad[scene, p], 0, 0, *image.shape[-2:]).reshape_as(local[p])
            values, risk, details = count_predictions(local[p], operator, b, crops[scene, p], count,
                fine[key], fine_count, coordinates, valid, members, canonical, query.parents, tuple(image.shape[-2:]))
            # Historical20 endpoints require numerical replay before exact reuse.
            values['Geometry'] = torch.from_numpy(previous['clean__'+p+'__Geometry__scores']).to(geometry.device)
            if scene == 'k20':
                errors = {m: float((values[m]-torch.from_numpy(previous['clean__'+p+'__'+old+'__scores']).to(geometry.device)).abs().max())
                    for m, old in HISTORY.items()}
                if max(errors.values()) > 1e-4:
                    raise RuntimeError('Fresh historical20 endpoint drift: '+str(errors))
                details['historical20_replay_max_errors'] = errors
                for m, old in HISTORY.items():
                    values[m] = torch.from_numpy(previous['clean__'+p+'__'+old+'__scores']).to(geometry.device)
            for m, value in values.items():
                frozen[key+'__'+m+'__scores'] = value.cpu().numpy()
            keep = (risk == 0).cpu().numpy()
            frozen[key+'__retained_pair_shape'] = np.asarray(keep.shape)
            frozen[key+'__retained_pair_packed'] = np.packbits(keep)
            output[scene][p], diag['scenarios'][scene][p] = values, details
    return output, frozen, diag


def main(args, smoke=False):
    output = Path(args.output_dir)
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    source = json.loads(VOCAB.read_text())
    if (output.exists() or not prior['coverage_verified'] or source['status'] != 'complete'
            or source['target_images_loaded'] or source['target_masks_loaded'] or source['semantic_or_visual_filtering']):
        raise RuntimeError('New output, verified historical source and unfiltered text-only bank required.')
    keys = prior['sample_keys'][:1] if smoke else prior['sample_keys']
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Frozen checkpoints changed.')
    context = source['datasets'][args.dataset]
    variants = {'k20': (banks, queries)}
    for k in COUNTS[1:]:
        extended = {p: vip.encode_queries(bank.class_names,
            tuple(tuple(cls['synonyms']) for cls in context[p][str(k)])) for p, bank in banks.items()}
        variants['k'+str(k)] = (banks, extended)
    for p, bank in banks.items():
        if context[p]['20'] != prior['vocabularies']['clean']['vocabularies'][p]:
            raise RuntimeError('Historical20 vocabulary prefix changed.')
        for k in COUNTS:
            cls = context[p][str(k)]
            query = variants['k'+str(k)][1][p]
            if (list(query.class_names) != [c['name'] for c in cls]
                    or list(query.aliases) != [a for c in cls for a in c['synonyms']]
                    or any(int((query.parents == c).sum()) != k for c in range(bank.class_count))):
                raise RuntimeError('Expanded words were truncated, regrouped or reordered.')
    states = {prefix+n: v.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
        for n, v in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    save(output/'vocabularies.json', context)
    matrices = {s: {p: {m: [] for m in METHODS} for p in banks} for s in variants}
    transitions = {s: {p: [] for p in banks} for s in variants}
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
                for scene in variants:
                    predictions = {m: dense(value).argmax(-1).cpu().numpy() for m, value in scores[scene][p].items()}
                    for m, pred in predictions.items():
                        cm = confusion_batch(torch.from_numpy(pred).to(args.device), torch.from_numpy(target).to(args.device), bank.class_count)[0].cpu().numpy()
                        matrices[scene][p][m].append(cm)
                    transitions[scene][p].append(transition_counts(predictions['NoAdmission_Exact'],
                        predictions['RivalFineHard_Exact'], target, bank.class_count))
        result = {'status': 'complete' if index+1 == len(keys) else 'running', 'implementation': IMPLEMENTATION,
            'config': asdict(CONFIG), 'processed_images': index+1, 'total_images': len(keys), 'sample_keys': keys,
            'base_signature': prior['signature'], 'vocabulary_source_sha256': hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
            'context_alias_counts': list(COUNTS), 'local_geometry_alias_count': 20, 'context_vocabularies': context,
            'window_pilot_only': True, 'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True,
            'diagnostics': diagnostics,
            'metrics': {s: {p: {m: summary(np.stack(v).sum(0), banks[p].class_names, ignored[p]) for m, v in group.items()}
                for p, group in protocols.items()} for s, protocols in matrices.items()} if not smoke else {},
            'transitions': {s: {p: {'counts': np.stack(v).sum(0).tolist(), **transition_summary(np.stack(v).sum(0))}
                for p, v in group.items()} for s, group in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(keys)}), flush=True)
    if not smoke:
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
            **{s+'__'+p+'__'+m: np.stack(v) for s, ps in matrices.items() for p, arms in ps.items() for m, v in arms.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(keys),
            **{s+'__'+p: np.stack(v) for s, ps in transitions.items() for p, v in ps.items()})
    result['weights_frozen'] = all(not v.requires_grad for model in (geometry.backbone, vip.backbone) for v in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(states[prefix+n], v) for prefix, model in
        (('g_', geometry.backbone), ('v_', vip.backbone)) for n, v in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Frozen model weights changed.')
    save(output/'results.json', result)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke)
