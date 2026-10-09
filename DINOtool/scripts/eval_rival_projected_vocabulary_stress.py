"""Frozen projected rule on exact nested pools and historical word-attachment stress."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
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
from dinotool.rival_alias_count import canonical_indices
from dinotool.rival_alias_fast import cache_observations, sampled_cached
from dinotool.rival_alias_stress import IMPLEMENTATION, METHODS, SCENARIOS, evidence_audit
from dinotool.rival_competition_admission import retained_competition_scores, settings
from audit_alias_action_capacity import dense
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_native_alias_noise import ORIGINAL
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save
from eval_rival_fine_graph import graph_features, bind
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/rival_fine_coupling_20261003'
COUNTS = TOOL/'results/rival_alias_count_203040_20261003'
PROJECTED = TOOL/'results/rival_competition_admission_20261005'
VOCAB = TOOL/'results/rival_alias_count_vocabulary_r2_20261003.json'
PRIMARY = 'FineRivalProjected_Exact'


def variants_for(dataset, source, prior, banks, vip, queries):
    context = source['datasets'][dataset]
    vocabularies = {'k'+str(k): {p: context[p][str(k)] for p in banks} for k in (20, 30, 40)}
    vocabularies.update({s: prior['vocabularies'][s]['vocabularies'] for s in ('wrong_parent', 'paraphrase')})
    for p, bank in banks.items():
        base = vocabularies['k20'][p]
        if base != prior['vocabularies']['clean']['vocabularies'][p]:
            raise RuntimeError('Historical20 contextual prefix changed.')
        for k in (20, 30, 40):
            entries = vocabularies['k'+str(k)][p]
            if any(c['synonyms'] != vocabularies['k40'][p][i]['synonyms'][:k]
                   or c['name'] != base[i]['name'] or len(c['synonyms']) != k for i, c in enumerate(entries)):
                raise RuntimeError('Contextual pools are not exact nested prefixes.')
    variants = {'k20': (banks, queries)}
    for scene in SCENARIOS[1:]:
        variants[scene] = (banks, {p: vip.encode_queries(bank.class_names,
            tuple(tuple(c['synonyms']) for c in vocabularies[scene][p])) for p, bank in banks.items()})
    for scene, (_, group) in variants.items():
        for p, query in group.items():
            classes = vocabularies[scene][p]
            if (list(query.class_names) != [c['name'] for c in classes]
                    or list(query.aliases) != [a for c in classes for a in c['synonyms']]
                    or any(int((query.parents == i).sum()) != len(c['synonyms']) for i, c in enumerate(classes))):
                raise RuntimeError('Contextual query identity or count changed.')
    return variants, vocabularies


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, projected, capture_observations=False):
    coordinates, known, operator = [torch.from_numpy(original[k]).to(geometry.device)
                                    for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    actual, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], known)
    if not torch.equal(actual, operator):
        raise RuntimeError('Frozen Geometry reconstruction changed.')
    wide, crops, _, wide_count = prepare_wide(image, vip, variants)
    flat_queries = {s+'__'+p: q for s, (_, qs) in variants.items() for p, q in qs.items()}
    fine, fine_count, costs = bind(observe_fine, fine_patch_features=graph_features)(
        image[:, :512, :512], vip, flat_queries, coordinates, known)
    locals = {}
    for p, bank in banks.items():
        local = alias_class_scores((prepared.geometry_projected.float() @
            F.normalize(bank.features.float(), dim=-1).T)[0], bank.parent_indices, bank.class_count)/.07
        if not torch.equal(local, torch.from_numpy(original[p+'__local']).to(geometry.device)):
            raise RuntimeError('Frozen local Geometry20 scores changed.')
        locals[p] = local
    scores, flags, metadata, diagnostics = {}, {}, {}, {}
    for scene, (_, queries) in variants.items():
        scores[scene], flags[scene], metadata[scene], diagnostics[scene] = {}, {}, {}, {}
        for p, query in queries.items():
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            canonical = canonical_indices(query.class_names, query.aliases, query.parents)
            broad = sample_broad(wide[scene, p], 0, 0, *image.shape[-2:]).reshape_as(locals[p])
            inputs = (locals[p], operator, broad, crops[scene, p], wide_count, fine[scene+'__'+p],
                      fine_count, coordinates, coordinates, known, members, canonical, query.parents, tuple(image.shape[-2:]))
            values, risk, details = retained_competition_scores(*inputs, methods=METHODS)
            if scene == 'k20' and not np.array_equal(values[PRIMARY].cpu().numpy(), projected[p+'__'+PRIMARY]):
                raise RuntimeError('Frozen clean20 projected score changed.')
            coarse = sampled_cached(cache_observations(crops[scene, p], wide_count, coordinates, tuple(image.shape[-2:]), members), coordinates)
            native = sampled_cached(cache_observations(fine[scene+'__'+p], fine_count, coordinates, (512, 512), members), coordinates)
            source_flags = (coarse > 0, native >= 0, risk > 0)
            # Subset names affect diagnostics only, after the frozen reader returns.
            subsets = {'all': torch.ones_like(query.parents, dtype=torch.bool),
                       'old20': torch.zeros_like(query.parents, dtype=torch.bool)}
            subsets['old20'][members[:, :20].flatten()] = True
            if members.shape[1] > 20:
                subsets['added'] = ~subsets['old20']
            if scene in ('wrong_parent', 'paraphrase'):
                subsets['replacement_slots'] = torch.zeros_like(query.parents, dtype=torch.bool)
                subsets['replacement_slots'][members[:, 16:20].flatten()] = True
            source_diagnostics = {name: evidence_audit(*source_flags, query.parents, canonical, known, alias_subset=subset)
                                  for name, subset in subsets.items()}
            if any(v['uncontradicted_but_rejected_comparisons'] for v in source_diagnostics.values()):
                raise RuntimeError('Selector contradicted its declared fine sign rule.')
            scores[scene][p], flags[scene][p] = values, source_flags
            metadata[scene][p] = {'parents': query.parents, 'canonical': canonical, 'known': known, 'subsets': subsets}
            if capture_observations:
                metadata[scene][p]['reader_arguments'] = inputs
            diagnostics[scene][p] = {'reader': details, 'source': source_diagnostics}
    return scores, flags, metadata, {'scenarios': diagnostics, **costs,
        'geometry_forwards': 1, 'wide_forwards': len(next(iter(crops.values()))),
        'context_scenarios_share_visual_observations': True, 'local_and_operator_bitwise_exact': True}


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing stress output.')
    prior = json.loads((SOURCE/args.dataset/'merged.json').read_text())
    pool = json.loads(VOCAB.read_text())
    if (prior['status'] != 'complete' or not prior['coverage_verified'] or pool['status'] != 'complete'
            or pool['target_images_loaded'] or pool['target_masks_loaded'] or pool['semantic_or_visual_filtering']):
        raise RuntimeError('Verified old source and exact unfiltered text-only extensions required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    keys = prior['sample_keys']
    if len(keys) != 8 or len(set(keys)) != 8:
        raise RuntimeError('Require the frozen eight-key developed panel.')
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Frozen checkpoints changed.')
    variants, vocabularies = variants_for(args.dataset, pool, prior, banks, vip, queries)
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'source_cache').mkdir()
    matrices = {s: {p: {m: np.zeros((b.class_count,)*2, np.int64) for m in METHODS} for p, b in banks.items()} for s in SCENARIOS}
    per_image = {s+'__'+p+'__'+m: [] for s in SCENARIOS for p in banks for m in METHODS}
    diagnostics, ignored = [], dict.fromkeys(banks, 0)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with np.load(COUNTS/args.dataset/'per_image_confusions.npz', allow_pickle=False) as historical_counts, np.load(
            SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as historical_stress:
        for index, key in enumerate(keys):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                    PROJECTED/args.dataset/'scores'/f'{index}.npz', allow_pickle=False) as old_projected:
                scores, flags, metadata, diag = predict(image, geometry, banks, vip, variants, original, old_projected)
            packed = {'sample_key': np.asarray(key)}
            for scene in SCENARIOS:
                for p, values in scores[scene].items():
                    prefix = scene+'__'+p+'__'
                    packed.update({prefix+m+'__scores': value.cpu().numpy() for m, value in values.items()})
                    packed.update({prefix+name: metadata[scene][p][name].cpu().numpy()
                                   for name in ('parents', 'canonical', 'known')})
                    packed[prefix+'flag_shape'] = np.asarray(flags[scene][p][0].shape)
                    for name, value in zip(('broad_positive', 'fine_nonnegative', 'rejected'), flags[scene][p]):
                        packed[prefix+name+'_packed'] = np.packbits(value.cpu().numpy())
            np.savez_compressed(output/'source_cache'/f'{index}.npz', **packed)
            # Only after all five scenario scores/source flags are persisted do masks enter.
            for p, bank in banks.items():
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                valid = (target >= 0) & (target < bank.class_count)
                ignored[p] += int((~valid).sum())
                query_target = torch.from_numpy(target[8::16, 8::16].reshape(-1)).to(geometry.device)
                for scene in SCENARIOS:
                    meta = metadata[scene][p]
                    diag['scenarios'][scene][p]['target_audit'] = {name: evidence_audit(*flags[scene][p],
                        meta['parents'], meta['canonical'], meta['known'], alias_subset=subset, query_target=query_target)['target_audit']
                        for name, subset in meta['subsets'].items()}
                    for method, value in scores[scene][p].items():
                        prediction = dense(value).argmax(-1).cpu().numpy()
                        encoded = target[valid]*bank.class_count+prediction[valid]
                        cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                        if method in ('NoAdmission_Exact', 'RivalFineHard_Exact'):
                            if scene.startswith('k'):
                                old = historical_counts[scene+'__'+p+'__'+method][index]
                            else:
                                old = historical_stress[scene+'__'+p+'__'+method][index]
                            if not np.array_equal(cm, old):
                                raise RuntimeError('Historical scenario per-image control changed: '+scene+'/'+p+'/'+method)
                        per_image[scene+'__'+p+'__'+method].append(cm)
                        matrices[scene][p][method] += cm
            diagnostics.append({'sample_key': key, **diag})
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': index+1,
                'total_images': 8, 'sample_keys': keys[:index+1], 'methods': METHODS, 'scenarios': SCENARIOS,
                'base_signature': prior['signature'], 'source_config': asdict(CONFIG), 'competition_action': settings(),
                'vocabulary_source_sha256': hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
                'context_vocabularies': vocabularies, 'local_geometry_alias_count': 20,
                'diagnostics': diagnostics, 'historical_per_image_controls_exact': True,
                'clean20_projected_scores_bitwise_exact': True, 'source_scores_persisted_before_masks': True,
                'target_audit_not_used_by_selector': True,
                'metrics': {s: {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in g.items()}
                    for p, g in ps.items()} for s, ps in matrices.items()},
                'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
            save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'controls_exact': True}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result.update(status='complete', coverage_verified=True, **check_frozen(states, geometry, vip))
    save(output/'results.json', result)
    print(json.dumps({'status': 'complete', 'dataset': args.dataset}), flush=True)


if __name__ == '__main__':
    main(parse_args())
