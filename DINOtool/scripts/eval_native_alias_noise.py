"""Matched fixed-window vocabulary robustness of frozen native alias admission."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator, suppression_caps
from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.excess_alias_rejection import semantic_contradictions
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.matched_contribution_alias import stencil_margins
from dinotool.model import checkpoint_manifest
from dinotool.native_alias_noise import (IMPLEMENTATION, CONFIG, METHODS, PRIMARY, SCENARIOS,
    STYLE_FILES, ALIAS_SHUFFLES, RANDOM_DELETIONS, hard_pair_observation, signed_potential)
from dinotool.native_query_alias import native_risk
from dinotool.pair_context_reader import pair_observation
from dinotool.prompts import load_class_specs
from dinotool.rival_preserving_alias import competitive_potential, directional_controls
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.stratified_soft_alias import crop_stencil
from dinotool.target_context_alias import alias_permutations
from audit_alias_action_capacity import confusion_batch, dense, save
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import make_variants, prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_native_query_alias import native_observations
from eval_pair_context_reader import ORIGINAL
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/native_query_alias_screen_r2_20261003'
HISTORY = {'Geometry': 'Geometry', 'NoAdmission_Exact': 'Anchored_Exact', PRIMARY: 'NativeQueryOnly_Exact'}


def variants_for_dataset(dataset, specs, geometry, banks, vip, queries):
    variants, manifest = make_variants(geometry, banks, vip, queries, specs)
    description = {s: {'kind': 'historical' if s == 'clean' else 'constructed-class-name-only'} for s in SCENARIOS}
    if dataset in STYLE_FILES:
        path = TOOL/'configs'/STYLE_FILES[dataset]
        choices = load_class_specs(path)
        by_name = {s.name: s for s in choices}
        changed = {p: [by_name[s.name] for s in classes] for p, classes in specs.items()}
        if any(len(s.synonyms) != 20 for classes in changed.values() for s in classes):
            raise ValueError('Style vocabulary must have exactly20 distinct aliases/class.')
        same = changed == specs
        if same:
            variants['llm_style'] = variants['clean']
        else:
            style_banks = {p: geometry.encode_text(classes) for p, classes in changed.items()}
            style_queries = {p: vip.encode_queries(style_banks[p].class_names, tuple(s.synonyms for s in classes))
                             for p, classes in changed.items()}
            variants['llm_style'] = (style_banks, style_queries)
        description['llm_style'] = {'kind': 'historical-LLM-style-not-verified-provider-output',
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_metadata': json.loads(path.read_text())['protocol'],
            'identical_to_clean': same}
    for scenario, (vbanks, vqueries) in variants.items():
        for p, bank in vbanks.items():
            if (bank.class_names != banks[p].class_names or bank.alias_names != vqueries[p].aliases
                    or not torch.equal(bank.parent_indices, vqueries[p].parents)
                    or any(int((bank.parent_indices == c).sum()) != 20 for c in range(bank.class_count))):
                raise ValueError('Scenario changes classes, grouping or20-count contract.')
        description[scenario]['vocabularies'] = {p: [{'name': name,
            'synonyms': [alias for i, alias in enumerate(bank.alias_names) if int(bank.parent_indices[i]) == c]}
            for c, name in enumerate(bank.class_names)] for p, bank in vbanks.items()}
        description[scenario]['replacements'] = manifest.get(scenario, {})
    return variants, description


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous, extension=None, methods=METHODS):
    device = geometry.device
    coordinates, valid, operator = [torch.from_numpy(original[k]).to(device) for k in ('coordinates', 'valid', 'operator')]
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(device))
    actual_operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    operator_error = float((actual_operator-operator).abs().max())
    if operator_error > 1e-10:
        raise RuntimeError('Changed original Geometry operator.')
    broad, crops, _, count = prepare_wide(image, vip, variants)
    qflat = {s+'__'+p: q for s, (_, qs) in variants.items() for p, q in qs.items()}
    groups = {s+'__'+p: torch.stack([(b.parent_indices == c).nonzero().flatten() for c in range(b.class_count)])
              for s, (bs, _) in variants.items() for p, b in bs.items()}
    native, _, frozen, costs = native_observations(image, vip, qflat, coordinates, valid, groups)
    output, diagnostics = {}, {'operator_replay_max_error': operator_error, **costs,
        'new_intervention_forwards': 0, 'wide_forwards': len(next(iter(crops.values()))), 'scenarios': {}}
    local = {}
    for p, bank in banks.items():
        actual = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
            bank.parent_indices, bank.class_count)/.07
        local[p] = torch.from_numpy(original[p+'__local']).to(device)
        error = float((actual-local[p]).abs().max())
        if error > 1e-4:
            raise RuntimeError('Changed original local Geometry scores.')
        diagnostics.setdefault('local_replay_max_error', {})[p] = error
    for scenario, (vbanks, queries) in variants.items():
        output[scenario], diagnostics['scenarios'][scenario] = {}, {}
        for p, bank in vbanks.items():
            key, group = scenario+'__'+p, groups[scenario+'__'+p]
            canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                for c in range(bank.class_count)])
            b = sample_broad(broad[scenario, p], 0, 0, *image.shape[-2:]).reshape_as(local[p])
            baseline = local[p].double()+operator @ (b.double()-local[p].double())
            full = torch.zeros_like(native[key])
            profiled = torch.zeros_like(b)
            for crop in crops[scenario, p]:
                ids, coeff = crop_stencil(crop, count, coordinates, tuple(image.shape[-2:]))
                evidence = profiled_logits(crop, group)
                full += stencil_margins(evidence, ids, coeff, CONFIG.beta)
                profiled += ((CONFIG.beta*evidence).logsumexp(-1)[ids]/CONFIG.beta*coeff[..., None]).sum(1)
            profile_error = float((profiled-b).abs().max())
            if profile_error > 1e-4:
                raise RuntimeError('Original profiled class field fails broad replay.')
            if scenario == 'clean':
                baseline = torch.from_numpy(previous[p+'__Anchored_Exact__scores']).to(device)
                full_error = float((full-torch.from_numpy(previous[p+'__broad_margin']).to(device)).abs().max())
                native_error = float((native[key]-torch.from_numpy(previous[p+'__native_margin']).to(device)).abs().max())
                b_error = float((b-torch.from_numpy(original['clean__'+p+'__broad']).to(device)).abs().max())
                if max(full_error, native_error, b_error) > 1e-4:
                    raise RuntimeError('Historical matched source observation differs.')
            gamma = native_risk(full, native[key], native[key], bank.parent_indices, canonical, valid)
            text = semantic_contradictions(queries[p].features.float().mean(1), bank.parent_indices, canonical, CONFIG.epsilon)
            text = text[None].expand_as(gamma).masked_fill(~valid[:, None, None], 0.).clone()
            text.scatter_(-1, bank.parent_indices[None, :, None].expand(len(valid), -1, 1), 0.)
            risks = {PRIMARY: gamma, 'TextOnly_Exact': text}
            permutations = alias_permutations(group, canonical, CONFIG.random_seed)
            random_risks = {}
            for soft_name, hard_name, permutation in zip(ALIAS_SHUFFLES, RANDOM_DELETIONS, permutations):
                shuffled = torch.empty_like(gamma)
                grouped = gamma[:, group]
                shuffled[:, group] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
                risks[soft_name], random_risks[hard_name] = shuffled, shuffled
            cap = suppression_caps(crops[scenario, p], count, coordinates, tuple(image.shape[-2:]), group,
                group == canonical[:, None], valid, CONFIG.beta, CONFIG.query_chunk)[1]
            zero = pair_observation(crops[scenario, p], count, coordinates, tuple(image.shape[-2:]), group,
                torch.zeros_like(gamma), valid, CONFIG.beta, CONFIG.query_chunk)[2]
            if bool((zero != 0).any()):
                raise RuntimeError('Zero-risk writer identity differs.')
            values = {'Geometry': torch.from_numpy(previous[p+'__Geometry__scores']).to(device),
                'NoAdmission_Exact': baseline, 'MeanLogit_Original': .5*(local[p].double()+b.double())}
            stats = {'profile_replay_max_error': profile_error, 'sources': {}, 'hard': {}, 'controls': {}}
            primary_directed = None
            for method, risk in risks.items():
                directed = pair_observation(crops[scenario, p], count, coordinates, tuple(image.shape[-2:]), group,
                    risk, valid, CONFIG.beta, CONFIG.query_chunk)[2]
                excess = float((-directed-cap[..., None]).clamp_min(0).max())
                if excess > 1e-6 or float(risk[:, canonical].abs().max()) != 0:
                    raise RuntimeError('Protected soft action bounds differ.')
                potential, _, consistency = competitive_potential(directed, valid)
                values[method] = baseline+operator @ potential
                stats['sources'][method] = {**consistency, 'capacity_excess_max': excess,
                    'canonical_risk_max': float(risk[:, canonical].abs().max()),
                    'mean_risk': float(risk[valid].mean()), 'active_fraction': float((risk[valid] > 0).float().mean()),
                    'risk_spectrum_max_error': float((risk[:, group].sort(2).values-gamma[:, group].sort(2).values).abs().max())
                        if method in ALIAS_SHUFFLES else None}
                if method == PRIMARY:
                    primary_directed = directed
                    values['NativeAlias_MeanLogit'] = .5*(local[p].double()+b.double()+potential)
                    frozen[key+'__risk'] = gamma.cpu().numpy()
                    frozen[key+'__directed'] = directed.cpu().numpy()
            for method, risk in {'HardDelete_Exact': gamma, **random_risks}.items():
                directed = hard_pair_observation(crops[scenario, p], count, coordinates, tuple(image.shape[-2:]), group,
                    risk, valid, CONFIG.beta, CONFIG.query_chunk)
                potential, consistency = signed_potential(directed, valid)
                values[method] = baseline+operator @ potential
                stats['hard'][method] = {**consistency,
                    'deleted_fraction': float((risk[valid] > 0).float().mean()),
                    'count_max_error': int(((risk[:, group] > 0).sum(2)-(gamma[:, group] > 0).sum(2)).abs().max()),
                    'canonical_risk_max': float(risk[:, canonical].abs().max()),
                    'positive_directed_fraction': float((directed[valid] > 0).double().mean())}
            for old_name, changed in directional_controls(primary_directed.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
                method = 'Noise'+old_name
                directed = torch.from_numpy(changed).to(device)
                potential, _, consistency = competitive_potential(directed, valid)
                values[method] = baseline+operator @ potential
                stats['controls'][method] = {**consistency,
                    'pair_budget_max_error': float((directed.sum(0)-primary_directed.double().sum(0)).abs().max()),
                    'capacity_excess_max': float((-directed-cap.double()[..., None]).clamp_min(0).max()),
                    'pair_spectrum_max_error': float((directed[valid].sort(0).values-primary_directed.double()[valid].sort(0).values).abs().max())}
            if scenario == 'clean':
                stats['clean_score_replay_max_errors'] = {m: float((values[m]-torch.from_numpy(previous[p+'__'+old+'__scores']).to(device)).abs().max())
                    for m, old in HISTORY.items()}
                if max(stats['clean_score_replay_max_errors'].values()) > 1e-4:
                    raise RuntimeError('Historical clean endpoints differ.')
                stats['clean_source_replay_max_errors'] = {'full': full_error, 'native': native_error, 'broad': b_error}
            if set(values) != set(METHODS):
                raise RuntimeError('Missing robustness controls.')
            if extension is not None:
                extension(values=values, stats=stats, frozen=frozen, key=key,
                    crops=crops[scenario, p], count=count, coordinates=coordinates,
                    image_size=tuple(image.shape[-2:]), members=group, risks=risks,
                    valid=valid, baseline=baseline, operator=operator, local=local[p], broad=b)
            if set(values) != set(methods):
                raise RuntimeError('Missing extended robustness controls.')
            for method, score in values.items():
                frozen[key+'__'+method+'__scores'] = score.cpu().numpy()
            frozen[key+'__broad_margin'], frozen[key+'__native_margin'] = full.cpu().numpy(), native[key].cpu().numpy()
            output[scenario][p], diagnostics['scenarios'][scenario][p] = values, stats
    if not all(np.isfinite(value).all() for value in frozen.values()):
        raise RuntimeError('Nonfinite pre-mask robustness fields.')
    return output, frozen, diagnostics


def main(args, smoke=False, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION):
    output = Path(args.output_dir)
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if output.exists() or len(keys) != 8 or len(set(keys)) != 8 or not prior['coverage_verified']:
        raise ValueError('New output and verified eight-image sequence required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if (checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != prior['signature']['vocabulary']['sha256']):
        raise ValueError('Changed baseline checkpoints or vocabulary.')
    variants, vocab_manifest = variants_for_dataset(args.dataset, specs, geometry, banks, vip, queries)
    states = {prefix+k: v.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
              for k, v in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    (output/'numerical_cache').mkdir()
    save(output/'vocabularies.json', vocab_manifest)
    selected = keys[:1] if smoke else keys
    matrices = {s: {p: {m: [] for m in methods} for p in banks} for s in variants}
    transitions = {s: {p: {m: [] for m in methods if m != 'NoAdmission_Exact'} for p in banks} for s in variants}
    diagnostics, ignored = {}, dict.fromkeys(banks, 0)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, key in enumerate(selected):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                PREVIOUS/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as previous:
            scores, frozen, diagnostics[key] = predictor(image, geometry, banks, vip, variants, original, previous)
        np.savez_compressed(output/'numerical_cache'/f'{index}.npz', **frozen)
        if not smoke:
            for p, bank in banks.items():
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, dtype=np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                ignored[p] += int(((target < 0) | (target >= bank.class_count)).sum())
                for scenario in variants:
                    predictions = {m: dense(v).argmax(-1).cpu().numpy() for m, v in scores[scenario][p].items()}
                    for method, pred in predictions.items():
                        cm = confusion_batch(torch.from_numpy(pred).to(args.device), torch.from_numpy(target).to(args.device), bank.class_count)[0].cpu().numpy()
                        matrices[scenario][p][method].append(cm)
                        if method != 'NoAdmission_Exact':
                            transitions[scenario][p][method].append(transition_counts(predictions['NoAdmission_Exact'], pred, target, bank.class_count))
        result = {'status': 'complete' if index+1 == len(selected) else 'running', 'implementation': implementation,
            'config': asdict(CONFIG), 'processed_images': index+1, 'total_images': len(selected),
            'sample_keys': selected, 'signature': prior['signature'], 'scenarios': list(variants),
            'vocabularies': vocab_manifest, 'target_masks_loaded': not smoke, 'numerical_caches_precede_masks': True,
            'local_anchor_unchanged': True, 'window_pilot_only': True, 'diagnostics': diagnostics,
            'metrics': {s: {p: {m: summary(np.stack(v).sum(0), banks[p].class_names, ignored[p]) for m, v in group.items()}
                for p, group in protocols.items()} for s, protocols in matrices.items()} if not smoke else {},
            'transitions': {s: {p: {m: {'counts': np.stack(v).sum(0).tolist(), **transition_summary(np.stack(v).sum(0))}
                for m, v in group.items()} for p, group in protocols.items()} for s, protocols in transitions.items()} if not smoke else {},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(selected)}), flush=True)
    if not smoke:
        np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(selected),
            **{s+'__'+p+'__'+m: np.stack(v) for s, protocols in matrices.items() for p, group in protocols.items() for m, v in group.items()})
        np.savez_compressed(output/'per_image_transitions.npz', sample_keys=np.asarray(selected),
            **{s+'__'+p+'__'+m: np.stack(v) for s, protocols in transitions.items() for p, group in protocols.items() for m, v in group.items()})
    result['weights_frozen'] = all(not v.requires_grad for model in (geometry.backbone, vip.backbone) for v in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(states[prefix+k], v) for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
        for k, v in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Frozen weights changed.')
    save(output/'results.json', result)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke)
