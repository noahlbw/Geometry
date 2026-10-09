"""Fixed-window alias counterfactuals and labeled action-capacity audit."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import (IMPLEMENTATION, action_envelope, changed_class_prediction,
    label_assisted_action, reconstruction_operator, suppression_caps, target_margin_upper)
from dinotool.class_relative_alias_rejection import ClassRelativeAliasReader, relative_retention
from dinotool.excess_alias_rejection import ExcessRejectConfig, canonical_support
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import make_variants, prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_stratified_soft_alias import tile_coordinates


CONFIG = ExcessRejectConfig()
METHODS = ('Geometry', 'Anchored_CG', 'Anchored_Exact', 'ClassRelativeReject_CG',
           'LabelPolicy_Protected', 'LabelPolicy_Unrestricted')
SCENARIOS = ('clean', 'wrong_parent', 'paraphrase')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dense(scores):
    return F.interpolate(scores.T.reshape(1, -1, 32, 32), (512, 512),
                         mode='bilinear', align_corners=False)[0].permute(1, 2, 0)


def confusion_batch(prediction, target, classes):
    prediction = prediction.reshape(-1, target.numel())
    flat = target.flatten()
    valid = (flat >= 0) & (flat < classes)
    offsets = torch.arange(len(prediction), device=target.device)[:, None] * classes ** 2
    encoded = offsets + flat[valid][None]*classes + prediction[:, valid]
    return torch.bincount(encoded.flatten(), minlength=len(prediction)*classes**2).reshape(-1, classes, classes)


def patch_targets(target, classes):
    centre = target[8::16, 8::16].flatten()
    valid = (target >= 0) & (target < classes)
    one_hot = F.one_hot(target.clamp(0, classes-1), classes).permute(2, 0, 1).float() * valid[None]
    counts = F.avg_pool2d(one_hot[None], 16, 16)[0]
    majority = counts.argmax(0).masked_fill(counts.sum(0) == 0, -1).flatten()
    return centre, majority


def envelope_counts(baseline, operator, cap, centre):
    low, high = action_envelope(baseline, operator, cap)
    margin = target_margin_upper(low, high, centre)
    guard = 64*torch.finfo(torch.float64).eps*(1+float(baseline.abs().max())
            + float(operator.abs().sum(-1).max())*float(cap.max()))
    wrong = (baseline.argmax(-1) != centre) & torch.isfinite(margin)
    result = []
    for c in range(baseline.shape[-1]):
        own = centre == c
        bad = wrong & own
        result.append([int(own.sum()), int(bad.sum()), int((bad & (margin < -guard)).sum()),
                       int((bad & (margin > guard)).sum()), int((bad & (margin.abs() <= guard)).sum())])
    return np.asarray(result, np.int64), guard


@torch.inference_mode()
def observe(image, geometry, banks, vip, variants, readers):
    h, w = image.shape[-2:]
    broad, crops, auxiliary, count = prepare_wide(image, vip, variants)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    coordinates = tile_coordinates(0, 0, geometry.device)
    valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
    operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    cache = {'operator': operator.cpu().numpy(), 'valid': valid.cpu().numpy(),
             'coordinates': coordinates.cpu().numpy()}
    entries, numerical = {}, {'operator_residual': residual}
    for key, original_bank in banks.items():
        local_aliases = (prepared.geometry_projected.float() @ F.normalize(original_bank.features.float(), dim=-1).T)[0]
        local = alias_class_scores(local_aliases, original_bank.parent_indices, original_bank.class_count) / CONFIG.local_temperature
        cache[key + '__local'] = local.cpu().numpy()
        for scenario, (group, _) in variants.items():
            reader, bank = readers[scenario, key], group[key]
            alias = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
            b = sample_broad(broad[scenario, key], 0, 0, h, w).reshape_as(local)
            cg, _ = anchored_innovation(local[None], b[None], prepared.geometry_patch_conditional, valid[None])
            exact = local.double() + operator @ (b.double()-local.double())
            corrected, diagnostics = reader.read(alias, prepared.geometry_patch_conditional[0], coordinates,
                valid, b, crops[scenario, key], auxiliary[scenario, key], count, (h, w), SETTINGS.tau)
            source, _ = anchored_innovation(local[None], corrected['ClassRelativeReject_Coupled'][None],
                                            prepared.geometry_patch_conditional, valid[None])
            support, supported, _ = canonical_support(alias[:, reader.canonical]/CONFIG.local_temperature,
                                                       prepared.geometry_patch_conditional[0], coordinates, valid, CONFIG)
            risk = 1-relative_retention(support, supported, reader.conflict, bank.parent_indices, CONFIG.epsilon)
            risk = risk[:, reader.members].masked_fill(~valid[:, None, None], 0.)
            caps = suppression_caps(crops[scenario, key], count, coordinates, (h, w), reader.members,
                                    bank.canonical_mask[reader.members], valid, SETTINGS.tau)
            prefix = scenario + '__' + key
            for name, value in (('broad', b), ('baseline_exact', exact), ('baseline_cg', cg[0]),
                                ('source_cg', source[0]), ('risk', risk), ('single_cap', caps[0]),
                                ('protected_cap', caps[1]), ('unrestricted_cap', caps[2])):
                cache[prefix + '__' + name] = value.cpu().numpy()
            numerical[prefix] = {'cg_exact_max_error': float((cg[0].double()-exact).abs().max()),
                'cg_exact_changed_patch_centres': int(((cg[0].argmax(-1) != exact.argmax(-1)) & valid).sum()),
                **diagnostics}
            entries[scenario, key] = {'Geometry': local, 'Anchored_CG': cg[0], 'Anchored_Exact': exact,
                'ClassRelativeReject_CG': source[0], 'risk': risk, 'single_cap': caps[0],
                'protected_cap': caps[1], 'unrestricted_cap': caps[2]}
    return cache, entries, operator, numerical


@torch.inference_mode()
def audit_entry(entry, operator, target, classes):
    centre, majority = patch_targets(target, classes)
    base = entry['Anchored_Exact']
    fields = {method: entry[method] for method in METHODS[:4]}
    envelope, guards = {}, {}
    for label, name in (('Protected', 'protected_cap'), ('Unrestricted', 'unrestricted_cap')):
        cap = entry[name]
        fields['LabelPolicy_' + label] = base + operator @ label_assisted_action(cap, majority).double()
        envelope[label], guards[label] = envelope_counts(base, operator, cap, centre)
    maps = {method: dense(value).reshape(-1, classes) for method, value in fields.items()}
    cms = {method: confusion_batch(value.argmax(-1), target, classes)[0].cpu().numpy()
           for method, value in maps.items()}
    baseline = maps['Anchored_Exact']
    single = operator @ (-entry['single_cap'].double().flatten(1))
    single = single.reshape(len(base), classes, -1)
    single_cms, group_cms = [], []
    groups = operator @ (-entry['protected_cap'].double())
    group_dense = dense(groups)
    for c in range(classes):
        shifts = F.interpolate(single[:, c].T.reshape(1, -1, 32, 32), (512, 512),
                               mode='bilinear', align_corners=False)[0].flatten(1)
        changed = changed_class_prediction(baseline, baseline[:, c][None]+shifts, c)
        single_cms.append(confusion_batch(changed, target, classes).cpu().numpy())
        shift = group_dense[:, :, c].flatten()
        changed = changed_class_prediction(baseline, baseline[:, c]+shift, c)
        group_cms.append(confusion_batch(changed, target, classes)[0].cpu().numpy())
    active = entry['single_cap'].flatten(1)
    risk = entry['risk'].flatten(1)
    source = (risk*active).sum(0) / active.sum(0).clamp_min(1e-12)
    return cms, np.concatenate(single_cms), np.stack(group_cms), envelope, source.cpu().numpy(), guards


def main(args, smoke=False):
    output = Path(args.output_dir)
    if output.exists() or args.num_shards != 1 or args.shard_index != 0:
        raise ValueError('New isolated single-shard audit output required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    requested = json.loads(args.source_diagnostic.read_text())['signature']['samples'][:8]
    if len(requested) != 8 or len(set(requested)) != 8:
        raise ValueError('Exactly8 fixed unique image keys required.')
    lookup = {sample.key: sample for sample in samples}
    selected = [lookup[key] for key in (requested[:1] if smoke else requested)]
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    checkpoint_identity = checkpoint_manifest(checkpoints)
    vocabulary_identity = {'sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        'aliases': {key: bank.alias_names for key, bank in banks.items()},
        'counts': {key: [20]*bank.class_count for key, bank in banks.items()}}
    variants, vocabulary = make_variants(geometry, banks, vip, queries, specs)
    readers = {(scenario, key): ClassRelativeAliasReader(bank, CONFIG)
               for scenario, (group, _) in variants.items() for key, bank in group.items()}
    head_states = {prefix+key: value.clone() for prefix, model in (('g_', geometry.backbone), ('b_', vip.backbone))
                   for key, value in model.model.visual_model.head.state_dict().items()}
    output.mkdir(parents=True)
    cache_dir = output / 'numerical_cache'
    cache_dir.mkdir()
    selection = {scenario+'__'+key: reader.report() for (scenario, key), reader in readers.items()}
    save(output / 'selection.json', selection)
    per_image, numerical, ignored = {}, {}, {key: 0 for key in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, sample in enumerate(selected):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        cache, entries, operator, numerical[sample.key] = observe(image, geometry, banks, vip, variants, readers)
        if not all(np.isfinite(value).all() for value in cache.values()):
            raise ValueError('Nonfinite numerical cache.')
        if any(fields['cg_exact_max_error'] > 1e-4 for fields in numerical[sample.key].values() if isinstance(fields, dict)):
            raise ValueError('Original CG and exact-system audit differ by more than1e-4.')
        np.savez_compressed(cache_dir / (str(index) + '.npz'), **cache)
        if not smoke:
            # All numerical observations, actions and frozen source scores precede masks.
            for key, bank in banks.items():
                full = load_mask(sample, key, tuple(image.shape[-2:]))
                target = torch.full((512, 512), -1, dtype=torch.long, device=operator.device)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = torch.from_numpy(full[:ah, :aw].astype(np.int64)).to(operator.device)
                ignored[key] += int(((target < 0) | (target >= bank.class_count)).sum())
                for scenario in SCENARIOS:
                    cms, single, groups, envelopes, source, guards = audit_entry(entries[scenario, key], operator, target, bank.class_count)
                    prefix = scenario + '__' + key + '__'
                    for method, cm in cms.items():
                        per_image.setdefault(prefix+method, []).append(cm)
                    for name, values in (('single', single), ('groups', groups), ('source', source),
                                         ('envelope_protected', envelopes['Protected']), ('envelope_unrestricted', envelopes['Unrestricted'])):
                        per_image.setdefault(prefix+name, []).append(values)
                    numerical[sample.key][scenario+'__'+key]['envelope_guards'] = guards
        result = {'status': 'complete' if index+1 == len(selected) else 'running', 'implementation': IMPLEMENTATION,
            'audit_only': True, 'smoke': smoke, 'target_masks_loaded': not smoke,
            'numerical_caches_precede_masks': True, 'used_to_fit_selector': False,
            'processed_images': index+1, 'total_images': len(selected), 'sample_keys': [row.key for row in selected[:index+1]],
            'signature': {'dataset': args.dataset, 'config': asdict(CONFIG), 'sample_keys': [row.key for row in selected],
                'window': 'fixed top-left512, original whole-image wide view',
                'history': str(args.source_diagnostic), 'classes': {key: bank.class_names for key, bank in banks.items()},
                'geometry': asdict(geometry.config), 'observation': asdict(SETTINGS),
                'checkpoints': checkpoint_identity, 'vocabulary': vocabulary_identity},
            'metrics': {scenario: {key: {method: summary(np.stack(per_image[scenario+'__'+key+'__'+method]).sum(0),
                bank.class_names, ignored[key]) for method in METHODS} for key, bank in banks.items()}
                for scenario in SCENARIOS} if not smoke else {},
            'numerical': numerical, 'vocabulary_stress': vocabulary, 'wall_seconds': time.perf_counter()-started,
            'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
        save(output / 'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'total': len(selected)}), flush=True)
    if not smoke:
        np.savez_compressed(output / 'per_image_audit.npz', sample_keys=np.asarray(requested),
                            **{key: np.stack(values) for key, values in per_image.items()})
    result['weights_frozen'] = all(not value.requires_grad for model in (geometry.backbone, vip.backbone)
                                  for value in model.model.parameters())
    result['head_weights_unchanged'] = all(torch.equal(head_states[prefix+key], value)
        for prefix, model in (('g_', geometry.backbone), ('b_', vip.backbone))
        for key, value in model.model.visual_model.head.state_dict().items())
    if not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise ValueError('Network weights changed in audit.')
    save(output / 'results.json', result)


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), is_smoke)
