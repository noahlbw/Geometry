"""CPU-only, cached source-versus-writeback audit; no selector or model forwards."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from audit_alias_action_capacity import confusion_batch, dense, patch_targets, save
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from run_region_semantic_suite_a800 import SETTINGS, TOOL


IMPLEMENTATION = 'geometry-context-reliability-source-audit-v1-20261003'
ORIGINAL = TOOL/'results/alias_action_capacity_audit_20261003'
SOURCE = TOOL/'results/target_context_alias_source_r2_20261003'
WRITER = TOOL/'results/rival_preserving_alias_reader_screen_20261003'
DEFAULT_OUTPUT = TOOL/'results/context_reliability_source_audit_20261003'
ENDPOINTS = ('Anchored_Exact', 'RivalPreserving_Exact', 'ContrastReversal_Exact', 'RivalShuffledSupport_Exact')


def describe(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'q25': None, 'median': None, 'q75': None,
                'negative_fraction': None, 'positive_fraction': None}
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic field.')
    q = np.quantile(values, [.25, .5, .75])
    return {'count': len(values), 'mean': float(values.mean()), 'q25': float(q[0]),
            'median': float(q[1]), 'q75': float(q[2]),
            'negative_fraction': float((values < 0).mean()), 'positive_fraction': float((values > 0).mean())}


def alias_groups(specs, stored_aliases, counts):
    groups = [tuple(dict.fromkeys((s.name, *s.synonyms))) for s in specs]
    if ([a for group in groups for a in group] != stored_aliases
            or list(map(len, groups)) != counts or set(counts) != {20}):
        raise ValueError('Actual aliases/groups differ from the frozen20 vocabulary.')
    starts = np.cumsum([0, *counts[:-1]])
    members = np.stack([np.arange(start, start+count) for start, count in zip(starts, counts)])
    canonical = np.asarray([members[c, groups[c].index(s.name)] for c, s in enumerate(specs)])
    return members, canonical


def frozen_fields(original, source, writer, p, members, canonical):
    n, classes, words = len(original['valid']), len(members), members.shape[1]
    risk = writer[p+'__pair_risk']
    cap = original['clean__'+p+'__single_cap'].astype(np.float64)
    if cap.shape != (n, classes, words) or risk.shape != (n, classes*words, classes):
        raise ValueError('Changed risk/capacity shapes.')
    full = source[p+'__source_full'][0]
    kept, removed = [source[p+'__source_'+name] for name in ('kept', 'removed')]
    known = source['supported_units'][source['assignments']] & original['valid']
    full_margin = full[..., None]-full[:, None, canonical]
    keep_margin = kept[..., None]-kept[:, :, None, canonical]
    remove_margin = removed[..., None]-removed[:, :, None, canonical]
    eligible = (full_margin[None] > 0) & (keep_margin < 0) & (remove_margin > 0)
    replay = np.where(eligible, np.clip(np.maximum(-keep_margin, 0)/np.maximum(remove_margin-keep_margin, 1e-6), 0, 1), 0).min(0)
    replay[:, canonical] = 0
    replay[~known] = 0
    for c, group in enumerate(members):
        replay[:, group, c] = 0
    replay_error = float(np.max(np.abs(replay-risk)))
    if replay_error > 1e-6:
        raise ValueError('Source risk replay failed: '+str(replay_error))
    denominator = np.maximum(cap.sum(-1), 1e-12)[..., None]

    def weighted(value):
        return (cap[..., None]*value[:, members]).sum(2)/denominator

    directed = writer[p+'__RivalPreserving_Exact__directed'].astype(np.float64)
    potential = writer[p+'__RivalPreserving_Exact__potential'].astype(np.float64)
    requested = directed-directed.transpose(0, 2, 1)
    if not np.allclose(potential, requested.sum(-1)/classes, atol=1e-12, rtol=0):
        raise ValueError('Potential replay failed.')
    baseline = writer[p+'__Anchored_Exact__scores']
    if not np.array_equal(baseline, original['clean__'+p+'__baseline_exact']):
        raise ValueError('Original exact endpoint changed.')
    receiving = original['operator'] @ potential
    endpoint = writer[p+'__RivalPreserving_Exact__scores']
    error = float(np.max(np.abs(endpoint-baseline-receiving)))
    if error > 1e-10:
        raise ValueError('Receiving endpoint replay failed.')
    fields = {'capacity_weighted_risk': weighted(risk), 'alias_full_margin': weighted(full_margin),
              'canonical_full': full[:, canonical], 'requested': requested,
              'directed': directed, 'potential': potential, 'receiving': receiving,
              'capacity': cap.sum(-1), 'valid': original['valid'],
              'assignments': source['assignments'], 'support_masks': source['support_masks'],
              'supported_units': source['supported_units'], 'local': original[p+'__local'],
              'broad': original['clean__'+p+'__broad'], 'baseline': baseline}
    for fill in range(2):
        fields['alias_kept_margin'+str(fill)] = weighted(keep_margin[fill])
        fields['alias_removed_margin'+str(fill)] = weighted(remove_margin[fill])
        fields['canonical_kept'+str(fill)] = kept[fill][:, canonical]
        fields['canonical_removed'+str(fill)] = removed[fill][:, canonical]
    for method in ENDPOINTS:
        fields[method+'__scores'] = writer[p+'__'+method+'__scores']
    return fields, {'risk_replay_max_error': replay_error, 'receiving_replay_max_error': error}


def support_audit(target, masks, assignments, valid, classes):
    flat = np.asarray(target).reshape(32, 16, 32, 16).transpose(0, 2, 1, 3).reshape(1024, 256)
    fractions = np.stack([(flat == c).mean(-1) for c in range(classes)], -1)
    weights = masks.reshape(16, 1024).astype(np.float64)*valid[None]
    total = weights.sum(-1)
    masses = weights @ fractions
    labeled = masses.sum(-1)
    distribution = np.divide(masses, labeled[:, None], out=np.zeros_like(masses), where=labeled[:, None] > 0)
    effective = np.divide(total**2, (weights**2).sum(-1), out=np.zeros_like(total), where=(weights**2).sum(-1) > 0)
    cell = np.stack([fractions[(assignments == i) & valid].mean(0) if ((assignments == i) & valid).any()
                     else np.zeros(classes) for i in range(16)])
    cell /= np.maximum(cell.sum(-1, keepdims=True), 1e-12)
    return {'class_mass': distribution[assignments], 'dominant_fraction': distribution.max(-1)[assignments],
            'effective_patches': effective[assignments], 'cell_class_mass': cell[assignments],
            'labeled_fraction': np.divide(labeled, total, out=np.zeros_like(total), where=total > 0)[assignments]}


def pair_fields(fields, support, c, d, truth):
    def margin(value):
        return value[:, c]-value[:, d]
    result = {'risk_c_against_d': fields['capacity_weighted_risk'][:, c, d],
              'risk_d_against_c': fields['capacity_weighted_risk'][:, d, c],
              'capacity_c': fields['capacity'][:, c], 'capacity_d': fields['capacity'][:, d],
              'suppression_c_against_d': fields['directed'][:, c, d],
              'suppression_d_against_c': fields['directed'][:, d, c],
              'requested_c_minus_d': fields['requested'][:, c, d],
              'donor_margin_change': margin(fields['potential']),
              'receiving_margin_change': margin(fields['receiving']),
              'support_c_fraction': support['class_mass'][:, c], 'support_d_fraction': support['class_mass'][:, d],
              'cell_c_fraction': support['cell_class_mass'][:, c], 'cell_d_fraction': support['cell_class_mass'][:, d],
              'support_truth_fraction': support['class_mass'][np.arange(len(truth)), np.clip(truth, 0, len(fields['local'][0])-1)],
              'support_dominant_fraction': support['dominant_fraction'],
              'support_effective_patches': support['effective_patches'], 'support_labeled_fraction': support['labeled_fraction']}
    for name in ('local', 'broad', 'baseline', 'canonical_full', 'canonical_kept0', 'canonical_kept1',
                 'canonical_removed0', 'canonical_removed1'):
        result[name+'_margin'] = margin(fields[name])
    for name in ('alias_full_margin', 'alias_kept_margin0', 'alias_kept_margin1',
                 'alias_removed_margin0', 'alias_removed_margin1'):
        result[name] = fields[name][:, c, d]
    for method in ('ContrastReversal_Exact', 'RivalShuffledSupport_Exact'):
        result[method+'_receiving_margin_change'] = margin(fields[method+'__scores']-fields['baseline'])
    return result


def audit_groups(fields, support, truth, store):
    base = fields['baseline'].argmax(-1)
    candidate = fields['RivalPreserving_Exact__scores'].argmax(-1)
    valid = fields['valid'] & (truth >= 0) & (truth < fields['baseline'].shape[-1])
    classes = fields['baseline'].shape[-1]
    for c in range(classes):
        for d in range(classes):
            if c == d:
                continue
            values = pair_fields(fields, support, c, d, truth)
            groups = {'true_positive': (base == c) & (truth == c),
                      'false_positive': (base == c) & (truth == d),
                      'miss': (base == d) & (truth == c)}
            for name, condition in groups.items():
                use = valid & condition
                entry = store.setdefault(str(c)+'/'+str(d)+'/'+name, {})
                for field, value in values.items():
                    entry.setdefault(field, []).append(value[use])
                for field, value in (('candidate_correct', candidate == truth), ('candidate_changed', candidate != base),
                                     ('supported_query', fields['supported_units'][fields['assignments']])):
                    entry.setdefault(field, []).append(value[use].astype(np.float64))


def run_dataset(setting, output):
    dataset, data, vocabulary, *_ = setting
    directory = output/dataset
    directory.mkdir()
    (directory/'fields').mkdir()
    prior = json.loads((WRITER/dataset/'merged.json').read_text())
    source_prior = json.loads((SOURCE/dataset/'merged.json').read_text())
    if (not prior['coverage_verified'] or prior['processed_images'] != 8 or prior['total_images'] != 8
            or len(set(prior['sample_keys'])) != 8 or prior['signature'] != source_prior['signature']):
        raise ValueError('Changed source coverage/signature: '+dataset)
    args = SimpleNamespace(dataset=dataset, data_root=Path('/data/test/datasets')/data,
        vocabulary_config=TOOL/'configs'/vocabulary, vdd_ontology='official', sample_seed=20260923)
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    groups = {p: {kind: {} for kind in ('centre', 'majority')} for p in specs}
    metadata, replay = [], []
    with np.load(WRITER/dataset/'per_image_confusions.npz', allow_pickle=False) as cms, np.load(
            WRITER/dataset/'per_image_transitions.npz', allow_pickle=False) as transitions:
        if cms['sample_keys'].tolist() != prior['sample_keys'] or transitions['sample_keys'].tolist() != prior['sample_keys']:
            raise ValueError('Changed per-image sequence.')
        for index, key in enumerate(prior['sample_keys']):
            sample = lookup[key]
            image = load_image(sample.image_path if dataset == 'loveda' else sample)
            h, w = image.shape[-2:]
            ratio = 448/max(h, w)
            rh, rw = int(h*ratio+.5), int(w*ratio+.5)
            metadata.append({'sample_key': key, 'height': h, 'width': w, 'resized_height': rh, 'resized_width': rw,
                'wide_token_original_height': 16*h/rh, 'wide_token_original_width': 16*w/rw,
                'wide_crop_original_height': 336*h/rh, 'wide_crop_original_width': 336*w/rw})
            with np.load(ORIGINAL/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                    SOURCE/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as source, np.load(
                    WRITER/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as writer:
                pending = {}
                for p, classes in specs.items():
                    names = [s.name for s in classes]
                    if names != prior['signature']['classes'][p]:
                        raise ValueError('Changed class order.')
                    members, canonical = alias_groups(classes, prior['signature']['vocabulary']['aliases'][p],
                        prior['signature']['vocabulary']['counts'][p])
                    fields, checks = frozen_fields(original, source, writer, p, members, canonical)
                    np.savez_compressed(directory/'fields'/f'{index}_{p}.npz', **fields)
                    pending[p] = (fields, checks)
            # All derived source and action fields have been saved before this mask read.
            for p, classes in specs.items():
                fields, checks = pending[p]
                full = load_mask(sample, p, (h, w))
                target = np.full((512, 512), -1, np.int64)
                ah, aw = min(512, h), min(512, w)
                target[:ah, :aw] = full[:ah, :aw]
                classes_count = len(classes)
                predictions = {}
                for method in ENDPOINTS:
                    pred = dense(torch.from_numpy(fields[method+'__scores'])).argmax(-1)
                    cm = confusion_batch(pred, torch.from_numpy(target), classes_count)[0].numpy()
                    if not np.array_equal(cm, cms[p+'__'+method][index]):
                        raise ValueError('Dense confusion replay failed: '+dataset+'/'+p+'/'+method)
                    predictions[method] = pred.numpy()
                old, new = [predictions[m].ravel() for m in ENDPOINTS[:2]]
                flat = target.ravel()
                valid = (flat >= 0) & (flat < classes_count)
                encoded = (old[valid]*classes_count+new[valid])*classes_count+flat[valid]
                counts = np.bincount(encoded, minlength=classes_count**3).reshape((classes_count,)*3)
                if not np.array_equal(counts, transitions[p+'__RivalPreserving_Exact'][index]):
                    raise ValueError('Dense transition replay failed.')
                support = support_audit(target, fields['support_masks'], fields['assignments'], fields['valid'], classes_count)
                centre, majority = patch_targets(torch.from_numpy(target), classes_count)
                for kind, truth in (('centre', centre.numpy()), ('majority', majority.numpy())):
                    audit_groups(fields, support, truth, groups[p][kind])
                replay.append({'sample_key': key, 'protocol': p, **checks,
                    'dense_confusion_replay': True, 'transition_replay': True})
            print(json.dumps({'dataset': dataset, 'processed': index+1, 'total': 8}), flush=True)
    summarized = {p: {kind: {pair: {field: describe(np.concatenate(arrays)) for field, arrays in fields.items()}
                    for pair, fields in group.items()} for kind, group in kinds.items()} for p, kinds in groups.items()}
    result = {'implementation': IMPLEMENTATION, 'status': 'complete', 'audit_only': True,
        'inference_forwards': 0, 'gpu_allocation': False, 'derived_fields_precede_masks': True,
        'used_to_fit_selector': False, 'coverage_verified': True, 'processed_images': 8, 'total_images': 8,
        'sample_keys': prior['sample_keys'], 'signature': prior['signature'], 'metadata': metadata,
        'replay': replay, 'groups': summarized}
    save(directory/'results.json', result)
    return result


def main(output):
    if output.exists():
        raise ValueError('New isolated audit output required; will not overwrite.')
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only audit cannot use initialized CUDA.')
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    save(output/'protocol.json', {'implementation': IMPLEMENTATION, 'datasets': [s[0] for s in SETTINGS],
        'original_cache': str(ORIGINAL), 'source_cache': str(SOURCE), 'writer_cache': str(WRITER),
        'groups': ['true_positive', 'false_positive', 'miss'], 'patch_targets': ['centre', 'majority'],
        'endpoints_replayed': list(ENDPOINTS), 'new_model_or_selector': False, 'inference_forwards': 0})
    begun = time.perf_counter()
    completed = []
    for setting in SETTINGS:
        run_dataset(setting, output)
        completed.append(setting[0])
        save(output/'suite_results.json', {'status': 'complete' if len(completed) == 8 else 'running',
            'implementation': IMPLEMENTATION, 'completed': completed, 'elapsed_seconds': time.perf_counter()-begun,
            'inference_forwards': 0, 'gpu_allocation': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output_dir)
