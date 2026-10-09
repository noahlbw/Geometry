"""Full-image evaluation of the frozen 20-alias RivalFineHard candidate."""
from contextlib import ExitStack
from dataclasses import asdict, replace
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
from dinotool.fine_reference_admission import fine_crop_positions, fine_patch_features
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_alias_count import canonical_indices
from dinotool.rival_fine_full import IMPLEMENTATION, METHODS, retained_scores, tile_reference_coordinates
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import prepare_wide
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_contribution_alias import crop_from_features
from eval_stratified_soft_alias import tile_coordinates
from eval_vip_official_eight import PINNED_COMMIT
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL / 'results/rival_fine_coupling_20261003'
ORIGINAL = TOOL / 'results/alias_action_capacity_audit_20261003'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def observe_fine(image, vip, queries, coordinates, valid, feature_batch=None):
    h, w = image.shape[-2:]
    positions = fine_crop_positions(h, w)
    count = torch.zeros(512, 512, device=vip.device)
    for top, left, ch, cw in positions:
        count[top:top+ch, left:left+cw] += 1
    count.clamp_min_(1)
    crops = {p: [] for p in queries}
    coverage = torch.zeros(len(valid), device=vip.device)
    layout = vip.backbone.model.visual_model.head.patch_size
    batch = None
    if feature_batch is not None:
        resized_crops = []
        for top, left, ch, cw in positions:
            rgb = F.pad(image[:, top:top+ch, left:left+cw].to(vip.device), (0, 256-cw, 0, 256-ch))
            resized_crops.append(F.interpolate(rgb[None], (512, 512), mode='bilinear', align_corners=False)[0])
        batch = feature_batch(vip, tuple(resized_crops))
        if len(batch) != len(positions):
            raise RuntimeError('Fine feature batch changed crop count.')
    for index, (top, left, ch, cw) in enumerate(positions):
        if batch is None:
            rgb = F.pad(image[:, top:top+ch, left:left+cw].to(vip.device), (0, 256-cw, 0, 256-ch))
            resized = F.interpolate(rgb[None], (512, 512), mode='bilinear', align_corners=False)[0]
            features = fine_patch_features(vip, resized)
        else:
            features = batch[index]
        blank = WideCrop(features.new_empty(1024, 1), features.new_empty(1), top, left, ch, cw, 32, 256)
        _, coefficients = crop_stencil(blank, count, coordinates, (512, 512))
        coverage += coefficients.sum(-1)
        for p, query in queries.items():
            crops[p].append(replace(crop_from_features(features, query, blank), grid_side=32, crop_side=256))
    error = float((coverage[valid]-1).abs().max()) if bool(valid.any()) else 0.
    if error > 1e-6 or vip.backbone.model.visual_model.head.patch_size != layout:
        raise RuntimeError('Fine coverage or original head layout changed.')
    return crops, count, {'fine_forwards': len(positions), 'fine_coverage_max_error': error}


@torch.inference_mode()
def predict_tile(image, top, left, geometry, banks, vip, queries, wide, crops, wide_count):
    h, w = image.shape[-2:]
    prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
    coordinates = tile_coordinates(top, left, geometry.device)
    valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
    fine_coordinates = tile_reference_coordinates(coordinates, top, left)
    fine, fine_count, costs = observe_fine(image[:, top:top+512, left:left+512], vip, queries, fine_coordinates, valid)
    operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    output, diagnostics, fields = {}, {}, {'operator': operator, 'coordinates': coordinates, 'valid': valid}
    for p, bank in banks.items():
        raw = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
                                 bank.parent_indices, bank.class_count)
        local = raw/.07
        broad = sample_broad(wide['clean', p], top, left, h, w).reshape_as(local)
        query = queries[p]
        members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
        canonical = canonical_indices(query.class_names, query.aliases, query.parents)
        output[p], pair, diagnostics[p] = retained_scores(local, operator, broad, crops['clean', p], wide_count,
            fine[p], fine_count, coordinates, fine_coordinates, valid, members, canonical, query.parents, (h, w))
        fields[p+'__local'], fields[p+'__raw'] = local, raw
        diagnostics[p].update(costs, operator_residual=residual)
    return output, diagnostics, fields


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work):
    h, w = image.shape[-2:]
    wide, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
    totals = {p: {'tiles': 0} for p in banks}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for p, bank in banks.items() for m in METHODS}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                scores, diagnostics, fields = predict_tile(image, top, left, geometry, banks, vip, queries, wide, crops, count)
                ah, aw = min(512, h-top), min(512, w-left)
                for p, bank in banks.items():
                    for method, value in scores[p].items():
                        # Retain Geometry's historical interpolation-before-temperature order.
                        source = fields[p+'__raw'] if method == 'Geometry' else value
                        dense = F.interpolate(source.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                              mode='bilinear', align_corners=False)[0]
                        if method == 'Geometry':
                            dense = dense/.07
                        accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(),
                                                    blend[:ah, :aw], left, top)
                    totals[p]['tiles'] += 1
                    for field, value in diagnostics[p].items():
                        totals[p][field] = totals[p].get(field, 0.) + value
        predictions = {p: {m: accumulators[p, m].finalize(None)[0] for m in METHODS} for p in banks}
    return predictions, {p: {field: value if field == 'tiles' else value/row['tiles'] for field, value in row.items()}
                         for p, row in totals.items()}


def frozen_state(geometry, vip):
    return {prefix+n: v.clone() for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
            for n, v in model.model.visual_model.head.state_dict().items()}


def check_frozen(states, geometry, vip):
    frozen = all(not v.requires_grad for model in (geometry.backbone, vip.backbone) for v in model.model.parameters())
    unchanged = all(torch.equal(states[prefix+n], v) for prefix, model in (('g_', geometry.backbone), ('v_', vip.backbone))
                    for n, v in model.model.visual_model.head.state_dict().items())
    if not frozen or not unchanged:
        raise RuntimeError('Frozen model weights changed.')
    return {'weights_frozen': frozen, 'head_weights_unchanged': unchanged}


@torch.inference_mode()
def replay(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing replay output.')
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Historical checkpoints changed.')
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    errors = {}
    started = time.perf_counter()
    for i, key in enumerate(prior['sample_keys']):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        wide, crops, _, count = prepare_wide(image, vip, {'clean': (banks, queries)})
        scores, _, fields = predict_tile(image, 0, 0, geometry, banks, vip, queries, wide, crops, count)
        with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as original, np.load(
                PREVIOUS/args.dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as old:
            errors[key] = {'operator': float((fields['operator']-torch.from_numpy(original['operator']).to(args.device)).abs().max())}
            for p in banks:
                errors[key][p+'__local'] = float((fields[p+'__local']-torch.from_numpy(original[p+'__local']).to(args.device)).abs().max())
                for m in METHODS:
                    errors[key][p+'__'+m] = float((scores[p][m]-torch.from_numpy(old['clean__'+p+'__'+m+'__scores']).to(args.device)).abs().max())
        if errors[key]['operator'] > 1e-10 or max(v for k, v in errors[key].items() if k != 'operator') > 1e-4:
            raise RuntimeError('Historical score replay failed: '+str(errors[key]))
    # One complete image also exercises nonzero offsets and accumulator coverage, without masks.
    first = lookup[prior['sample_keys'][0]]
    image = load_image(first.image_path if args.dataset == 'loveda' else first)
    predictions, diagnostics = predict_image(image, geometry, banks, vip, queries, output)
    if any(pred.shape != tuple(image.shape[-2:]) for ps in predictions.values() for pred in ps.values()):
        raise RuntimeError('Full-image prediction coverage differs.')
    save(output/'results.json', {'status': 'complete', 'implementation': IMPLEMENTATION, 'sample_keys': prior['sample_keys'],
        'processed_images': len(errors), 'total_images': len(errors), 'errors': errors, 'full_image_diagnostics': diagnostics,
        'target_masks_loaded': False, 'wall_seconds': time.perf_counter()-started, **check_frozen(states, geometry, vip)})


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError('Existing output or invalid shard.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    keys = [s.key for s in samples]
    selected = samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise RuntimeError('Empty shard or duplicate global samples.')
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    signature = {'implementation': IMPLEMENTATION, 'dataset': args.dataset, 'methods': METHODS,
        'classes': {p: bank.class_names for p, bank in banks.items()},
        'gear': {'geometry': asdict(geometry.config), 'observation': asdict(SETTINGS), 'admission': asdict(CONFIG),
                 'upstream_commit': PINNED_COMMIT, 'local_tiles': [512, 128], 'fine_physical_pixels_per_token': 8,
                 'fine_tile_policy': 'Four physical256 crops per local512 tile, including edge padding; original fixed rule.'},
        'competitive': None, 'vocabulary': {'sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
            'aliases': {p: bank.alias_names for p, bank in banks.items()}, 'counts': {p: [20]*bank.class_count for p, bank in banks.items()}},
        'checkpoints': checkpoint_manifest(checkpoints), 'global_sample_count': len(keys), 'global_sample_keys_sha256': digest(keys),
        'sample_keys': [s.key for s in selected], 'sample_keys_sha256': digest([s.key for s in selected]),
        'num_shards': args.num_shards, 'shard_index': args.shard_index, 'config': vars(args),
        'note': 'Frozen complete model; local Geometry20 unchanged, rival-conditioned wide alias rejection; no target-label tuning.'}
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    transitions = {p: {m: np.zeros((bank.class_count,)*3, np.int64) for m in METHODS[1:]} for p, bank in banks.items()}
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    ignored, totals = dict.fromkeys(banks, 0), {p: {'tiles': 0} for p in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostics = predict_image(image, geometry, banks, vip, queries, output)
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count+predictions[p][method][valid]
                cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                matrices[p][method] += cm
                per_image[p+'__'+method].append(cm)
                if method != 'Geometry':
                    transitions[p][method] += transition_counts(predictions[p]['Geometry'], predictions[p][method], target, bank.class_count)
            tiles = diagnostics[p]['tiles']
            totals[p]['tiles'] += tiles
            for field, value in diagnostics[p].items():
                if field != 'tiles':
                    totals[p][field] = totals[p].get(field, 0.) + value*tiles
        result = {'status': 'running', 'processed_images': number, 'total_images': len(selected), 'signature': signature,
            'metrics': {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in ms.items()} for p, ms in matrices.items()},
            'transitions': {p: {m: {'counts': cm.tolist(), **transition_summary(cm)} for m, cm in ms.items()} for p, ms in transitions.items()},
            'diagnostics': {p: {f: v if f == 'tiles' else v/row['tiles'] for f, v in row.items()} for p, row in totals.items()},
            'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576,
            'target_masks_used_only_after_prediction': True}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': number, 'total': len(selected),
            'miou': {p: ms['RivalFineHard_Exact']['mean_iou_percent'] for p, ms in result['metrics'].items()}}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result.update(status='complete', **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    is_replay = '--replay' in sys.argv
    if is_replay:
        sys.argv.remove('--replay')
    (replay if is_replay else main)(parse_args())
