"""Freeze an RGB-only corpus reference before the developed alias pilot."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.bounded_physical_coupling import geometry_windows, resize_geometry
from dinotool.cross_image_alias_reference import (DIAGNOSTICS, IMPLEMENTATION, METHODS,
    OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, PROTOCOL, REFERENCE_NULL, RankReference,
    canonical_reference, predict_image)
from dinotool.model import checkpoint_manifest
from dinotool.one_sided_alias_audit import (HARD as PREVIOUS_HARD, PRIMARY as PREVIOUS_SOFT,
    OBSERVATION_MEAN as PREVIOUS_MEAN, predict_image as previous_predict)
from dinotool.rival_alias_count import canonical_indices
from dinotool.rival_fine_full import tile_reference_coordinates
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, observe_fine, save, tile_coordinates


ORIGINAL_INPUTS = evaluator.inputs
CALIBRATION_COUNT = 64


def panel_indices(total, dataset):
    return list(range(total)) if dataset == 'udd5' else [round(i*(total-1)/7) for i in range(8)]


def reference_folder(args):
    return Path(args.output_dir).parents[2]/'calibrate'/args.dataset/'s0'


def select_calibration(samples, args):
    heldout = panel_indices(len(samples), args.dataset)
    excluded = sorted(set(heldout+[0, len(samples)//2, len(samples)-1]))
    if args.dataset == 'udd5':
        folder = Path(args.data_root)/'train/src'
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        candidates = [replace(samples[0], image_path=p, mask_path=Path('/MASKS_FORBIDDEN'), key='train/'+p.name) for p in paths]
        origin = 'UDD5 train RGB; no train masks'
    else:
        candidates = [s for i, s in enumerate(samples) if i not in excluded]
        origin = 'held-out evaluation RGB; transductive'
    if len(candidates) < CALIBRATION_COUNT:
        raise RuntimeError('Insufficient disjoint64 RGB reference images: '+args.dataset)
    selected = [candidates[round(i*(len(candidates)-1)/(CALIBRATION_COUNT-1))] for i in range(CALIBRATION_COUNT)]
    forbidden = {samples[i].image_path.resolve() for i in excluded}
    actual = {s.image_path.resolve() for s in selected}
    if len(actual) != CALIBRATION_COUNT or forbidden & actual:
        raise RuntimeError('Reference RGB source duplicates or overlaps developed evaluation/timing images.')
    return selected, dict(origin=origin, sample_keys=[s.key for s in selected],
        image_paths=[str(s.image_path.resolve()) for s in selected],
        evaluation_and_timing_keys=[samples[i].key for i in excluded],
        evaluation_and_timing_paths=[str(samples[i].image_path.resolve()) for i in excluded],
        calibration_evaluation_disjoint=True)


@torch.inference_mode()
def calibrate(args):
    output, samples, load_image, _, geometry, banks, vip, queries, checkpoints, old = ORIGINAL_INPUTS(args)
    selected, source = select_calibration(samples, args)
    states = frozen_state(geometry, vip)
    execution = FineCoverageExecution(cached=True, burst=True)
    responses = {p: [] for p in queries}
    aliases = {p: dict(class_names=list(q.class_names), aliases=list(q.aliases), parents=q.parents.tolist()) for p, q in queries.items()}
    canonical = {p: canonical_indices(q.class_names, q.aliases, q.parents) for p, q in queries.items()}
    rows, actual_forwards = [], 0
    started = time.perf_counter()
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    for index, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        resized = resize_geometry(image)
        image_rows, calls = dict.fromkeys(queries, 0), 0
        reader = execution.reader(vip)
        for top, left in geometry_windows(*resized.shape[-2:]):
            coordinates = tile_coordinates(top, left, vip.device)
            valid = (coordinates[:, 0] < resized.shape[-2]) & (coordinates[:, 1] < resized.shape[-1])
            fine_coordinates = tile_reference_coordinates(coordinates, top, left)
            crops, _, costs = observe_fine(resized[:, top:top+512, left:left+512], vip, queries,
                fine_coordinates, valid, feature_batch=reader)
            calls += costs['fine_forwards']
            for p, group in crops.items():
                for crop in group:
                    axis = torch.arange(4, 32, 8, device=vip.device)
                    y, x = torch.meshgrid(axis, axis, indexing='ij')
                    keep = ((y+.5)*8 < crop.actual_height) & ((x+.5)*8 < crop.actual_width)
                    ids = (y*32+x)[keep].flatten()
                    responses[p].append(crop.alias_logits[ids].cpu())
                    image_rows[p] += len(ids)
        if not 1 <= calls <= 16 or reader.calls != calls:
            raise RuntimeError('Calibration per-image visual cap violated.')
        actual_forwards += calls
        rows.append(dict(sample_key=sample.key, resized_shape=list(resized.shape[-2:]), rows=image_rows, fine_encodings=calls))
        if index == 1 or index % 8 == 0:
            save(output/'results.json', dict(status='running', processed_images=index, total_images=CALIBRATION_COUNT,
                implementation=IMPLEMENTATION, target_masks_loaded=False, source=source))
            print(json.dumps(dict(dataset=args.dataset, calibration_images=index, total=CALIBRATION_COUNT)), flush=True)
    tensors = {}
    for p in queries:
        raw = torch.cat(responses[p])
        tensors[p+'__raw'] = raw
        tensors[p+'__reference'] = canonical_reference(raw, canonical[p].cpu())
    torch.save(tensors, output/'reference.pt')
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        processed_images=CALIBRATION_COUNT, total_images=CALIBRATION_COUNT,
        target_masks_loaded=False, aliases=aliases, source=source, rows=rows,
        source_vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        calibration_seconds=time.perf_counter()-started, actual_fine_encodings=actual_forwards,
        calibration_geometry_encodings=0, peak_shared_resident_mib=torch.cuda.max_memory_allocated()/1048576,
        response_rows={p: int(tensors[p+'__raw'].shape[0]) for p in queries},
        pseudo_class_mass={p: tensors[p+'__reference'].sum(0).tolist() for p in queries},
        tensor_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
        execution=execution.report(), **check_frozen(states, geometry, vip)))


def load_references(args, queries, old, *, controls=True):
    folder = reference_folder(args)
    row = json.loads((folder/'results.json').read_text())
    if (row['status'] != 'complete' or row['implementation'] != IMPLEMENTATION or row['target_masks_loaded']
            or row['processed_images'] != CALIBRATION_COUNT or row['total_images'] != CALIBRATION_COUNT
            or not row['weights_frozen'] or not row['head_weights_unchanged']
            or row['source_vocabulary'] != old['signature']['vocabulary']
            or row['checkpoints'] != old['signature']['checkpoints']):
        raise RuntimeError('Complete frozen image-only reference required before inference.')
    source = row['source']
    if (len(set(source['image_paths'])) != CALIBRATION_COUNT
            or set(source['image_paths']) & set(source['evaluation_and_timing_paths'])):
        raise RuntimeError('Reference/evaluation disjointness failed.')
    tensors = torch.load(folder/'reference.pt', map_location='cpu', weights_only=True)
    references, shuffled = {}, {}
    started = time.perf_counter()
    for p, query in queries.items():
        actual = dict(class_names=list(query.class_names), aliases=list(query.aliases), parents=query.parents.tolist())
        if actual != row['aliases'][p]:
            raise RuntimeError('Reference alias identity/layout differs.')
        raw, pseudo = tensors[p+'__raw'].to(query.parents.device), tensors[p+'__reference'].to(query.parents.device)
        references[p] = RankReference.build(raw, pseudo)
        if controls:
            generator = torch.Generator().manual_seed(CONFIG_SEED)
            permutation = torch.randperm(len(raw), generator=generator).to(raw.device)
            shuffled[p] = RankReference.build(raw, pseudo[permutation])
    return references, shuffled, dict(calibration=row, index_setup_seconds=time.perf_counter()-started,
        index_tensor_bytes=sum(t.numel()*t.element_size() for bank in references.values()
            for t in (bank.values, bank.prefix, bank.masses)))


CONFIG_SEED = 20261006


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, old = ORIGINAL_INPUTS(args)
    states = frozen_state(geometry, vip)
    references, shuffled, provenance = load_references(args, queries, old)
    image = load_image(samples[0].image_path if args.dataset == 'loveda' else samples[0])
    execution = FineCoverageExecution(cached=True, burst=True)
    actual, diagnostics = predict_image(image, geometry, banks, vip, queries, references=references,
        shuffled_references=shuffled, execution=execution)
    previous, _ = previous_predict(image, geometry, banks, vip, queries,
        methods=(*METHODS[:3], PREVIOUS_MEAN, PREVIOUS_SOFT, PREVIOUS_HARD), execution=execution)
    replay = (*[(m, m) for m in METHODS[:3]], (OBSERVATION_MEAN, PREVIOUS_MEAN),
              (OLD_SOFT, PREVIOUS_SOFT), (OLD_HARD, PREVIOUS_HARD))
    for p in banks:
        for new, old_name in replay:
            if not np.array_equal(actual[p][new], previous[p][old_name]):
                raise RuntimeError('Changed retained endpoint: '+p+'/'+new)
    single, _ = predict_image(image, geometry, banks, vip, queries, references=references,
        methods=(PRIMARY,), execution=execution)
    if any(not np.array_equal(single[p][PRIMARY], actual[p][PRIMARY]) for p in banks):
        raise RuntimeError('Primary singleton differs.')
    for row in diagnostics.values():
        if (not 0 < row['fine_forwards'] <= 16 or row['geometry_encodings'] > 4 or row['wide_encodings'] > 4
                or row['canonical_risk_max'] != 0 or row['positive_directed_delta_max'] > 1e-12
                or row['fine_coverage_max_error'] > 1e-6 or row['matched_norm_relative_error'] > 1e-10):
            raise RuntimeError('Source or reference control invariant failed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=samples[0].key, original_endpoints_exact=True, same_information_mean_exact=True,
        strong_soft_exact=True, strong_hard_exact=True, singleton_primary_exact=True,
        target_masks_loaded=False, diagnostics=diagnostics, reference_provenance=provenance,
        execution=execution.report(), **check_frozen(states, geometry, vip)))


def run(args):
    execution = FineCoverageExecution(cached=True, burst=True)
    state = {}
    def inputs(current):
        values = panel_inputs(current) if args.mode == 'full' else ORIGINAL_INPUTS(current)
        state['references'], state['shuffled'], state['provenance'] = load_references(
            current, values[7], values[-1], controls=args.mode == 'full')
        return values
    def predictor(*values, **kwargs):
        return predict_image(*values, **kwargs, references=state['references'],
            shuffled_references=state['shuffled'], execution=execution)
    evaluator.inputs = inputs
    if args.mode == 'full':
        evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION, methods=METHODS,
            view_protocol=PROTOCOL, competitive=dict(alias_admission=PROTOCOL['alias_admission'],
                fixed_aliases_per_class=20, target_labels_used_for_selection=False,
                frozen_primary=PRIMARY, fitted_parameters=0, calibration_images=CALIBRATION_COUNT,
                independent_validation=False))
    else:
        evaluator.benchmark(args, predictor=predictor, implementation=IMPLEMENTATION,
            view_protocol=PROTOCOL, names=(METHODS[2], OBSERVATION_MEAN, OLD_SOFT, PRIMARY, 'VIP_All20'))
    result = json.loads((Path(args.output_dir)/'results.json').read_text())
    result.update(execution=execution.report(), reference_provenance=state['provenance'])
    save(Path(args.output_dir)/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('calibrate', 'smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    {'calibrate': calibrate, 'smoke': smoke, 'full': run, 'benchmark': run}[args.mode](args)
