"""Evaluate the retained complete model with natural text, without rule tuning."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.fine_alias_view import CONFIG
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.natural_evaluation import CLASS_COUNTS, TOTALS, discover_samples, load_rgb, load_target
from dinotool.natural_rival_reader import retained_scores_chunked
from dinotool.prompts import load_class_specs
from dinotool.rival_fine_full import IMPLEMENTATION as CORE_IMPLEMENTATION, METHODS, PRIMARY
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_geometry_semantic_innovation import SETTINGS
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_vip_reliability import summary, sample_broad
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, tile_reference_coordinates
from eval_rival_alias_count import canonical_indices
from eval_stratified_soft_alias import tile_coordinates
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT, digest


IMPLEMENTATION = 'geometry-physical8-rival-admission-natural-imagenet20-v1-20261003'


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
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coordinates = tile_coordinates(top, left, geometry.device)
                valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
                fine_coordinates = tile_reference_coordinates(coordinates, top, left)
                fine, fine_count, costs = observe_fine(image[:, top:top+512, left:left+512], vip, queries, fine_coordinates, valid)
                operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                ah, aw = min(512, h-top), min(512, w-left)
                for p, bank in banks.items():
                    raw = alias_class_scores((prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0],
                                             bank.parent_indices, bank.class_count)
                    local = raw/.07
                    broad = sample_broad(wide['clean', p], top, left, h, w).reshape_as(local)
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)
                    scores, pair, diagnostics = retained_scores_chunked(local, operator, broad, crops['clean', p], count,
                        fine[p], fine_count, coordinates, fine_coordinates, valid, members, canonical, query.parents, (h, w))
                    del pair
                    diagnostics.update(costs, operator_residual=residual)
                    for method, value in scores.items():
                        source = raw if method == 'Geometry' else value
                        dense = F.interpolate(source.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                        if method == 'Geometry':
                            dense = dense/.07
                        accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[p]['tiles'] += 1
                    for field, value in diagnostics.items():
                        totals[p][field] = totals[p].get(field, 0.)+value
        predictions = {p: {m: accumulators[p, m].finalize(None)[0] for m in METHODS} for p in banks}
    return predictions, {p: {f: v if f == 'tiles' else v/row['tiles'] for f, v in row.items()} for p, row in totals.items()}


def make_models(args):
    commit = subprocess.check_output(['git', '-C', args.upstream_root, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError('Pinned upstream visual/text source changed.')
    specs = load_class_specs(args.vocabulary_config)
    if len(specs) != CLASS_COUNTS[args.dataset] or any(len(s.synonyms) != 20 for s in specs):
        raise ValueError('Locked class count and twenty candidates per class required.')
    checkpoints = make_checkpoints(args)
    geometry = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device),
                            TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), Path(args.upstream_root))
    names = tuple(s.name for s in specs)
    aliases = tuple(alias for spec in specs for alias in spec.synonyms)
    identity = dict(vocabulary_sha256=hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                    checkpoints=checkpoint_manifest(checkpoints), upstream_commit=commit,
                    template_source='openai_imagenet_template', template_count=len(vip.templates))
    cache = Path(args.text_cache)
    if cache.exists():
        stored = torch.load(cache, map_location=args.device, weights_only=True)
        if stored['identity'] != identity or tuple(stored['aliases']) != aliases or tuple(stored['class_names']) != names:
            raise ValueError('Frozen natural text cache identity differs.')
        query = VIPQueries(stored['features'], stored['parents'], names, aliases)
    else:
        query = vip.encode_queries(names, tuple(s.synonyms for s in specs))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        torch.save(dict(identity=identity, features=query.features.cpu(), parents=query.parents.cpu(),
                        aliases=aliases, class_names=names), temporary)
        temporary.rename(cache)
    if not bool(torch.isfinite(query.features).all()):
        raise ValueError('Nonfinite natural text features.')
    # Both views use the same natural templates; the Geometry relation is untouched.
    canonical = torch.zeros(len(aliases), dtype=torch.bool, device=geometry.device)
    canonical[::20] = True
    bank = TCPRTextBank(F.normalize(query.features.float().mean(1), dim=-1), query.parents,
                       canonical, names, aliases)
    bank.validate()
    if any(int((query.parents == c).sum()) != 20 for c in range(bank.class_count)):
        raise ValueError('Exact all20 groups required.')
    return geometry, {args.dataset: bank}, vip, {args.dataset: query}, identity


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Existing output or invalid shard.')
    samples = discover_samples(args.dataset, args.data_root)
    full_count = len(samples)
    if args.max_images:
        samples = samples[:args.max_images]
    keys = [s.key for s in samples]
    selected = samples[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError('Empty natural evaluation shard.')
    print(json.dumps(dict(stage='loading_frozen_model_and_natural_text', dataset=args.dataset,
                          classes=CLASS_COUNTS[args.dataset], aliases_per_class=20)), flush=True)
    started = time.perf_counter()
    geometry, banks, vip, queries, identity = make_models(args)
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    bank = banks[args.dataset]
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={args.dataset: bank.class_names},
        gear=dict(core_implementation=CORE_IMPLEMENTATION, geometry=asdict(geometry.config),
                  observation=asdict(SETTINGS), admission=asdict(CONFIG), upstream_commit=PINNED_COMMIT,
                  local_tiles=[512, 128], fine_physical_pixels_per_token=8,
                  memory_execution='128-query blocks; unaltered alias/rival equations and full Geometry reconstruction',
                  input='Original RGB dimensions; no evaluation-image resize',
                  text='ImageNet80 templates in BOTH local Geometry and contextual views; same aliases',
                  empty_row_repair='Self-Value only on undefined all-masked proxy rows'),
        competitive=None, vocabulary=dict(sha256=identity['vocabulary_sha256'],
            aliases={args.dataset: bank.alias_names}, counts={args.dataset: [20]*bank.class_count}),
        checkpoints=identity['checkpoints'], global_sample_count=len(keys), global_sample_keys_sha256=digest(keys),
        sample_keys=[s.key for s in selected], sample_keys_sha256=digest([s.key for s in selected]),
        num_shards=args.num_shards, shard_index=args.shard_index, config=vars(args),
        full_validation_image_count=full_count, partial_run=bool(args.max_images),
        protocol='Standard class/ignore labels; frozen model-native field of view, not the upstream VIP resize protocol')
    matrices = {m: np.zeros((bank.class_count,)*2, dtype=np.int64) for m in METHODS}
    per_image = {m: [] for m in METHODS}
    totals, ignored = {'tiles': 0}, 0
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_rgb(sample)
        predictions, diagnostics = predict_image(image, geometry, banks, vip, queries, output)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        valid = (target >= 0) & (target < bank.class_count)
        ignored += int((~valid).sum())
        for method in METHODS:
            prediction = predictions[args.dataset][method]
            if prediction.shape != target.shape or np.any(prediction[valid] >= bank.class_count):
                raise ValueError('Prediction dimensions or class mapping differ.')
            encoded = target[valid]*bank.class_count+prediction[valid].astype(np.int64)
            cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, bank.class_count)
            matrices[method] += cm
            per_image[method].append(cm)
        diag = diagnostics[args.dataset]
        totals['tiles'] += diag['tiles']
        for key, value in diag.items():
            if key != 'tiles':
                totals[key] = totals.get(key, 0.)+value*diag['tiles']
        if number == 1 or number % args.progress_every == 0 or number == len(selected):
            result = dict(status='running', processed_images=number, total_images=len(selected), signature=signature,
                metrics={args.dataset: {m: summary(cm, bank.class_names, ignored) for m, cm in matrices.items()}},
                diagnostics={args.dataset: {k: v if k == 'tiles' else v/totals['tiles'] for k, v in totals.items()}},
                wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_masks_used_only_after_prediction=True, target_label_tuning=False)
            save(output/'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(selected),
                miou={m: v['mean_iou_percent'] for m, v in result['metrics'][args.dataset].items()},
                peak_memory_mb=result['peak_cuda_memory_mb'])), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{args.dataset+'__'+m: np.stack(v) for m, v in per_image.items()})
    result.update(status='complete', wall_seconds=time.perf_counter()-started, **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=tuple(TOTALS))
    for name in ('dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir', 'text-cache'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=10)
    main(parser.parse_args())
