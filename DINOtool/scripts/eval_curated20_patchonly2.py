"""Paired old/curated vocabulary evaluation of unchanged bounded PatchOnly2."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_grouped_class_readout import CheckedGroupedCache
from dinotool.bounded_patch_only import IMPLEMENTATION, PRIMARY, PROTOCOL, predict_image
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.geometry_execution import GeometryExecution
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.natural_evaluation import CLASS_COUNTS, discover_samples, load_rgb, load_target
from dinotool.prompts import ClassSpec, REMOTE_SENSING_TEMPLATES
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


ARMS = ('original20', 'curated20')


def exact_specs(path):
    payload = json.loads(Path(path).read_text())
    specs = [ClassSpec(v['name'], tuple(v['synonyms'])) for v in payload['classes']]
    if (any(len(s.synonyms) != 20 or len(set(s.synonyms)) != 20 for s in specs)
            or len({s.name for s in specs}) != len(specs)):
        raise ValueError('Exactly twenty queries per unchanged class required.')
    return specs


def inputs(args):
    old, new = exact_specs(args.original_vocabulary), exact_specs(args.curated_vocabulary)
    if [s.name for s in old] != [s.name for s in new]:
        raise ValueError('Vocabulary class order changed.')
    if args.family == 'natural':
        if len(old) != CLASS_COUNTS[args.dataset]:
            raise ValueError('Natural taxonomy class count changed.')
        samples = discover_samples(args.dataset, args.data_root)
        specs = {args.dataset + '__' + arm: group for arm, group in zip(ARMS, (old, new))}
        image_loader = load_rgb
        mask_loader = lambda sample, p, shape: load_target(sample, args.dataset, shape)
    else:
        samples, old_specs, image_loader, mask_loader = protocol(args, old)
        _, new_specs, _, _ = protocol(args, new)
        specs = {p + '__' + arm: group for arm, groups in zip(ARMS, (old_specs, new_specs))
                 for p, group in groups.items()}
        original_mask_loader = mask_loader
        mask_loader = lambda sample, p, shape: original_mask_loader(sample, p.split('__')[0], shape)
    return samples, specs, image_loader, mask_loader


@torch.inference_mode()
def models(args, specs):
    commit = subprocess.check_output(['git', '-C', args.upstream_root, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError('Pinned source changed.')
    checkpoints = make_checkpoints(args)
    segmenter = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device),
                             TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), Path(args.upstream_root))
    identity = dict(checkpoints=checkpoint_manifest(checkpoints), upstream_commit=commit,
        vocabulary_sha256={arm: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for arm, path in zip(ARMS, (args.original_vocabulary, args.curated_vocabulary))},
        local_templates='ImageNet80' if args.family == 'natural' else list(REMOTE_SENSING_TEMPLATES),
        wide_templates='ImageNet80',
        aliases={p: [list(s.synonyms) for s in group] for p, group in specs.items()})
    cache_path = Path(args.text_cache)
    stored = torch.load(cache_path, map_location=args.device, weights_only=True) if cache_path.exists() else None
    if stored is not None and stored['identity'] != identity:
        raise ValueError('Frozen text-cache identity changed.')
    banks, queries, encoded = {}, {}, {}
    for p, group in specs.items():
        names = tuple(s.name for s in group)
        groups = tuple(s.synonyms for s in group)
        aliases = tuple(alias for g in groups for alias in g)
        if stored is None:
            query = vip.encode_queries(names, groups)
            if args.family == 'natural':
                features = F.normalize(query.features.float().mean(1), dim=-1)
                parents = query.parents
                canonical = torch.zeros(len(aliases), dtype=torch.bool, device=segmenter.device)
                canonical[::20] = True
            else:
                features, parents, canonical = segmenter.backbone.encode_text_aliases(
                    groups, templates=REMOTE_SENSING_TEMPLATES)
            encoded[p] = dict(local=features.cpu(), parents=parents.cpu(), canonical=canonical.cpu(),
                              wide=query.features.cpu())
        else:
            row = stored['encoded'][p]
            features, parents, canonical = row['local'], row['parents'], row['canonical']
            query = VIPQueries(row['wide'], parents, names, aliases)
        bank = TCPRTextBank(features, parents, canonical, names, aliases)
        bank.validate()
        if not torch.equal(bank.parent_indices, query.parents) or not torch.isfinite(bank.features).all():
            raise ValueError('Local/wide words or finite features differ.')
        banks[p], queries[p] = bank, query
    if stored is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.tmp')
        torch.save(dict(identity=identity, encoded=encoded), temporary)
        temporary.rename(cache_path)
    return segmenter, banks, vip, queries, identity


def check_predictions(predictions, shape):
    if any(v[PRIMARY].shape != shape for v in predictions.values()):
        raise RuntimeError('Original-size output restoration failed.')


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError('Existing output or invalid shard; preserve it.')
    samples, specs, load_image, load_mask = inputs(args)
    keys = [s.key for s in samples]
    selected = samples[:1] if args.mode == 'smoke' else samples[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError('Empty shard.')
    output.mkdir(parents=True)
    started = time.perf_counter()
    print(json.dumps(dict(stage='load_model_and_fixed_text', dataset=args.dataset, mode=args.mode)), flush=True)
    geometry, banks, vip, queries, identity = models(args, specs)
    if getattr(args, 'only_protocol', None):
        # Load and verify the complete historical cache before selecting a
        # reference-only replay; never re-encode or slice another text bank.
        if args.only_protocol not in banks:
            raise ValueError('Unknown historical protocol: ' + args.only_protocol)
        banks = {args.only_protocol: banks[args.only_protocol]}
        queries = {args.only_protocol: queries[args.only_protocol]}
    state = frozen_state(geometry, vip)
    execution = GeometryExecution(geometry)
    cache = CheckedGroupedCache()
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=[PRIMARY],
        classes={p: bank.class_names for p, bank in banks.items()},
        gear=dict(geometry=asdict(geometry.config), input_protocol=PROTOCOL,
                  vocabulary_experiment='taxonomy-aware-curated20-v1-20261006',
                  family=args.family, local_templates=identity['local_templates'],
                  wide_templates=identity['wide_templates'], fine_forwards=0,
                  upstream_commit=PINNED_COMMIT),
        competitive=dict(alias_admission='none', fixed_aliases_per_class=20),
        vocabulary=dict(sha256=identity['vocabulary_sha256'],
            aliases={p: bank.alias_names for p, bank in banks.items()},
            counts={p: [20] * bank.class_count for p, bank in banks.items()}),
        checkpoints=identity['checkpoints'], global_sample_count=len(keys),
        global_sample_keys_sha256=digest(keys), num_shards=args.num_shards,
        shard_index=args.shard_index, sample_keys=[s.key for s in selected],
        sample_keys_sha256=digest([s.key for s in selected]), config=vars(args))
    matrices = {p: np.zeros((b.class_count,) * 2, np.int64) for p, b in banks.items()}
    per_image = {p: [] for p in banks}
    ignored = dict.fromkeys(banks, 0)
    totals = {p: dict(tiles=0) for p in banks}
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostics = predict_image(image, execution, banks, vip, queries,
                                                   methods=(PRIMARY,), cache=cache)
        check_predictions(predictions, tuple(image.shape[-2:]))
        for p, diag in diagnostics.items():
            if diag['geometry_encodings'] > 4 or diag['wide_encodings'] > 4 or diag['fine_forwards']:
                raise RuntimeError('Frozen forward budget changed.')
        if args.mode == 'smoke':
            for p in banks:
                single, _ = predict_image(image, execution, {p: banks[p]}, vip, {p: queries[p]},
                                          methods=(PRIMARY,), cache=cache)
                if not np.array_equal(single[p][PRIMARY], predictions[p][PRIMARY]):
                    raise RuntimeError('Paired/singleton output differs: ' + p)
            save(output / 'results.json', dict(status='complete', implementation=IMPLEMENTATION,
                dataset=args.dataset, processed_images=1, total_images=1,
                target_masks_loaded=False, paired_singleton_exact=True, diagnostics=diagnostics,
                text_identity=identity, **check_frozen(state, geometry, vip)))
            return
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            prediction = predictions[p][PRIMARY]
            if np.any(prediction[valid] >= bank.class_count) or np.any(prediction[valid] < 0):
                raise ValueError('Prediction class mapping differs.')
            ignored[p] += int((~valid).sum())
            encoded = target[valid].astype(np.int64) * bank.class_count + prediction[valid]
            cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
            matrices[p] += cm
            per_image[p].append(cm)
            diag = diagnostics[p]
            totals[p]['tiles'] += diag['tiles']
            for field, value in diag.items():
                if field != 'tiles':
                    totals[p][field] = totals[p].get(field, 0.) + value * diag['tiles']
        if number == 1 or number % 10 == 0 or number == len(selected):
            result = dict(status='running', processed_images=number, total_images=len(selected),
                signature=signature, metrics={p: {PRIMARY: summary(cm, banks[p].class_names, ignored[p])}
                                             for p, cm in matrices.items()},
                diagnostics={p: {k: v if k == 'tiles' else v / row['tiles'] for k, v in row.items()}
                             for p, row in totals.items()},
                wall_seconds=time.perf_counter() - started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1048576,
                target_masks_used_only_after_prediction=True, target_label_tuning=False)
            save(output / 'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(selected),
                miou={p: g[PRIMARY]['mean_iou_percent'] for p, g in result['metrics'].items()})), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{p + '__' + PRIMARY: np.stack(v) for p, v in per_image.items()})
    result.update(status='complete', wall_seconds=time.perf_counter() - started,
                  **check_frozen(state, geometry, vip))
    save(output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--family', choices=('remote_sensing', 'natural'), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'data-root', 'upstream-root',
                 'original-vocabulary', 'curated-vocabulary', 'output-dir', 'text-cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--only-protocol')
    main(parser.parse_args())
