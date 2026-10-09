"""Finite per-dataset static word/parameter search and frozen full evaluation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import eval_development_readout as base
from dinotool.natural_static_search import (IMPLEMENTATION, StaticProfile, StaticReader,
    BIASES, THRESHOLDS, projected, stage_profiles, rank)
from dinotool.development_readout import threshold_histogram, biased_probabilities
from dinotool.natural_wide_resolution import replace_wide
from dinotool.natural_text_adaptation import CLASS_FILES
from dinotool.vip_official_adapter import upstream_aliases, _load_upstream
from eval_natural_text_adaptation import official_options, official_prediction
from eval_geometry_vip_reliability import summary
from eval_rival_fine_full import save, frozen_state, check_frozen


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    return value


@torch.inference_mode()
def load_models(args, entry, needed):
    from dinotool.finite_vip_observer import FiniteVIPObserver
    from dinotool.model import DINOTextSegmenter
    from dinotool.geometry_execution import GeometryExecution
    from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
    from dinotool.vip_official_adapter import VIPQueries
    from eval_stride_ov_loveda_e1 import make_checkpoints
    from eval_vip_official_eight import PINNED_COMMIT
    import subprocess
    if subprocess.check_output(['git', '-C', args.upstream_root, 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise RuntimeError('Pinned VIP source differs.')
    checkpoints = make_checkpoints(args)
    geometry = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device), TCPRConfig(geometry_depth=2, maximum_aliases_per_class=20))
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), Path(args.upstream_root))
    original = torch.load(args.original_cache, map_location=args.device, weights_only=True)
    previous = torch.load(entry['previous_text_cache'], map_location=args.device, weights_only=True)
    path = Path(args.suite_root) / 'text_cache' / (args.dataset + '.pt')
    identity = dict(banks=entry['banks'], checkpoints=base.checkpoint_manifest(checkpoints), upstream_commit=PINNED_COMMIT)
    stored = torch.load(path, map_location=args.device, weights_only=True) if path.exists() else None
    if stored is not None and stored['identity'] != identity:
        raise RuntimeError('Frozen text-cache configuration changed.')
    encoded = {} if stored is None else stored['encoded']
    prompts = _load_upstream(Path(args.upstream_root) / 'prompts/imagenet_template.py', 'static_search_templates')
    banks, queries = {}, {}
    for name in needed:
        record = entry['banks'][name]
        names = tuple(c['name'] for c in record['classes'])
        groups = tuple(tuple(c['synonyms']) for c in record['classes'])
        aliases = tuple(a for g in groups for a in g)
        if name == 'original':
            row = original['encoded'][args.dataset + '__original20']
            if original['identity']['aliases'][args.dataset + '__original20'] != [list(g) for g in groups]:
                raise RuntimeError('Original text-bank identity changed.')
        elif name in encoded:
            row = encoded[name]
        elif name in previous['encoded'] and previous['identity']['banks'].get(name) == record:
            row = previous['encoded'][name]
        else:
            vip.templates = prompts.get_text_template('openai_imagenet_template' if record['template'] == 'ImageNet80' else record['template'])
            query = vip.encode_queries(names, groups)
            parents = query.parents
            canonical = torch.zeros(len(aliases), dtype=torch.bool, device=args.device)
            canonical[torch.tensor(np.cumsum([0] + [len(g) for g in groups[:-1]]), device=args.device)] = True
            row = dict(local=torch.nn.functional.normalize(query.features.float().mean(1), dim=-1),
                parents=parents, canonical=canonical, wide=query.features)
        encoded[name] = move(row, 'cpu')
        row = move(row, args.device)
        bank = TCPRTextBank(row['local'], row['parents'], row['canonical'], names, aliases)
        bank.validate()
        banks[name] = bank
        queries[name] = VIPQueries(row['wide'], row['parents'], names, aliases)
        print(json.dumps(dict(stage='text_bank', dataset=args.dataset, bank=name, queries=len(aliases))), flush=True)
    if stored is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(identity=identity, encoded=encoded), temporary)
        temporary.rename(path)
    return GeometryExecution(geometry), vip, banks, queries, checkpoints


def observation_pair(image, geometry, vip, strengths):
    # Temporary function binding changes only the search's requested head scalar.
    original = base.projected
    base.projected = projected
    try:
        source = base.observations(image, geometry, vip, strengths)
    finally:
        base.projected = original
    changed = replace_wide(source, image, vip)
    return {'long448': source, 'natural_short336_cap672': changed}


def matrix(prediction, target, classes, columns=None):
    columns = classes if columns is None else columns
    valid = (target >= 0) & (target < classes)
    pred = prediction[valid].astype(np.int64)
    if columns > classes:
        pred = np.where(pred == 255, classes, pred)
    if np.any(pred < 0) or np.any(pred >= columns):
        raise RuntimeError('Invalid prediction category.')
    return np.bincount(target[valid].astype(np.int64) * columns + pred, minlength=classes * columns).reshape(classes, columns)


@torch.inference_mode()
def main(args):
    root, output = Path(args.suite_root), Path(args.output_dir)
    if output.exists() and not (args.resume_search and args.mode == 'search'):
        raise RuntimeError('Existing output is preserved.')
    if args.resume_search and args.mode == 'search':
        if (output / 'selection.json').exists():
            raise RuntimeError('Completed selection is preserved.')
        previous = output / 'results.json'
        backup = output / 'results.before_resume.json'
        if previous.exists() and not backup.exists():
            shutil.copyfile(previous, backup)
    manifest = json.loads((root / 'protocol.json').read_text())
    entry = manifest['datasets'][args.dataset]
    validation, samples, load_image, load_mask = base.load_samples(args, entry)
    selection = json.loads((root / 'search' / args.dataset / 'selection.json').read_text()) if args.mode == 'full' else None
    selected = StaticProfile(**(selection['profile'] if selection else entry['incumbent']['profile']))
    needed = tuple(entry['banks']) if args.mode == 'search' else (selected.bank,)
    geometry, vip, banks, queries, checkpoints = load_models(args, entry, needed)
    state = frozen_state(geometry, vip)
    residual = None
    if entry.get('residual_identity') is not None:
        saved = torch.load(root / 'residual_cache.pt', map_location=args.device, weights_only=True)
        if saved['identity'] != entry['residual_identity']:
            raise RuntimeError('Frozen residual ontology identity changed.')
        residual = saved['wide']
    reader = StaticReader(banks, queries, entry['background_index'], residual)
    names = next(iter(banks.values())).class_names
    classes, background = len(names), entry['background_index']
    output.mkdir(parents=True, exist_ok=args.resume_search)
    started = time.perf_counter()
    if args.mode == 'search':
        source_dir = root / 'observation_cache' / args.dataset
        source_dir.mkdir(parents=True, exist_ok=True)
        lookup = {s.key: s for s in samples}
        coarse_keys = entry['development_keys'][:entry['coarse_count']]
        pool = [selected]
        history = []
        for stage in ('words', 'readout', 'calibration', 'revisit_words', 'refinement'):
            saved_stage = output / ('stage_' + stage + '.json')
            if args.resume_search and saved_stage.exists():
                record = json.loads(saved_stage.read_text())
                history.append(record)
                winners = [StaticProfile(**p) for p in record['top_profiles']]
                pool.extend(winners)
                selected = winners[0]
                continue
            candidates = tuple(dict.fromkeys(pool)) if stage == 'refinement' else stage_profiles(stage, selected, needed, background, residual is not None)
            keys = entry['development_keys'] if stage == 'refinement' else coarse_keys
            biases = BIASES if background is not None else (0.,)
            histograms = np.zeros((len(candidates), len(biases), len(THRESHOLDS) + 1, classes, classes), np.int64)
            for n, key in enumerate(keys, 1):
                sample = lookup[key]
                cache_file = source_dir / (key + '.pt')
                if cache_file.exists():
                    pair = move(torch.load(cache_file, map_location='cpu', weights_only=True), args.device)
                else:
                    image = load_image(sample)
                    pair = observation_pair(image, geometry, vip, ('original', .5, 1., 2., 3., 4.))
                    temporary = cache_file.with_suffix('.tmp')
                    torch.save(move(pair, 'cpu'), temporary)
                    temporary.rename(cache_file)
                target = load_mask(sample, pair['long448']['output_size'])
                target_tensor = torch.as_tensor(target, device=args.device)
                caches = {p: {} for p in pair}
                for index, profile in enumerate(candidates):
                    probability = reader.probabilities(pair[profile.wide_policy], profile, caches[profile.wide_policy])
                    for j, bias in enumerate(biases):
                        histograms[index, j] += threshold_histogram(probability, target_tensor, background, bias, THRESHOLDS)
                if n == 1 or n % 4 == 0 or n == len(keys):
                    save(output / 'results.json', dict(status='running', dataset=args.dataset, stage=stage,
                        processed_images=n, total_images=len(keys), profiles=len(candidates),
                        wall_seconds=time.perf_counter() - started))
                    print(json.dumps(dict(dataset=args.dataset, stage=stage, processed=n, total=len(keys), profiles=len(candidates))), flush=True)
                del pair, caches, target_tensor
            ranked = rank(histograms, candidates, background)
            winners = []
            for row in ranked:
                profile = StaticProfile(**row['profile'])
                if profile not in winners:
                    winners.append(profile)
                if len(winners) == 3:
                    break
            pool.extend(winners)
            selected = winners[0]
            history.append(dict(stage=stage, images=len(keys), evaluated_profiles=len(candidates),
                winner=ranked[0], top_profiles=[p.record() for p in winners]))
            save(output / ('stage_' + stage + '.json'), history[-1])
        # Copy: history's last winner must not recursively contain history itself.
        choice = dict(history[-1]['winner'])
        choice.update(dataset=args.dataset, development_source=entry['development_source'],
            development_keys=entry['development_keys'], heldout_keys=entry['heldout_keys'],
            coarse_keys=coarse_keys, search_history=history, target_label_tuning=True,
            automatic_parameter_selection_at_inference=False, fixed_backbones=True)
        save(output / 'selection.json', choice)
        save(output / 'results.json', dict(status='complete', processed_images=len(samples), total_images=len(samples),
            selection=choice, wall_seconds=time.perf_counter() - started, **check_frozen(state, geometry, vip)))
        return

    # The official comparator is kept at its own published repository settings.
    upstream = Path(args.upstream_root)
    settings, template = official_options(upstream, args.dataset)
    groups = upstream_aliases(upstream / 'configs' / ('cls_' + CLASS_FILES[args.dataset] + '.txt'))
    prompt = _load_upstream(upstream / 'prompts/imagenet_template.py', 'static_official_prompts')
    old_templates = vip.templates
    vip.templates = prompt.get_text_template(template)
    official = vip.encode_queries(names, groups)
    vip.templates = old_templates
    methods = ('StaticTuned', 'VIP_Official_Finite')
    vip_columns = classes + int(not settings.background and settings.prob_thd > 0)
    matrices = {methods[0]: np.zeros((classes, classes), np.int64), methods[1]: np.zeros((classes, vip_columns), np.int64)}
    arrays = {m: [] for m in methods}
    ignored = 0
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=methods, classes={args.dataset: names},
        gear=dict(profile=selected.record(), background_bias=selection['background_bias'],
            background_threshold=selection['background_threshold'], observation_cap=[4, 4, 0],
            automatic_parameter_selection_at_inference=False), competitive=None,
        vocabulary=entry['banks'][selected.bank], checkpoints=base.checkpoint_manifest(checkpoints),
        global_sample_count=len(validation), global_sample_keys_sha256=base.digest([s.key for s in validation]),
        sample_keys=[s.key for s in samples], sample_keys_sha256=base.digest([s.key for s in samples]),
        num_shards=args.num_shards, shard_index=args.shard_index, config=vars(args))
    tiles = 0
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        old = base.projected
        base.projected = projected
        try:
            source = base.observations(image, geometry, vip, (selected.strength,), wide_policy=selected.wide_policy)
        finally:
            base.projected = old
        probability = reader.probabilities(source, selected, {})
        probability = biased_probabilities(probability, selection['background_bias'], background)
        confidence, prediction = probability.max(0)
        if background is not None:
            prediction = prediction.masked_fill(confidence < selection['background_threshold'], background)
        official_pred, _ = official_prediction(image, vip, official, settings)
        target = load_mask(sample, source['output_size'])
        ignored += int(((target < 0) | (target >= classes)).sum())
        for method, pred in zip(methods, (prediction.cpu().numpy(), official_pred)):
            cm = matrix(pred, target, classes, matrices[method].shape[1])
            matrices[method] += cm
            arrays[method].append(cm)
        tiles += len(source['local'])
        if number == 1 or number % 10 == 0 or number == len(samples):
            row = dict(status='running', processed_images=number, total_images=len(samples), signature=signature,
                metrics={args.dataset: {m: summary(cm, names, ignored) for m, cm in matrices.items()}},
                diagnostics={args.dataset: dict(tiles=tiles, fine_forwards=0)},
                wall_seconds=time.perf_counter() - started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_label_tuning=True, inference_uses_target_labels=False, backbone_training=False)
            save(output / 'results.json', row)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(samples))), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset + '__' + m: np.stack(v) for m, v in arrays.items()})
    row.update(status='complete', **check_frozen(state, geometry, vip))
    save(output / 'results.json', row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('dataset', 'data-root', 'suite-root', 'output-dir', 'original-cache', 'dinov3-repo', 'checkpoint-dir', 'upstream-root'):
        p.add_argument('--' + option, required=True)
    p.add_argument('--mode', choices=('search', 'full'), required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--sample-seed', type=int, default=20260923)
    p.add_argument('--vdd-ontology', default='official')
    p.add_argument('--resume-search', action='store_true')
    main(p.parse_args())
