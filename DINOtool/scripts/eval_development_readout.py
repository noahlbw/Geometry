"""Bounded fixed-weight word/readout search, followed by frozen paired validation."""
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

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.bounded_patch_only import PRIMARY as REFERENCE, predict_image as retained_predict
from dinotool.bounded_physical_coupling import geometry_windows, resize_geometry
from dinotool.development_readout import (IMPLEMENTATION, Profile, profiles, projected,
    coupled, threshold_histogram, biased_probabilities, select, BACKGROUND_BIASES, THRESHOLDS)
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.geometry_execution import GeometryExecution
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.gear_ov import _crop_at
from dinotool.inference import hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.natural_evaluation import NaturalSample, discover_samples, load_rgb, load_target
from dinotool.prompts import ClassSpec, REMOTE_SENSING_TEMPLATES
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, VIPSettings, _load_upstream
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_rival_fine_full import check_frozen, frozen_state, save, tile_coordinates
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


METHODS = ('Reference', 'Tuned')


def load_samples(args, entry):
    original = [ClassSpec(c['name'], tuple(c['synonyms'])) for c in entry['banks']['original']['classes']]
    if entry['family'] == 'natural':
        validation = discover_samples(args.dataset, args.data_root)
        loader = load_rgb
        target_loader = lambda s, shape: load_target(s, args.dataset, shape)
    else:
        validation, specs, loader, masks = protocol(args, original)
        if len(specs) != 1:
            raise ValueError('This focused development suite does not tune LoveDA protocols.')
        p = next(iter(specs))
        target_loader = lambda s, shape: masks(s, p, shape)
    lookup = {s.key: s for s in validation}
    if args.mode == 'search' and entry['development_source'] == 'ADE_training':
        root = Path(args.data_root)
        selected = [NaturalSample(k, root / 'images/training' / (k + '.jpg'),
            root / 'annotations/training' / (k + '.png')) for k in entry['development_keys']]
    elif args.mode == 'search':
        selected = [lookup[k] for k in entry['development_keys']]
    elif args.mode == 'smoke':
        selected = validation[:1]
    else:
        selected = validation[args.shard_index::args.num_shards]
    return validation, selected, loader, target_loader


@torch.inference_mode()
def load_models(args, entry, needed):
    commit = subprocess.check_output(['git', '-C', args.upstream_root, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError('Pinned upstream changed.')
    checkpoints = make_checkpoints(args)
    geometry = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device),
        TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), Path(args.upstream_root))
    old = torch.load(args.original_cache, map_location=args.device, weights_only=True)
    cache = Path(args.suite_root) / 'text_cache' / (args.dataset + '.pt')
    input_identity = dict(banks=entry['banks'], checkpoints=checkpoint_manifest(checkpoints), upstream_commit=commit)
    extra = torch.load(cache, map_location=args.device, weights_only=True) if cache.exists() else None
    if extra is not None and extra['identity'] != input_identity:
        raise ValueError('Frozen bank cache differs.')
    encoded = {} if extra is None else extra['encoded']
    banks, queries = {}, {}
    for name in needed:
        classes = entry['banks'][name]['classes']
        names = tuple(c['name'] for c in classes)
        groups = tuple(tuple(c['synonyms']) for c in classes)
        aliases = tuple(a for g in groups for a in g)
        if name == 'original':
            original_key = getattr(args, 'original_cache_key', args.dataset + '__original20')
            row = old['encoded'][original_key]
            if old['identity']['aliases'][original_key] != [list(g) for g in groups]:
                raise ValueError('Original cached words differ.')
        elif name in encoded:
            row = encoded[name]
        else:
            original_templates = vip.templates
            template = entry['banks'][name]['template']
            if template == 'seg_template':
                module = _load_upstream(Path(args.upstream_root) / 'prompts/imagenet_template.py', 'development_prompts')
                vip.templates = module.get_text_template(template)
            try:
                query = vip.encode_queries(names, groups)
            finally:
                vip.templates = original_templates
            if template == 'RS6':
                local, parents, canonical = geometry.backbone.encode_text_aliases(groups, templates=REMOTE_SENSING_TEMPLATES)
            else:
                local = F.normalize(query.features.float().mean(1), dim=-1)
                parents = query.parents
                canonical = torch.zeros(len(aliases), dtype=torch.bool, device=geometry.device)
                offsets = np.cumsum([0] + [len(g) for g in groups[:-1]])
                canonical[torch.tensor(offsets, device=geometry.device)] = True
            row = dict(local=local, parents=parents, canonical=canonical, wide=query.features)
            encoded[name] = {k: v.cpu() for k, v in row.items()}
        query = VIPQueries(row['wide'].to(geometry.device), row['parents'].to(geometry.device), names, aliases)
        bank = TCPRTextBank(row['local'].to(geometry.device), query.parents,
            row['canonical'].to(geometry.device), names, aliases)
        bank.validate()
        banks[name], queries[name] = bank, query
    if extra is None and args.mode in ('smoke', 'search'):
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        torch.save(dict(identity=input_identity, encoded=encoded), temporary)
        temporary.rename(cache)
    return GeometryExecution(geometry), vip, banks, queries, checkpoints


@torch.inference_mode()
def observations(image, geometry, vip, strengths, *, wide_policy='long448'):
    if wide_policy not in ('long448','natural_short336_cap672'):
        raise ValueError('Declared bounded wide-view policy required.')
    resized = resize_geometry(image)
    h, w = resized.shape[-2:]
    local = []
    for top, left in geometry_windows(h, w):
        prepared = geometry.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
        coordinates = tile_coordinates(top, left, geometry.device)
        valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
        operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        features = {}
        with geometry.backbone._autocast():
            for strength in strengths:
                features[strength] = projected(geometry.backbone.model.visual_model.head, prepared, strength)[0]
        local.append(dict(top=top, left=left, features=features, operator=operator))
    if wide_policy=='natural_short336_cap672':
        from dinotool.natural_wide_resolution import replace_wide
        if not 1<=len(local)<=4:
            raise RuntimeError('Actual bounded Geometry budget exceeded.')
        return replace_wide(dict(local=local,size=(h,w),output_size=tuple(image.shape[-2:])),image,vip)
    wide_rgb = resize_rgb(image, 448)
    wh, ww = wide_rgb.shape[-2:]
    wide = []
    for top in tile_starts(wh, 336, 224):
        for left in tile_starts(ww, 336, 224):
            ah, aw = min(336, wh - top), min(336, ww - left)
            crop = F.pad(wide_rgb[:, top:top + ah, left:left + aw], (0, 336 - aw, 0, 336 - ah))
            wide.append(dict(top=top, left=left, ah=ah, aw=aw, features=vip.crop_patch_features(crop)))
    if not 1 <= len(local) <= 4 or not 1 <= len(wide) <= 4:
        raise RuntimeError('Actual bounded visual budget exceeded.')
    return dict(local=local, wide=wide, size=(h, w), wide_size=(wh, ww), output_size=tuple(image.shape[-2:]))


@torch.inference_mode()
def wide_scores(source, query, tau, tem):
    h, w = source['wide_size']
    output = torch.zeros(len(query.class_names), h, w, device=query.features.device)
    count = torch.zeros(h, w, device=query.features.device)
    settings = VIPSettings(tau=tau, tem=tem)
    for crop in source['wide']:
        top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
        logits = imagenet_geometry_logits(crop['features'], query, settings)
        output[:, top:top + ah, left:left + aw] += logits[:, :ah, :aw]
        count[top:top + ah, left:left + aw] += 1
    if not bool((count > 0).all()):
        raise RuntimeError('Wide coverage gap.')
    return output / count[None]


@torch.inference_mode()
def probabilities(source, profile, bank, broad, local_cache):
    h, w = source['size']
    text = F.normalize(bank.features.float(), dim=-1)
    blend = torch.from_numpy(hann_blend_window(512)).to(text.device)
    with DeviceProbabilityAccumulator(bank.class_count, h, w, text.device) as accumulator:
        for number, tile in enumerate(source['local']):
            key = profile.bank, profile.strength, profile.temperature, number
            if key not in local_cache:
                alias = tile['features'][profile.strength].float() @ text.T
                # Retain the original LME temperature; tune the final local scale separately.
                local_cache[key] = alias_class_scores(alias, bank.parent_indices, bank.class_count) / profile.temperature
            local = local_cache[key]
            top, left = tile['top'], tile['left']
            wide = sample_broad(broad, top, left, h, w).reshape_as(local)
            logits = coupled(local, wide, tile['operator'], profile.coupling)
            dense = F.interpolate(logits.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                mode='bilinear', align_corners=False)[0]
            ah, aw = min(512, h - top), min(512, w - left)
            accumulator.add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
        if not bool((accumulator.normalizer > 0).all()):
            raise RuntimeError('Local coverage gap.')
        normalized = accumulator.probabilities / accumulator.normalizer[None]
        result = F.interpolate(normalized[None], source['output_size'], mode='bilinear', align_corners=False)[0]
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite probabilities.')
    return result


def confusion(prediction, target, classes):
    valid = (target >= 0) & (target < classes)
    return np.bincount(target[valid].astype(np.int64) * classes + prediction[valid],
        minlength=classes * classes).reshape(classes, classes)


@torch.inference_mode()
def main(args):
    root, output = Path(args.suite_root), Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    manifest = json.loads((root / 'protocol.json').read_text())
    entry = manifest['datasets'][args.dataset]
    validation, samples, load_image, load_mask = load_samples(args, entry)
    selection = json.loads((root / 'search' / args.dataset / 'selection.json').read_text()) if args.mode == 'full' else None
    candidates = profiles(entry['banks']) if args.mode == 'search' else (Profile(),)
    if selection:
        candidates = (Profile(), Profile(**selection['profile']))
    needed = tuple(dict.fromkeys(c.bank for c in candidates)) if args.mode == 'full' else tuple(entry['banks'])
    geometry, vip, banks, queries, checkpoints = load_models(args, entry, needed)
    state = frozen_state(geometry, vip)
    names = banks['original'].class_names
    background = entry['background_index']
    biases = BACKGROUND_BIASES if background is not None else (0.,)
    classes = len(names)
    output.mkdir(parents=True)
    started = time.perf_counter()
    histograms = np.zeros((len(candidates), len(biases), len(THRESHOLDS) + 1, classes, classes), dtype=np.int64) if args.mode == 'search' else None
    matrices = {m: np.zeros((classes, classes), np.int64) for m in METHODS}
    per_image = {m: [] for m in METHODS}
    costs = []
    ignored = 0
    all_keys = [s.key for s in validation]
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={args.dataset: names}, gear=dict(core=manifest['retained_core'], selected=selection,
            observation_cap=[4, 4, 0]), competitive=None, vocabulary=entry['banks'],
        checkpoints=checkpoint_manifest(checkpoints), global_sample_count=len(all_keys),
        global_sample_keys_sha256=digest(all_keys), sample_keys=[s.key for s in samples],
        sample_keys_sha256=digest([s.key for s in samples]), num_shards=args.num_shards,
        shard_index=args.shard_index, config=vars(args))
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        source = observations(image, geometry, vip, tuple(dict.fromkeys(c.strength for c in candidates)))
        local_cache, wide_cache = {}, {}
        if args.mode == 'smoke':
            p = probabilities(source, Profile(), banks['original'],
                wide_scores(source, queries['original'], 1., 1.), local_cache)
            reference, _ = retained_predict(image, geometry, {args.dataset: banks['original']}, vip,
                {args.dataset: queries['original']}, methods=(REFERENCE,))
            if not np.array_equal(p.argmax(0).cpu().numpy(), reference[args.dataset][REFERENCE]):
                raise RuntimeError('Exact retained reference replay failed.')
            save(output / 'results.json', dict(status='complete', processed_images=1, total_images=1,
                exact_retained_prediction=True, target_masks_loaded=False,
                geometry_encodings=len(source['local']), wide_encodings=len(source['wide']), fine_forwards=0,
                **check_frozen(state, geometry, vip)))
            return
        # Search masks are an explicitly authorized supervised development input.
        target = load_mask(sample, source['output_size'])
        target_tensor = torch.as_tensor(target, device=geometry.device)
        ignored += int((target < 0).sum())
        for index, profile in enumerate(candidates):
            wkey = profile.bank, profile.tau, profile.tem
            if wkey not in wide_cache:
                wide_cache[wkey] = wide_scores(source, queries[profile.bank], profile.tau, profile.tem)
            p = probabilities(source, profile, banks[profile.bank], wide_cache[wkey], local_cache)
            if args.mode == 'search':
                for b, bias in enumerate(biases):
                    histograms[index, b] += threshold_histogram(p, target_tensor, background, bias)
            else:
                method = METHODS[index]
                bias = 0. if index == 0 else selection['background_bias']
                threshold = 0. if index == 0 else selection['background_threshold']
                q = biased_probabilities(p, bias, background)
                confidence, prediction = q.max(0)
                if background is not None and threshold > 0:
                    prediction = prediction.masked_fill(confidence < threshold, background)
                cm = confusion(prediction.cpu().numpy(), target, classes)
                matrices[method] += cm
                per_image[method].append(cm)
            del p
        costs.append((len(source['local']), len(source['wide'])))
        if number == 1 or number % 5 == 0 or number == len(samples):
            result = dict(status='running', processed_images=number, total_images=len(samples), mode=args.mode,
                signature=signature, metrics={args.dataset: {m: summary(cm, names, ignored) for m, cm in matrices.items()}},
                diagnostics={args.dataset: dict(tiles=sum(v[0] for v in costs),
                    geometry_encodings=float(np.mean([v[0] for v in costs])),
                    wide_encodings=float(np.mean([v[1] for v in costs])), fine_forwards=0)},
                wall_seconds=time.perf_counter() - started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1048576,
                target_label_tuning=True, tuning_labels_source=entry['development_source'],
                backbone_training=False)
            save(output / 'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, phase=args.mode, processed=number, total=len(samples))), flush=True)
        del source, wide_cache, local_cache, target_tensor
    if args.mode == 'search':
        best, ranked = select(histograms, entry['banks'], background)
        best.update(dataset=args.dataset, development_source=entry['development_source'],
            development_keys=entry['development_keys'], heldout_keys=entry['heldout_keys'],
            target_label_tuning=True, fixed_backbones=True, evaluated_profiles=len(candidates),
            top_candidates=ranked[:20])
        save(output / 'selection.json', best)
        result['selection'] = best
        np.savez_compressed(output / 'development_histograms.npz', histograms=histograms)
    else:
        np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
            **{args.dataset + '__' + m: np.stack(v) for m, v in per_image.items()})
    result.update(status='complete', **check_frozen(state, geometry, vip))
    save(output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('dataset', 'data-root', 'suite-root', 'output-dir', 'original-cache', 'dinov3-repo',
                   'checkpoint-dir', 'upstream-root'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--mode', choices=('smoke', 'search', 'full'), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    main(parser.parse_args())
