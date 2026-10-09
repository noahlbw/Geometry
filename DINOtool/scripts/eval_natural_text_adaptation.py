"""Calibrate text inputs without masks, then evaluate the unchanged coupled model."""
import argparse
import ast
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.natural_evaluation import TOTALS, discover_samples, load_rgb, load_target
from dinotool.natural_text_adaptation import (IMPLEMENTATION, CLASS_FILES, CONFIG_FILES,
    TEMPLATE_FAMILIES, TAU_GRID, TEM_GRID, SEED, CALIBRATION_IMAGES, semantic_pool,
    class_logits, trusted_witnesses, choose_aliases, balanced_nll, background_threshold)
from dinotool.natural_variable_alias_reader import retained_variable_scores
from dinotool.rival_fine_full import IMPLEMENTATION as CORE
from dinotool.stratified_soft_alias import crop_stencil
from dinotool.tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries, VIPSettings, _load_upstream, upstream_aliases
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_vip_reliability import summary, sample_broad
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, tile_reference_coordinates
from eval_stratified_soft_alias import tile_coordinates
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT, digest


METHODS = ('Geometry_Pool', 'Coupled_Pool', 'RivalFine_Pool', 'Adaptive_NoThreshold',
           'Adaptive_RivalFine', 'VIP_Official_Finite', 'VIP_Official_NoThreshold')


def official_options(root, dataset):
    tree = ast.parse((root/'configs'/('cfg_'+CONFIG_FILES[dataset]+'.py')).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'model' for n in node.targets):
            if not isinstance(node.value, ast.Call) or getattr(node.value.func, 'id', None) != 'dict':
                raise ValueError('Expected a literal upstream model configuration.')
            options = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            return VIPSettings(**{k: v for k, v in options.items() if k in VIPSettings.__dataclass_fields__}), options.get('text_template', 'openai_imagenet_template')
    raise ValueError('Missing upstream model settings.')


def bank_from_query(query):
    canonical = torch.tensor([(query.parents == c).nonzero()[0, 0]
                              for c in range(len(query.class_names))], device=query.parents.device)
    mask = torch.zeros_like(query.parents, dtype=torch.bool)
    mask[canonical] = True
    bank = TCPRTextBank(F.normalize(query.features.float().mean(1), dim=-1), query.parents,
                       mask, query.class_names, query.aliases)
    bank.validate()
    return bank, canonical


def subset(query, indices):
    return VIPQueries(query.features[indices], query.parents[indices], query.class_names,
                      tuple(query.aliases[i] for i in indices.tolist()))


def models(args):
    upstream = Path(args.upstream_root)
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Pinned VIP source changed.')
    source = json.loads(Path(args.source_vocabulary).read_text())
    names = tuple(c['name'] for c in source['classes'])
    official = upstream_aliases(upstream/'configs'/('cls_'+CLASS_FILES[args.dataset]+'.txt'))
    groups, removed = semantic_pool(names, official)
    checkpoints = make_checkpoints(args)
    geometry = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device), TCPRConfig(geometry_depth=2, maximum_aliases_per_class=20))
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), upstream)
    prompt = _load_upstream(upstream/'prompts/imagenet_template.py', 'natural_calibration_prompts')
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoints=checkpoint_manifest(checkpoints), upstream_commit=PINNED_COMMIT,
                    dataset=args.dataset, classes=names, groups=groups, implementation=IMPLEMENTATION)
    encoded = {}
    for family in TEMPLATE_FAMILIES:
        path = cache/(family+'.pt')
        if path.exists():
            row = torch.load(path, map_location=args.device, weights_only=True)
            if row['identity'] != identity:
                raise ValueError('Changed frozen text cache.')
            query = VIPQueries(row['features'], row['parents'], names, tuple(a for g in groups for a in g))
        else:
            vip.templates = prompt.get_text_template(family)
            query = vip.encode_queries(names, groups)
            torch.save(dict(identity=identity, features=query.features.cpu(), parents=query.parents.cpu()), path)
        encoded[family] = query
    settings, family = official_options(upstream, args.dataset)
    path = cache/'official.pt'
    if path.exists():
        row = torch.load(path, map_location=args.device, weights_only=True)
        if row['groups'] != official or row['family'] != family or row['checkpoints'] != checkpoint_manifest(checkpoints):
            raise ValueError('Changed official text cache.')
        reference = VIPQueries(row['features'], row['parents'], names, tuple(a for g in official for a in g))
    else:
        vip.templates = prompt.get_text_template(family)
        reference = vip.encode_queries(names, official)
        torch.save(dict(features=reference.features.cpu(), parents=reference.parents.cpu(),
                        groups=official, family=family, checkpoints=checkpoint_manifest(checkpoints)), path)
    return geometry, vip, encoded, reference, settings, identity, removed


def sampled_aliases(crops, count, coordinates, image_size):
    output = coordinates.new_zeros((len(coordinates), crops[0].alias_logits.shape[-1]))
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, image_size)
        output += (crop.alias_logits[ids] * coeff[..., None]).sum(1)
    return output


@torch.inference_mode()
def select(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Refusing an existing selection output.')
    samples = discover_samples(args.dataset, args.data_root)
    indices = sorted(random.Random(SEED).sample(range(len(samples)), min(CALIBRATION_IMAGES, len(samples))))
    chosen = [samples[i] for i in indices]
    started = time.perf_counter()
    geometry, vip, queries, reference, settings, identity, removed = models(args)
    states = frozen_state(geometry, vip)
    banks = {f: bank_from_query(q)[0] for f, q in queries.items()}
    canonical = bank_from_query(queries['seg_template'])[1]
    parents = queries['seg_template'].parents
    records = {family: [] for family in TEMPLATE_FAMILIES}
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        h, w = image.shape[-2:]
        top, left = max(0, (h-512)//2), max(0, (w-512)//2)
        prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
        coordinates = tile_coordinates(top, left, geometry.device)
        valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
        _, crops, _, count = prepare_wide(image, vip, {'calibration': (banks, queries)})
        fields = {}
        operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        for family, bank in banks.items():
            geo = ((prepared.geometry_projected.float() @ bank.features.T)[0] * 40)[valid]
            wide = sampled_aliases(crops['calibration', family], count, coordinates[valid], (h, w))
            salience = torch.stack([c.salience for c in crops['calibration', family]]).mean(0)
            fields[family] = dict(geo=geo.cpu(), wide=wide.cpu(), salience=salience.cpu(),
                                  operator=operator[valid][:, valid].float().cpu())
        labels, quality, _ = trusted_witnesses(fields['seg_template']['geo'], fields['seg_template']['wide'],
            canonical.cpu(), parents.cpu(), len(identity['classes']), identity['classes'][0] == 'background')
        for family in TEMPLATE_FAMILIES:
            records[family].append(dict(**fields[family], labels=labels, quality=quality))
        print(json.dumps(dict(stage='image_only_calibration', dataset=args.dataset, processed=number,
                              total=len(chosen), trusted_tokens=int((quality > 0).sum()))), flush=True)
    selected, details = choose_aliases(records['seg_template'], parents.cpu(), canonical.cpu(), identity['classes'])
    candidates = []
    classes = len(identity['classes'])
    subparents = parents.cpu()[selected]
    for family in TEMPLATE_FAMILIES:
        for tau in TAU_GRID:
            for tem in TEM_GRID:
                scores, labels, quality = [], [], []
                for row in records[family]:
                    scores.append(calibration_scores(row, selected, subparents, classes, tau, tem, geometry.device).cpu())
                    labels.append(row['labels'])
                    quality.append(row['quality'])
                candidate = dict(template=family, tau=tau, tem=tem,
                    class_balanced_witness_nll=balanced_nll(torch.cat(scores), torch.cat(labels), torch.cat(quality)))
                candidates.append(candidate)
    finite = [c for c in candidates if np.isfinite(c['class_balanced_witness_nll'])]
    best = min(finite, key=lambda c: c['class_balanced_witness_nll']) if finite else dict(template='seg_template', tau=1., tem=1., class_balanced_witness_nll=None)
    confidence, bg, quality = [], [], []
    if identity['classes'][0] == 'background':
        for row in records[best['template']]:
            prob = calibration_scores(row, selected, subparents, classes, best['tau'], best['tem'], geometry.device).softmax(-1).cpu()
            confidence.extend(prob.amax(-1).tolist())
            bg.extend((row['labels'] == 0).tolist())
            quality.extend(row['quality'].tolist())
        threshold, threshold_reason = background_threshold(confidence, bg, quality)
    else:
        threshold, threshold_reason = 0., 'No scored background class; threshold stays off.'
    output.mkdir(parents=True)
    result = dict(status='complete', implementation=IMPLEMENTATION, core_implementation=CORE,
        dataset=args.dataset, identity=identity, image_keys=[s.key for s in chosen], selected_indices=selected.tolist(),
        pool_counts=[int((parents == c).sum()) for c in range(classes)],
        selected_counts=[int((subparents == c).sum()) for c in range(classes)], lexical_removals=removed,
        alias_audit=[dict(**row, alias=queries['seg_template'].aliases[row['alias_index']],
                          parent=identity['classes'][int(parents[row['alias_index']])]) for row in details],
        chosen=dict(**best, prob_thd=threshold), threshold_reason=threshold_reason, candidates=candidates,
        trusted_tokens=sum(int((r['quality'] > 0).sum()) for r in records['seg_template']),
        source_vocabulary_sha256=hashlib.sha256(Path(args.source_vocabulary).read_bytes()).hexdigest(),
        target_masks_loaded=False, target_label_tuning=False, transductive=True,
        selection_rule='Canonical cross-view witnesses, conservative upper95 negative alias margin, class-balanced NLL; no quota.',
        wall_seconds=time.perf_counter()-started, **check_frozen(states, geometry, vip))
    save(output/'selection.json', result)
    print(json.dumps(dict(dataset=args.dataset, chosen=result['chosen'], counts=result['selected_counts'])), flush=True)


def calibration_scores(row, selected, parents, classes, tau, tem, device):
    selected, parents = selected.to(device), parents.to(device)
    geo = row['geo'].to(device)[:, selected]
    raw = alias_class_scores(geo/40, parents, classes)/.07
    broad = class_logits(row['wide'].to(device)[:, selected], row['salience'].to(device)[selected],
                         parents, classes, tau, tem)
    return raw + row['operator'].to(device) @ (broad-raw)


@torch.inference_mode()
def official_prediction(image, vip, query, settings):
    h, w = image.shape[-2:]
    ratio = min(336/min(h, w), 2048/max(h, w))
    nh, nw = int(h*ratio+.5), int(w*ratio+.5)
    array = (image.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
    resized = torch.from_numpy(np.ascontiguousarray(cv2.resize(array, (nw, nh)))).permute(2, 0, 1).float()/255
    total = torch.zeros(len(query.class_names), nh, nw, device=vip.device)
    count = torch.zeros(nh, nw, device=vip.device)
    for top in tile_starts(nh, 336, 224):
        for left in tile_starts(nw, 336, 224):
            ah, aw = min(336, nh-top), min(336, nw-left)
            crop = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            total[:, top:top+ah, left:left+aw] += vip.crop_logits(crop, query, settings)[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
    probabilities = F.interpolate((total/count)[None], (h, w), mode='bilinear', align_corners=False)[0].softmax(0)
    plain = probabilities.argmax(0)
    calibrated = plain.masked_fill(probabilities.amax(0) < settings.prob_thd,
                                  settings.bg_idx if settings.background else 255)
    return calibrated.cpu().numpy().astype(np.uint8), plain.cpu().numpy().astype(np.uint8)


@torch.inference_mode()
def predict(image, geometry, vip, pool, adapted, reference, settings, choice, work):
    h, w = image.shape[-2:]
    queries = {'pool': pool, 'adapted': adapted}
    banks = {k: bank_from_query(q)[0] for k, q in queries.items()}
    wide, crops, _, count = prepare_wide(image, vip, {'text': (banks, queries)})
    # Recompute only the class aggregation with the frozen calibrated tau/tem.
    adapted_map = torch.zeros_like(wide['text', 'adapted'])
    for crop in crops['text', 'adapted']:
        logits = class_logits(crop.alias_logits, crop.salience, adapted.parents, len(adapted.class_names), choice['tau'], choice['tem'])
        dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336), mode='bilinear', align_corners=False)[0]
        adapted_map[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
    wide['text', 'adapted'] = adapted_map / count
    blend = hann_blend_window(512)
    diag = dict(tiles=0, mean_absolute_admission_potential=0., retained_count_mean=0.)
    with ExitStack() as stack:
        acc = {m: stack.enter_context(ProbabilityAccumulator(len(pool.class_names), h, w, 256, work))
               for m in METHODS[:5] if m != 'Adaptive_RivalFine'}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coords = tile_coordinates(top, left, geometry.device)
                valid = (coords[:, 0] < h) & (coords[:, 1] < w)
                fine_coords = tile_reference_coordinates(coords, top, left)
                fine, fine_count, _ = observe_fine(image[:, top:top+512, left:left+512], vip, queries, fine_coords, valid)
                operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                scores = {}
                for variant, bank in banks.items():
                    raw = alias_class_scores((prepared.geometry_projected.float() @ bank.features.T)[0], bank.parent_indices, bank.class_count)
                    broad = sample_broad(wide['text', variant], top, left, h, w).reshape_as(raw)
                    selected_crops = crops['text', variant]
                    selected_fine = fine[variant]
                    if variant == 'adapted':
                        selected_crops = [replace(c, salience=c.salience/choice['tem']) for c in selected_crops]
                        selected_fine = [replace(c, salience=c.salience/choice['tem']) for c in selected_fine]
                    value, stats = retained_variable_scores(raw/.07, operator, broad, selected_crops, count,
                        selected_fine, fine_count, coords, fine_coords, valid, bank.parent_indices,
                        bank.canonical_mask.nonzero().flatten(), (h, w), beta=choice['tau'] if variant == 'adapted' else 1.)
                    if variant == 'pool':
                        scores.update(Geometry_Pool=value['Geometry'], Coupled_Pool=value['NoAdmission_Exact'], RivalFine_Pool=value['RivalFineHard_Exact'])
                    else:
                        scores['Adaptive_NoThreshold'] = value['RivalFineHard_Exact']
                        for field in ('mean_absolute_admission_potential', 'retained_count_mean'):
                            diag[field] += stats[field]
                ah, aw = min(512, h-top), min(512, w-left)
                for method, logits in scores.items():
                    dense = F.interpolate(logits.T.reshape(1, -1, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                    acc[method].add(dense[:, :ah, :aw].softmax(0).float().cpu().numpy(), blend[:ah, :aw], left, top)
                diag['tiles'] += 1
        predictions = {m: accumulator.finalize(None)[0] for m, accumulator in acc.items() if m != 'Adaptive_NoThreshold'}
        adaptive, _ = acc['Adaptive_NoThreshold'].finalize(None)
        predictions['Adaptive_NoThreshold'] = adaptive
        predictions['Adaptive_RivalFine'] = acc['Adaptive_NoThreshold'].finalize(choice['prob_thd'], background_index=0)[0]
    predictions['VIP_Official_Finite'], predictions['VIP_Official_NoThreshold'] = official_prediction(image, vip, reference, settings)
    return predictions, {k: v if k == 'tiles' else v/diag['tiles'] for k, v in diag.items()}


@torch.inference_mode()
def evaluate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Refusing an existing evaluation output.')
    selection = json.loads(Path(args.selection).read_text())
    if selection['target_masks_loaded'] or selection['target_label_tuning'] or selection['status'] != 'complete':
        raise ValueError('Frozen mask-free selection required.')
    geometry, vip, encoded, reference, settings, identity, _ = models(args)
    if json.loads(json.dumps(identity)) != selection['identity']:
        raise ValueError('Selection/source identity differs.')
    states = frozen_state(geometry, vip)
    choice = selection['chosen']
    adapted = subset(encoded[choice['template']], torch.tensor(selection['selected_indices'], device=geometry.device))
    samples = discover_samples(args.dataset, args.data_root)
    if args.max_images:
        keys = set(random.Random(SEED+1).sample([s.key for s in samples], min(args.max_images, len(samples))))
        samples = [s for s in samples if s.key in keys]
    keys = [s.key for s in samples]
    chosen = samples[args.shard_index::args.num_shards]
    if not chosen:
        raise ValueError('Empty shard.')
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={args.dataset: identity['classes']}, checkpoints=identity['checkpoints'],
        gear=dict(core_implementation=CORE, geometry=asdict(geometry.config), selected_calibration=choice,
                  selection_sha256=hashlib.sha256(Path(args.selection).read_bytes()).hexdigest(),
                  local_tiles=[512, 128], wide_view=[448, 336, 112], fine_physical_pixels_per_token=8,
                  ragged_reader='Same retained equations, actual counts; equal-count equivalence unit tested.',
                  official_reference='Upstream short-edge336/max-long2048, original queries/settings, empty-row Self-Value repair.'),
        vocabulary=dict(pool_counts=selection['pool_counts'], selected_counts=selection['selected_counts'],
                        selected_aliases=adapted.aliases, selected_parents=adapted.parents.tolist()),
        global_sample_count=len(keys), global_sample_keys_sha256=digest(keys), sample_keys=[s.key for s in chosen],
        sample_keys_sha256=digest([s.key for s in chosen]), num_shards=args.num_shards, shard_index=args.shard_index,
        partial_run=bool(args.max_images), config=vars(args), selection_masks_used=False)
    output.mkdir(parents=True)
    classes = len(identity['classes'])
    matrices = {m: np.zeros((classes, classes+int(m == 'VIP_Official_Finite' and not settings.background
                                                and settings.prob_thd > 0)), dtype=np.int64) for m in METHODS}
    arrays = {m: [] for m in METHODS}
    ignored, started = 0, time.perf_counter()
    for number, sample in enumerate(chosen, 1):
        image = load_rgb(sample)
        predictions, diag = predict(image, geometry, vip, encoded['seg_template'], adapted, reference, settings, choice, output)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        valid = (target >= 0) & (target < len(identity['classes']))
        ignored += int((~valid).sum())
        for method, pred in predictions.items():
            if method == 'VIP_Official_Finite' and not settings.background and settings.prob_thd > 0:
                pred = np.where(pred == 255, classes, pred).astype(np.int64)
            if pred.shape != target.shape or np.any(pred[valid] >= matrices[method].shape[1]):
                raise ValueError('Out-of-range predicted labels or changed image coverage.')
            cm = np.bincount(target[valid]*matrices[method].shape[1]+pred[valid], minlength=matrices[method].size).reshape(matrices[method].shape)
            matrices[method] += cm
            arrays[method].append(cm)
        result = dict(status='running', processed_images=number, total_images=len(chosen), signature=signature,
            metrics={args.dataset: {m: summary(cm, identity['classes'], ignored) for m, cm in matrices.items()}}, diagnostics={args.dataset: diag},
            wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
            target_masks_used_only_after_prediction=True, target_label_tuning=False)
        if number == 1 or number % 10 == 0 or number == len(chosen):
            save(output/'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(chosen), miou={m: x['mean_iou_percent'] for m, x in result['metrics'][args.dataset].items()})), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{args.dataset+'__'+m: np.stack(a) for m, a in arrays.items()})
    result.update(status='complete', **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'evaluate'))
    parser.add_argument('--dataset', choices=tuple(TOTALS), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary', 'cache-dir', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    select(args) if args.action == 'select' else evaluate(args)
