"""Unlabeled VIP paper-rule distillation, then paired aggregation from cached logits."""
from __future__ import annotations

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

from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPSettings, upstream_settings
from dinotool.vip_paper_distillation import (
    BackboneAttentionCapture, PaperAliasAccumulator, PaperDistillationConfig,
    aggregate_logits, cached_prediction, canonical_indices, image_crops,
    source_logits, valid_patches,
)
from eval_vip_official_eight import PINNED_COMMIT, Confusion, candidates, checkpoint_config, digest, protocol


IMPLEMENTATION = 'vip-paper-canonical-replacement-finite-v1-20261003'
CONFIG = PaperDistillationConfig()
DATASETS = ('loveda', 'udd5', 'oem', 'landcoverai', 'flair1')
ARMS = ('VIP_All20', 'VIP_Distilled')
CONVENTIONS = {
    'attention': 'Mean over heads then all backbone layers; full-token softmax sliced to patch-patch.',
    'padding': 'Keep patch centers within real crop; exclude padded random-walk keys and scoring positions.',
    'image_pooling': 'Sum high-region intersection/union/entropy/count across pinned sliding crops; then equal-image averaging, skipping absent aliases.',
    'cosine_prefilter': 'Cosine of normalized means of the pinned 80 ImageNet-template patch-text features; fixed threshold0.7.',
    'alias_maps': 'VIP template-mean similarity times pinned logit_scale40; replace one parent canonical, softmax across canonical classes.',
    'protocol': 'Pinned repository long-edge448/crop336/stride112 retained. Paper appendix shorter-edge336/crop224 is not substituted.',
    'code_provenance': 'Paper-rule distillation reimplementation; public pinned VIP has no distillation implementation.',
    'numeric_repair': 'Fixed self-Value fallback ONLY on all-masked proxy rows; otherwise pinned VIP equations unchanged. Historical raw VIP could yield NaN.',
}


def save(path, row):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(row, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'diagnostic', 'selection', 'evaluation'), required=True)
    parser.add_argument('--dataset', choices=(*DATASETS, 'vdd', 'potsdam'), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'vocabulary-config', 'output-dir', 'reference-json'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-dir', default='')
    parser.add_argument('--selection-json', default='')
    parser.add_argument('--candidate-reference-json', default='',
                        help='Verified full20-bank result when the VIP reference uses official short queries.')
    parser.add_argument('--vocabulary-source', default='json')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-fraction', type=float, default=.6)
    parser.add_argument('--progress-every', type=int, default=10)
    return parser.parse_args()


def sample_protocol(args):
    all_samples, scored, load_image, load_mask = protocol(args)
    ref = json.loads(Path(args.reference_json).read_text())
    keys = [sample.key for sample in all_samples]
    if (not ref['coverage_verified'] or keys != ref['signature']['sample_keys']
            or len(set(keys)) != len(keys) or len(keys) != ref['processed_images']):
        raise RuntimeError('Full image sequence differs from the historical VIP comparator.')
    return all_samples, scored, load_image, load_mask, ref


def verify_references(ref, manifest, vocab_sha, settings, commit, groups, names, candidate_ref=None):
    old = ref['signature']
    if (manifest != old['checkpoint_manifest'] or asdict(settings) != old['settings']
            or commit != old['upstream_commit']):
        raise RuntimeError('Checkpoint/settings differ from reference VIP.')
    if candidate_ref is None:
        if vocab_sha != old['vocabulary_sha256']:
            raise RuntimeError('Vocabulary differs from reference VIP.')
        return
    candidate = candidate_ref['signature']
    flattened = {p: [alias for group in bank for alias in group] for p, bank in groups.items()}
    counts = {p: [len(group) for group in bank] for p, bank in groups.items()}
    if (not candidate_ref['coverage_verified'] or candidate_ref['status'] != 'complete'
            or candidate_ref['processed_images'] != ref['processed_images']
            or candidate_ref['total_images'] != ref['total_images']
            or candidate['dataset'] != old['dataset']
            or candidate['sample_keys'] != old['sample_keys']
            or candidate['checkpoints'] != manifest
            or candidate['vocabulary'] != dict(sha256=vocab_sha, aliases=flattened, counts=counts)
            or candidate['classes'] != {p: list(n) for p, n in names.items()}):
        raise RuntimeError('Full20 candidate reference identities differ.')


def make_model(args, scored, ref):
    commit = subprocess.check_output(['git', '-C', args.upstream_root, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError('Pinned upstream revision changed.')
    config = checkpoint_config(args)
    groups, names, path, source = candidates(args, scored)
    if any(len(group) != 20 for bank in groups.values() for group in bank):
        raise RuntimeError('This paired comparison requires unchanged20-alias candidate groups.')
    manifest = checkpoint_manifest(config)
    settings = upstream_settings(args.dataset)
    vocab_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    reference_path = getattr(args, 'candidate_reference_json', '')
    candidate_ref = json.loads(Path(reference_path).read_text()) if reference_path else None
    verify_references(ref, manifest, vocab_sha, settings, commit, groups, names, candidate_ref)
    model = FiniteVIPObserver(DINOTextSegmenter(config, device=args.device, amp=True), Path(args.upstream_root))
    banks = {p: model.encode_queries(names[p], groups[p]) for p in scored}
    metadata = {}
    for p, bank in banks.items():
        canonical = canonical_indices(bank.class_names, bank.aliases, bank.parents)
        text = F.normalize(bank.features.float().mean(1), dim=-1)
        cosine = (text * text[canonical[bank.parents]]).sum(-1)
        metadata[p] = dict(names=list(bank.class_names), aliases=list(bank.aliases), parents=bank.parents.cpu().tolist(),
                           canonical=canonical.cpu().tolist(), cosine=cosine.cpu().tolist())
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, config=asdict(CONFIG),
        conventions=CONVENTIONS, upstream_commit=commit, settings=asdict(settings), vocabulary_sha256=vocab_sha,
        checkpoint_manifest=manifest, query_metadata=metadata, scored_classes=scored, template_count=len(model.templates),
        data_root=str(Path(args.data_root).resolve()), num_shards=args.num_shards,
        shard_index=args.shard_index, target_label_tuning=False, transductive_all_evaluation_images=True)
    if candidate_ref is not None:
        signature['candidate_reference'] = dict(implementation=candidate_ref['signature']['implementation'],
            vocabulary_sha256=vocab_sha, sample_keys_sha256=candidate_ref['signature']['sample_keys_sha256'])
        signature['comparator_vocabulary_sha256'] = ref['signature']['vocabulary_sha256']
    return model, banks, settings, signature


@torch.inference_mode()
def observe_image(image, model, banks, settings, capture, collectors=None):
    empty_before, rows_before = model.empty_rows, model.observed_rows
    original, resized, crops = image_crops(image, settings)
    arrays = dict(original_shape=np.asarray(original), resized_shape=np.asarray(resized),
                  bounds=np.asarray([bounds for _, bounds in crops]))
    for p in banks:
        arrays[p + '_logits'], arrays[p + '_salience'] = [], []
    canonical = {p: canonical_indices(bank.class_names, bank.aliases, bank.parents.cpu()).to(model.device)
                 for p, bank in banks.items()} if collectors is not None else {}
    for crop, bounds in crops:
        capture.reset()
        patch = model.crop_patch_features(crop)
        attention = capture.mean()
        valid = valid_patches(bounds, model.device)
        for p, bank in banks.items():
            logits, salience = source_logits(patch, bank, settings)
            arrays[p + '_logits'].append(logits.cpu().numpy())
            arrays[p + '_salience'].append(salience.cpu().numpy())
            if collectors is not None:
                collectors[p].update_crop(logits.flatten(1).T, bank.parents, canonical[p], attention, valid, CONFIG)
    for p in banks:
        arrays[p + '_logits'] = np.stack(arrays[p + '_logits'])
        arrays[p + '_salience'] = np.stack(arrays[p + '_salience'])
        if collectors is not None:
            collectors[p].finalize_image()
    arrays['empty_proxy_rows'] = np.asarray(model.empty_rows - empty_before)
    arrays['observed_proxy_rows'] = np.asarray(model.observed_rows - rows_before)
    return arrays


def common_signature(signature, all_samples, samples):
    return dict(signature, sample_keys=[s.key for s in samples], sample_count=len(samples),
                sample_keys_sha256=digest([s.key for s in samples]), global_sample_count=len(all_samples),
                global_sample_keys_sha256=digest([s.key for s in all_samples]))


def smoke(args, output, all_samples, scored, load_image, ref):
    model, banks, settings, signature = make_model(args, scored, ref)
    image = load_image(all_samples[0])
    crop = image_crops(image, settings)[2][0][0]
    plain_features = model.crop_patch_features(crop)
    upstream = VIPOfficialAdapter(model.backbone, Path(args.upstream_root))
    upstream_features = upstream.crop_patch_features(crop)
    originally_finite = bool(torch.isfinite(upstream_features).all())
    upstream_error = float((upstream_features - plain_features).abs().max()) if originally_finite else None
    versions = {name: tensor._version for name, tensor in model.backbone.model.state_dict().items()}
    errors = {}
    with BackboneAttentionCapture(model.backbone.model.visual_model.backbone) as capture:
        arrays = observe_image(image, model, banks, settings, capture)
        hooked = model.crop_patch_features(crop)
        feature_error = float((hooked - plain_features).abs().max())
    for p, bank in banks.items():
        logit, salience = source_logits(plain_features, bank, settings)
        direct = model.crop_logits(crop, bank, settings)
        recovered = aggregate_logits(logit, salience, bank.parents, len(bank.class_names), settings)
        prediction, _ = model.predict(image, bank, settings)
        cached = cached_prediction(arrays, p, bank.parents, bank.class_names, settings, None, model.device)
        errors[p] = dict(crop_logit_max_error=float((direct - recovered).abs().max()),
                         cached_prediction_different_pixels=int((prediction != cached).sum()))
    unchanged = versions == {name: tensor._version for name, tensor in model.backbone.model.state_dict().items()}
    frozen = not any(parameter.requires_grad for parameter in model.backbone.model.parameters())
    if feature_error != 0 or upstream_error not in (None, 0.) or not unchanged or not frozen or any(
            e['crop_logit_max_error'] != 0 or e['cached_prediction_different_pixels'] for e in errors.values()):
        raise RuntimeError(f'Unchanged VIP/source cache smoke failed: {feature_error}, {errors}')
    row = dict(status='complete', processed_images=1, total_images=1, target_masks_loaded=False,
               weights_frozen=frozen, weights_unchanged=unchanged, feature_replay_max_error=feature_error,
               original_upstream_features_finite=originally_finite, originally_finite_feature_max_error=upstream_error,
               errors=errors, signature=signature)
    save(output/'results.json', row)
    print(json.dumps(row), flush=True)


def diagnostic(args, output, all_samples, scored, load_image, ref):
    finite, _, settings, signature = make_model(args, scored, ref)
    original = VIPOfficialAdapter(finite.backbone, Path(args.upstream_root))
    images = []
    for sample in all_samples[:8]:
        crops = []
        for crop, bounds in image_crops(load_image(sample), settings)[2]:
            raw = original.crop_patch_features(crop)
            before = finite.empty_rows
            repaired = finite.crop_patch_features(crop)
            empty = finite.empty_rows - before
            bad = int((~torch.isfinite(raw).all(-1)).sum())
            good = torch.isfinite(raw).all(-1)
            finite_error = float((raw[good] - repaired[good]).abs().max()) if good.any() else None
            crops.append(dict(bounds=bounds, nonfinite_projected_patches=bad,
                empty_proxy_rows_two_blocks=empty, repaired_features_finite=bool(torch.isfinite(repaired).all()),
                originally_finite_feature_max_error=finite_error))
        images.append(dict(sample_key=sample.key, crops=crops))
    row = dict(status='complete', processed_images=len(images), total_images=len(images), target_masks_loaded=False,
        diagnostics_only=True, source_model_unchanged=True, hypothetical_repair='Identity self-Value on empty proxy rows only; not promoted to evaluation.',
        signature=signature, images=images, nonfinite_crops=sum(c['nonfinite_projected_patches'] > 0 for i in images for c in i['crops']),
        total_crops=sum(len(i['crops']) for i in images),
        empty_proxy_rows_two_blocks=sum(c['empty_proxy_rows_two_blocks'] for i in images for c in i['crops']))
    save(output/'results.json', row)
    print(json.dumps(row), flush=True)


def selection(args, output, all_samples, scored, load_image, ref):
    indexed = list(enumerate(all_samples))[args.shard_index::args.num_shards]
    model, banks, settings, signature = make_model(args, scored, ref)
    signature = common_signature(signature, all_samples, [sample for _, sample in indexed])
    save(output/'signature.json', signature)
    cache = output/'cache'
    cache.mkdir()
    collectors = {p: PaperAliasAccumulator(len(bank.aliases), model.device) for p, bank in banks.items()}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with BackboneAttentionCapture(model.backbone.model.visual_model.backbone) as capture:
        for number, (index, sample) in enumerate(indexed, 1):
            arrays = observe_image(load_image(sample), model, banks, settings, capture, collectors)
            arrays['sample_key'] = np.asarray(sample.key)
            np.savez_compressed(cache/f'{index:06d}.npz', **arrays)
            if number % args.progress_every == 0 or number == len(indexed):
                row = dict(status='complete' if number == len(indexed) else 'running', processed_images=number,
                    total_images=len(indexed), signature=signature, target_masks_loaded=False, weights_frozen=True,
                    statistics={p: collector.state() for p, collector in collectors.items()},
                    empty_proxy_rows=model.empty_rows, observed_proxy_rows=model.observed_rows,
                    wall_seconds=time.perf_counter() - started,
                    peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
                save(output/'results.json', row)
                print(json.dumps(dict(dataset=args.dataset, phase='selection', processed=number, total=len(indexed),
                                      wall_seconds=row['wall_seconds'])), flush=True)


def evaluation(args, output, all_samples, scored, load_mask):
    source = Path(args.source_dir)
    selected = json.loads(Path(args.selection_json).read_text())
    row = json.loads((source/'results.json').read_text())
    if (row['status'] != 'complete' or row['processed_images'] != row['total_images'] or row['target_masks_loaded']
            or selected['status'] != 'complete' or not selected['coverage_verified']):
        raise RuntimeError('Complete mask-free full selection required before evaluation.')
    signature = row['signature']
    if selected['source_signature'] != {k: v for k, v in signature.items() if k not in
            ('sample_keys', 'sample_keys_sha256', 'sample_count', 'shard_index')}:
        raise RuntimeError('Selected vocabulary does not match its cache source.')
    indexed = list(enumerate(all_samples))[args.shard_index::args.num_shards]
    if [s.key for _, s in indexed] != signature['sample_keys']:
        raise RuntimeError('Cache sample keys differ.')
    matrices = {p: {arm: Confusion(tuple(names), len(signature['query_metadata'][p]['names'])) for arm in ARMS}
                for p, names in scored.items()}
    parents = {p: torch.tensor(meta['parents'], device=args.device) for p, meta in signature['query_metadata'].items()}
    keeps = {p: torch.tensor(selected['banks'][p]['keep'], device=args.device) for p in scored}
    save(output/'signature.json', signature)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    per_image = {p+'__'+arm: [] for p in scored for arm in ARMS}
    for number, (index, sample) in enumerate(indexed, 1):
        with np.load(source/'cache'/f'{index:06d}.npz', allow_pickle=False) as arrays:
            if arrays['sample_key'].item() != sample.key:
                raise RuntimeError('Cache sample ID differs.')
            predictions = {p: {arm: cached_prediction(arrays, p, parents[p], signature['query_metadata'][p]['names'],
                VIPSettings(**signature['settings']), None if arm == 'VIP_All20' else keeps[p], args.device)
                for arm in ARMS} for p in scored}
            shape = tuple(map(int, arrays['original_shape']))
        for p, arms in predictions.items():
            # Masks are first loaded after both predictions are finalized.
            target = load_mask(sample, p, shape)
            for arm, prediction in arms.items():
                single = Confusion(tuple(scored[p]), len(signature['query_metadata'][p]['names']))
                single.update(prediction, target)
                matrices[p][arm].matrix += single.matrix
                matrices[p][arm].ignored += single.ignored
                per_image[p+'__'+arm].append(single.matrix)
        if number % args.progress_every == 0 or number == len(indexed):
            result = dict(status='complete' if number == len(indexed) else 'running', processed_images=number,
                total_images=len(indexed), signature=signature, selection=selected['banks'],
                metrics={p: {arm: m.summary() for arm, m in arms.items()} for p, arms in matrices.items()},
                selection_sha256=hashlib.sha256(Path(args.selection_json).read_bytes()).hexdigest(),
                target_masks_loaded_only_after_prediction=True, wall_seconds=time.perf_counter() - started,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                selection_wall_seconds=row['wall_seconds'], selection_peak_cuda_memory_mb=row['peak_cuda_memory_mb'])
            if number == len(indexed):
                np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                                    **{k: np.asarray(v) for k, v in per_image.items()})
            save(output/'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, phase='evaluation', processed=number, total=len(indexed),
                                  miou={p: {a: m['mean_iou_percent'] for a, m in arms.items()}
                                        for p, arms in result['metrics'].items()})), flush=True)


def main(args):
    if not 0 <= args.shard_index < args.num_shards or args.progress_every < 1:
        raise ValueError('Invalid shard/progress configuration.')
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing an existing output: '+str(output))
    all_samples, scored, load_image, load_mask, ref = sample_protocol(args)
    if not all_samples[args.shard_index::args.num_shards]:
        raise ValueError('Empty shard.')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    output.mkdir(parents=True)
    if args.phase == 'smoke':
        smoke(args, output, all_samples, scored, load_image, ref)
    elif args.phase == 'diagnostic':
        diagnostic(args, output, all_samples, scored, load_image, ref)
    elif args.phase == 'selection':
        selection(args, output, all_samples, scored, load_image, ref)
    else:
        evaluation(args, output, all_samples, scored, load_mask)


if __name__ == '__main__':
    main(arguments())
