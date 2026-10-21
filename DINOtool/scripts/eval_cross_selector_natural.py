"""Paired selector comparison on the natural-image validation protocols.

The candidate pool is fixed before this script is called.  A bank is only a
different subset of that pool; the image/mask reader and the score metric are
identical for all banks.  Development labels are optional and are used only
to choose Geov2's declared scalar g0 menu.  Full validation is then run with
the frozen choice.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dinotool.config import CheckpointConfig
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.model import DINOTextSegmenter
from dinotool.natural_evaluation import discover_samples, load_rgb, load_target
from dinotool.vip_official_adapter import VIPQueries, _load_upstream
from eval_natural_text_adaptation import official_options, official_prediction


DATASET_INFO = {
    'voc21': dict(classes=21, total_images=1449, background=0, temperature=.07, tau=3., tem=.3,
                  background_rule='retained', background_bias=-.25, background_threshold=.15,
                  config_name='voc21'),
    'coco_object81': dict(classes=81, total_images=5000, background=0, temperature=.07, tau=6., tem=.3,
                          background_rule='retained', background_bias=-4., background_threshold=.05,
                          config_name='coco_object81'),
    'ade150': dict(classes=150, total_images=2000, background=None, temperature=.14, tau=5., tem=5.,
                   background_rule='retained', background_bias=0., background_threshold=0.,
                   config_name='ade150'),
}


def checkpoints(args):
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(Path(args.dinov3_repo).resolve(), root,
        root/'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth',
        root/'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        root/'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        root/'bpe_simple_vocab_16e6.txt.gz')


def metric(cm):
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    return float(np.nanmean(iou) * 100.)


def update(cm, prediction, target):
    valid = (target >= 0) & (target < cm.shape[0])
    prediction = np.asarray(prediction, dtype=np.int64)
    prediction = np.where(prediction == 255, cm.shape[0] - 1, prediction)
    encoded = target[valid].astype(np.int64) * cm.shape[0] + prediction[valid]
    cm += np.bincount(encoded, minlength=cm.size).reshape(cm.shape)


def read_bank(path):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    names = tuple(c['name'] for c in bank['classes'])
    groups = tuple(tuple(c['synonyms']) for c in bank['classes'])
    if not all(groups):
        raise ValueError('empty alias group')
    return bank, names, groups


def make_vocab_config(args, bank_path, g0):
    info = DATASET_INFO[args.dataset]
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    bank, names, groups = read_bank(bank_path)
    vocab_path = root / f'{args.dataset}_{args.bank_name}.vocab.json'
    vocab = dict(dataset=args.dataset, family='natural', banks={'custom': bank}, residual=None)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    config_path = root / f'{args.dataset}_{args.bank_name}.yaml'
    import yaml
    cfg = dict(schema=1, dataset=args.dataset, family='natural', total_images=info['total_images'],
               background_index=info['background'], vocabulary=vocab_path.name,
               profile=dict(bank='custom', strength='original', coupling=float(g0),
                            temperature=info['temperature'], tau=info['tau'], tem=info['tem'],
                            wide_policy='natural_short336_cap672',
                            background_rule=info['background_rule']),
               background_bias=info['background_bias'], background_threshold=info['background_threshold'],
               geometry=dict(geometry_depth=2, geometry_temperature=.1,
                             spatial_sigma=.25, maximum_aliases_per_class=20),
               visual_budget=dict(geometry_max=4, wide_max=4, fine=0),
               upstream=dict(vip_commit='5bd25ee03ec25c1538622cf7da661e8c0461e769',
                             dinov3_commit='f6704c1b76137543328567c8f8b49a2dc318824c'))
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return config_path, vocab_path


def make_vocab_config_batch(args, bank_paths, g0):
    """Build one FrozenGeometry instance containing all selector banks."""
    info = DATASET_INFO[args.dataset]
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    records = {name: read_bank(path)[0] for name, path in bank_paths.items()}
    vocab_path = root / f'{args.dataset}_allbanks.vocab.json'
    vocab_path.write_text(json.dumps(dict(dataset=args.dataset, family='natural', banks=records, residual=None),
                                      ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    config_path = root / f'{args.dataset}_allbanks.yaml'
    import yaml
    first = next(iter(bank_paths))
    cfg = dict(schema=1, dataset=args.dataset, family='natural', total_images=info['total_images'],
               background_index=info['background'], vocabulary=vocab_path.name,
               profile=dict(bank=first, strength='original', coupling=float(g0),
                            temperature=info['temperature'], tau=info['tau'], tem=info['tem'],
                            wide_policy='natural_short336_cap672', background_rule=info['background_rule']),
               background_bias=info['background_bias'], background_threshold=info['background_threshold'],
               geometry=dict(geometry_depth=2, geometry_temperature=.1, spatial_sigma=.25,
                             maximum_aliases_per_class=20),
               visual_budget=dict(geometry_max=4, wide_max=4, fine=0),
               upstream=dict(vip_commit='5bd25ee03ec25c1538622cf7da661e8c0461e769',
                             dinov3_commit='f6704c1b76137543328567c8f8b49a2dc318824c'))
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return config_path, vocab_path


def make_vip(args, bank_path):
    info = DATASET_INFO[args.dataset]
    ckpt = checkpoints(args)
    vip = FiniteVIPObserver(DINOTextSegmenter(ckpt, device=args.device, amp=True), Path(args.upstream_root))
    bank, names, groups = read_bank(bank_path)
    prompt = _load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py',
                            'cross_selector_vip_prompts')
    # All three banks are encoded with the same ImageNet80 prompt family as
    # the Qwen candidate pool; selectors therefore change aliases only.
    template = bank.get('template', 'ImageNet80')
    template = 'openai_imagenet_template' if template == 'ImageNet80' else template
    vip.templates = prompt.get_text_template(template)
    query = vip.encode_queries(names, groups)
    settings, _ = official_options(Path(args.upstream_root), args.dataset)
    return vip, query, settings


def run_vip(args, bank_path, phase):
    vip, query, settings = make_vip(args, bank_path)
    samples = discover_samples(args.dataset, args.data_root)
    dev_record = json.loads(Path(args.dev_keys).read_text()) if args.dev_keys else {}
    dev_keys = set(dev_record.get('keys', dev_record.get('development_keys', [])))
    selected = [s for s in samples if s.key in dev_keys] if phase == 'dev' and dev_keys else samples
    if phase == 'dev' and not selected:
        selected = samples[:64]
    cm = np.zeros((len(query.class_names), len(query.class_names)), dtype=np.int64)
    started = time.perf_counter()
    for i, sample in enumerate(selected, 1):
        image = load_rgb(sample)
        prediction, _ = official_prediction(image, vip, query, settings)
        if phase == 'full':
            target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
            update(cm, prediction, target)
        if i == 1 or i % 20 == 0 or i == len(selected):
            print(json.dumps(dict(model='VIP', dataset=args.dataset, bank=args.bank_name,
                                  phase=phase, processed=i, total=len(selected))), flush=True)
    return dict(status='complete', model='VIP', dataset=args.dataset, bank=args.bank_name,
                aliases=[int((query.parents == c).sum()) for c in range(len(query.class_names))],
                phase=phase, images=len(selected), mean_iou_percent=metric(cm) if phase == 'full' else None,
                confusion_matrix=cm.tolist() if phase == 'full' else None,
                wall_seconds=time.perf_counter()-started,
                settings=getattr(settings, '__dict__', str(settings)))


def make_geov2(args, bank_path, g0):
    config_path, vocab_path = make_vocab_config(args, bank_path, g0)
    cache = Path(args.work_root) / f'{args.dataset}_{args.bank_name}.text_cache.pt'
    model = FrozenGeometry(config_path, checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo, upstream_root=args.upstream_root,
        text_cache=cache, device=args.device, compact=True, efficient=True,
        wide_branch='query_slot', query_slot_capacity=24,
        wide_alias_aggregation=args.wide_alias_aggregation)
    return model


def run_geov2(args, bank_path, phase, g0):
    model = make_geov2(args, bank_path, g0)
    samples = discover_samples(args.dataset, args.data_root)
    dev_record = json.loads(Path(args.dev_keys).read_text()) if args.dev_keys else {}
    dev_keys = set(dev_record.get('keys', dev_record.get('development_keys', [])))
    selected = [s for s in samples if s.key in dev_keys] if phase == 'dev' and dev_keys else samples
    if phase == 'dev' and not selected:
        selected = samples[:64]
    class_names = model.banks['custom'].class_names
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    started = time.perf_counter()
    for i, sample in enumerate(selected, 1):
        image = load_rgb(sample)
        prediction = model.predict(image)
        if phase == 'full':
            target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
            update(cm, prediction, target)
        if i == 1 or i % 20 == 0 or i == len(selected):
            print(json.dumps(dict(model='Geov2Anchored', dataset=args.dataset, bank=args.bank_name,
                                  phase=phase, processed=i, total=len(selected), g0=g0)), flush=True)
    return dict(status='complete', model='Geov2Anchored', dataset=args.dataset, bank=args.bank_name,
                phase=phase, g0=g0, aliases=[int((model.reader.banks['custom'].parent_indices == c).sum())
                                              for c in range(len(class_names))],
                images=len(selected), mean_iou_percent=metric(cm) if phase == 'full' else None,
                confusion_matrix=cm.tolist() if phase == 'full' else None,
                wall_seconds=time.perf_counter()-started,
                profile=model.profile.record(), cache_identity=model.cache_identity)


def vip_predict_all(image, vip, queries, settings):
    """Replay the same visual crops for all text banks."""
    original = vip.crop_patch_features
    features = []
    position = 0
    capturing = True

    def observed(crop):
        nonlocal position
        if capturing:
            value = original(crop)
            features.append(value)
            return value
        if position >= len(features):
            raise RuntimeError('VIP visual replay consumed too many crops.')
        value = features[position]
        position += 1
        return value

    vip.crop_patch_features = observed
    predictions = {}
    try:
        for name, query in queries.items():
            position = 0
            predictions[name] = official_prediction(image, vip, query, settings)[0]
            if not capturing and position != len(features):
                raise RuntimeError('VIP visual replay crop count changed.')
            capturing = False
    finally:
        vip.crop_patch_features = original
    return predictions


def run_vip_batch(args, bank_paths, phase):
    info = DATASET_INFO[args.dataset]
    ckpt = checkpoints(args)
    vip = FiniteVIPObserver(DINOTextSegmenter(ckpt, device=args.device, amp=True), Path(args.upstream_root))
    prompt = _load_upstream(Path(args.upstream_root)/'prompts/imagenet_template.py', 'cross_selector_vip_prompts')
    queries = {}
    for name, path in bank_paths.items():
        bank, names, groups = read_bank(path)
        template = 'openai_imagenet_template' if bank.get('template') == 'ImageNet80' else bank.get('template', 'seg_template')
        vip.templates = prompt.get_text_template(template)
        queries[name] = vip.encode_queries(names, groups)
    settings, _ = official_options(Path(args.upstream_root), args.dataset)
    samples = discover_samples(args.dataset, args.data_root)
    dev_record = json.loads(Path(args.dev_keys).read_text()) if args.dev_keys else {}
    dev_keys = set(dev_record.get('keys', dev_record.get('development_keys', [])))
    selected = [s for s in samples if s.key in dev_keys] if phase == 'dev' and dev_keys else samples
    if phase == 'dev' and not selected:
        selected = samples[:64]
    matrices = {n: np.zeros((len(q.class_names), len(q.class_names)), dtype=np.int64) for n,q in queries.items()}
    started = time.perf_counter()
    for i, sample in enumerate(selected, 1):
        image = load_rgb(sample)
        predictions = vip_predict_all(image, vip, queries, settings)
        if phase == 'full':
            target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
            for name, prediction in predictions.items():
                update(matrices[name], prediction, target)
        if i == 1 or i % 20 == 0 or i == len(selected):
            print(json.dumps(dict(model='VIP', dataset=args.dataset, phase=phase,
                                  processed=i, total=len(selected), banks=list(queries))), flush=True)
    return dict(status='complete', model='VIP', dataset=args.dataset, phase=phase,
                results={name: dict(mean_iou_percent=metric(cm) if phase == 'full' else None,
                                    confusion_matrix=cm.tolist() if phase == 'full' else None,
                                    aliases=[int((q.parents == c).sum()) for c in range(len(q.class_names))])
                         for (name, q), cm in zip(queries.items(), matrices.values())},
                images=len(selected), wall_seconds=time.perf_counter()-started,
                settings=getattr(settings, '__dict__', str(settings)))


def run_geov2_batch(args, bank_paths, phase, g0):
    config_path, _ = make_vocab_config_batch(args, bank_paths, g0)
    cache = Path(args.work_root) / f'{args.dataset}_allbanks.text_cache.pt'
    model = FrozenGeometry(config_path, checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo, upstream_root=args.upstream_root,
        text_cache=cache, device=args.device, compact=True, efficient=True,
        wide_branch='query_slot', query_slot_capacity=24,
        wide_alias_aggregation=args.wide_alias_aggregation)
    from dinotool.query_slot_memory import query_slot_wide_observations
    from dinotool.efficient_deployment import reconstruction, write_dense, restore_and_finalize, GroupedLocalScorer
    from dinotool.natural_static_search import StaticReader
    info = DATASET_INFO[args.dataset]
    readers = {name: StaticReader({name: model.banks[name]}, {name: model.queries[name]}, info['background'])
               for name in bank_paths}
    scorers = {name: {name: GroupedLocalScorer(model.banks[name])} for name in bank_paths}
    profiles = {name: replace(model.profile, bank=name, coupling=float(g0)) for name in bank_paths}
    samples = discover_samples(args.dataset, args.data_root)
    dev_record = json.loads(Path(args.dev_keys).read_text()) if args.dev_keys else {}
    dev_keys = set(dev_record.get('keys', dev_record.get('development_keys', [])))
    selected = [s for s in samples if s.key in dev_keys] if phase == 'dev' and dev_keys else samples
    if phase == 'dev' and not selected:
        selected = samples[:64]
    matrices = {name: np.zeros((len(model.banks[name].class_names), len(model.banks[name].class_names)), dtype=np.int64)
                for name in bank_paths}
    started = time.perf_counter()
    for i, sample in enumerate(selected, 1):
        image = load_rgb(sample)
        source = query_slot_wide_observations(image, model.compact_geometry, model.query_slot_observer,
            model.strengths, wide_policy=model.profile.wide_policy,
            temperature=model.query_slot_temperature, beta=model.query_slot_beta,
            reconstruction=reconstruction)
        predictions = {}
        for name in bank_paths:
            def finalize(normalized, size, _bg=info['background']):
                return restore_and_finalize(normalized, size, model.config['background_bias'],
                                            model.config['background_threshold'], _bg)
            predictions[name] = readers[name].probabilities(
                source, profiles[name], {}, wide_scorer=model.wide_scorer,
                fusion=model.writeback, dense_writer=write_dense,
                local_scorers=scorers[name], output_finalizer=finalize)
        if phase == 'full':
            target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
            for name, prediction in predictions.items():
                update(matrices[name], prediction, target)
        if i == 1 or i % 20 == 0 or i == len(selected):
            print(json.dumps(dict(model='Geov2Anchored', dataset=args.dataset, phase=phase,
                                  processed=i, total=len(selected), g0=g0, banks=list(bank_paths))), flush=True)
        del source, predictions
    return dict(status='complete', model='Geov2Anchored', dataset=args.dataset, phase=phase, g0=g0,
                results={name: dict(mean_iou_percent=metric(cm) if phase == 'full' else None,
                                    confusion_matrix=cm.tolist() if phase == 'full' else None,
                                    aliases=[int((model.banks[name].parent_indices == c).sum())
                                             for c in range(len(model.banks[name].class_names))])
                         for name, cm in matrices.items()},
                images=len(selected), wall_seconds=time.perf_counter()-started,
                profile=model.profile.record(), cache_identity=model.cache_identity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('vip','geov2'), required=True)
    p.add_argument('--dataset', choices=tuple(DATASET_INFO), required=True)
    p.add_argument('--bank-name', default='')
    p.add_argument('--bank-path', default='')
    p.add_argument('--bank-spec', default='', help='name=path;name=path;name=path')
    p.add_argument('--data-root', required=True)
    p.add_argument('--dev-keys', default='')
    p.add_argument('--phase', choices=('dev','full'), required=True)
    p.add_argument('--g0', type=float, default=.5)
    p.add_argument('--wide-alias-aggregation', choices=('logmeanexp', 'logsumexp'),
                   default='logmeanexp',
                   help='Alias reduction in the Geov2 QuerySlot wide branch. '
                        'Use the same bank/configuration when comparing reductions.')
    p.add_argument('--work-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dinov3-repo', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--upstream-root', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.bank_spec:
        bank_paths = dict(item.split('=', 1) for item in args.bank_spec.split(';'))
        result = run_vip_batch(args, bank_paths, args.phase) if args.model == 'vip' else run_geov2_batch(args, bank_paths, args.phase, args.g0)
    else:
        if not args.bank_name or not args.bank_path:
            raise ValueError('--bank-name and --bank-path are required without --bank-spec')
        result = run_vip(args, args.bank_path, args.phase) if args.model == 'vip' else run_geov2(args, args.bank_path, args.phase, args.g0)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), metric=result.get('mean_iou_percent'), status='complete')), flush=True)


if __name__ == '__main__':
    main()
