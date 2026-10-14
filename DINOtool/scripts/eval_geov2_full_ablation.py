"""Full frozen Geov2 evaluation with matched writer and gate ablations.

Every row for one image reuses the same local features, wide observation,
alias bank and background rule.  The script is explicitly sharded by the
authoritative validation list; the 16-image development screen is never used
as the full split.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dinotool.compact_deployment import CompactGeometry, compact_observations, CachedWideScorer
from dinotool.context_memory_deployment import memory_wide_observations
from dinotool.context_memory_observer import ContextMemoryConfig, ContextMemoryObserver, CompactContextScorer
from dinotool.natural_static_search import StaticProfile, StaticReader
from dinotool.query_slot_memory import query_slot_wide_observations
from dinotool.response_geometry_deployment import ResponseGeometry
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary
from eval_kev_alias_search import task_entry, load_models, predict


METHODS = (
    'VIPWide', 'Raw24Slots', 'QuerySlot_Raw', 'DonorBefore',
    'LocalEndpoint', 'WideEndpoint', 'EqualFusion',
    'Geov2_FixedTarget', 'Geov2_AdaptiveGate',
    'AdaptiveMarginOnly', 'AdaptiveSupportOnly',
    'AdaptiveShuffledGate', 'AdaptiveConstantGate',
    'Geov2_CanonicalOnly', 'Geov2_ShuffledQuery',
)
PER_IMAGE = (
    'Geov2_FixedTarget', 'Geov2_AdaptiveGate', 'EqualFusion',
    'AdaptiveMarginOnly', 'AdaptiveSupportOnly', 'AdaptiveShuffledGate',
)
SUPPLEMENT = (
    'AdaptiveReference', 'DoubleBaseStep', 'QuadraticOptimum',
    'CanonicalAdaptive', 'NoDonorAdaptive', 'NoRoutingAdaptive', 'RandomGate',
)
ACTIVE_LOCAL = ('OriginalGeometryAdaptive', 'DonorBeforeActiveAdaptive')


def canonical_pair(banks, queries, bank_name):
    bank, query = banks[bank_name], queries[bank_name]
    keep = bank.canonical_mask
    if int(keep.sum()) != bank.class_count:
        raise RuntimeError('Canonical alias mask does not contain one phrase per class.')
    cb = TCPRTextBank(
        bank.features[keep], bank.parent_indices[keep],
        torch.ones(int(keep.sum()), dtype=torch.bool, device=bank.features.device),
        bank.class_names,
        tuple(alias for alias, flag in zip(bank.alias_names, keep.tolist()) if flag),
    )
    cq = VIPQueries(query.features[keep], query.parents[keep], query.class_names,
                    tuple(alias for alias, flag in zip(query.aliases, keep.tolist()) if flag))
    cb.validate()
    return cb, cq


@torch.inference_mode()
def main(args):
    manifest = json.loads(Path(args.experiment).read_text())
    record = manifest['datasets'][args.dataset]
    for key, value in record['loader'].items():
        setattr(args, key, value)
    args.mode, args.device = 'full', 'cuda'
    args.num_shards, args.shard_index = args.shards, args.shard
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('Require 0 <= shard < shards and shards >= 1.')

    from dinov3.eval.text import tokenizer as _dino_tokenizer
    _dino_tokenizer.get_tokenizer = lambda *_a, **_k: _dino_tokenizer.Tokenizer(
        vocab_path=str(Path(args.checkpoint_dir) / 'bpe_simple_vocab_16e6.txt.gz'))

    output = Path(args.output_dir)
    if output.exists() and (output / 'results.json').exists():
        raise RuntimeError('Preserve completed experiment: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)

    _, entry, _ = task_entry(args)
    validation, selected, loader, target_loader, protocols = load_samples(args, entry)
    if len(validation) != entry['total_images']:
        raise RuntimeError('Validation count differs from frozen dataset protocol.')
    if not selected:
        raise RuntimeError('Empty shard.')
    keys = [s.key for s in selected]
    protocol = record.get('evaluation_protocol', next(iter(protocols)))
    profile = StaticProfile(**record['profile'])
    if args.arm_set == 'active-local':
        profile = replace(profile, strength='original')
    geometry, vip, banks, queries, _, residual = load_models(args, entry, (profile.bank,))
    if residual is not None:
        raise ValueError('Residual ontology is outside this isolated ablation.')

    compact = CompactGeometry(geometry)
    response = ResponseGeometry(compact)
    observer = ContextMemoryObserver(
        vip.backbone,
        ContextMemoryConfig(center=False, capacity=args.capacity,
                            temperature=.1, iterations=5),
    )
    reader = StaticReader(banks, queries, entry['background_index'])
    canonical_bank, canonical_query = canonical_pair(banks, queries, profile.bank)
    canonical_reader = StaticReader(
        {profile.bank: canonical_bank}, {profile.bank: canonical_query},
        entry['background_index'])
    vip_scorer, raw_scorer = CachedWideScorer(), CompactContextScorer()
    names = banks[profile.bank].class_names
    methods = {'primary': METHODS, 'supplement': SUPPLEMENT, 'active-local': ACTIVE_LOCAL}[args.arm_set]
    stored = PER_IMAGE if args.arm_set == 'primary' else methods
    matrices = {m: np.zeros((len(names), len(names)), np.int64) for m in methods}
    per_image = {m: [] for m in stored}

    def probs(source, scorer, *, selected_reader=reader, fusion='coupled', selected_profile=profile):
        # A fresh cache is deliberate: each row is a matched readout from the
        # same source and does not accidentally reuse a transformed logit.
        return selected_reader.probabilities(
            source, selected_profile, {}, wide_scorer=scorer, fusion=fusion)

    for number, sample in enumerate(selected, 1):
        key = sample.key
        image = loader(sample)
        if args.arm_set == 'active-local':
            full_source = query_slot_wide_observations(
                image, response, observer, (profile.strength,),
                wide_policy=profile.wide_policy,
                temperature=args.query_temperature, beta=args.beta)
            query_raw = query_slot_wide_observations(
                image, compact, observer, (profile.strength,),
                wide_policy=profile.wide_policy,
                temperature=args.query_temperature, beta=args.beta)
            predictions = {
                'OriginalGeometryAdaptive': predict(probs(query_raw, raw_scorer, fusion='adaptive'), record, entry['background_index']),
                'DonorBeforeActiveAdaptive': predict(probs(full_source, raw_scorer, fusion='adaptive'), record, entry['background_index']),
            }
        elif args.arm_set == 'supplement':
            full_source = query_slot_wide_observations(
                image, response, observer, (profile.strength,),
                wide_policy=profile.wide_policy,
                temperature=args.query_temperature, beta=args.beta)
            query_raw = query_slot_wide_observations(
                image, compact, observer, (profile.strength,),
                wide_policy=profile.wide_policy,
                temperature=args.query_temperature, beta=args.beta)
            raw_wide = memory_wide_observations(image, observer, wide_policy=profile.wide_policy)
            raw_source = dict(full_source, **raw_wide)
            predictions = {
                'AdaptiveReference': predict(probs(full_source, raw_scorer, fusion='adaptive'), record, entry['background_index']),
                'DoubleBaseStep': predict(probs(full_source, raw_scorer, fusion='fixed_g2'), record, entry['background_index']),
                'QuadraticOptimum': predict(probs(full_source, raw_scorer, selected_profile=replace(profile, coupling=1.)), record, entry['background_index']),
                'CanonicalAdaptive': predict(probs(full_source, raw_scorer, fusion='adaptive', selected_reader=canonical_reader), record, entry['background_index']),
                'NoDonorAdaptive': predict(probs(query_raw, raw_scorer, fusion='adaptive'), record, entry['background_index']),
                'NoRoutingAdaptive': predict(probs(raw_source, raw_scorer, fusion='adaptive'), record, entry['background_index']),
                'RandomGate': predict(probs(full_source, raw_scorer, fusion='adaptive_random'), record, entry['background_index']),
            }
        else:
            predictions = primary_predictions(image, compact, response, vip, observer, profile, args, probs, record, entry, canonical_reader, vip_scorer, raw_scorer)

        target = target_loader(sample, protocol, tuple(image.shape[-2:]))
        for method in methods:
            cm = confusion(predictions[method], target, len(names))
            matrices[method] += cm
            if method in per_image:
                per_image[method].append(cm.astype(np.int32, copy=False))
        metrics = {m: summary(cm, names, 0) for m, cm in matrices.items()}
        payload = {
            'status': 'complete' if number == len(keys) else 'running',
            'dataset': args.dataset, 'shard': args.shard, 'shards': args.shards,
            'processed_images': number, 'total_images': len(keys),
            'global_sample_count': len(validation), 'sample_keys': keys,
            'protocol': protocol, 'capacity': args.capacity,
            'query_temperature': args.query_temperature, 'beta': args.beta,
            'arm_set': args.arm_set, 'profile': profile.record(),
            'methods': methods, 'metrics': metrics,
            'full_split_note': 'Authoritative validation list, sharded before inference; no labels used for writer selection.',
        }
        temporary = output / 'results.tmp'
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(output / 'results.json')
        print(json.dumps({'dataset': args.dataset, 'shard': args.shard,
                          'processed': number, 'total': len(keys)}), flush=True)

    np.savez_compressed(
        output / 'per_image_confusions.npz', sample_keys=np.asarray(keys),
        **{m: np.stack(v).astype(np.int32, copy=False) for m, v in per_image.items()})


def primary_predictions(image, compact, response, vip, observer, profile, args, probs, record, entry, canonical_reader, vip_scorer, raw_scorer):
        vip_source = compact_observations(
            image, compact, vip, (profile.strength,),
            wide_policy=profile.wide_policy)
        raw_wide = memory_wide_observations(
            image, observer, wide_policy=profile.wide_policy)
        raw_source = dict(vip_source, **raw_wide)
        query_raw = query_slot_wide_observations(
            image, compact, observer, (profile.strength,),
            wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta)
        donor_source = compact_observations(
            image, response, vip, (profile.strength,),
            wide_policy=profile.wide_policy)
        full_source = query_slot_wide_observations(
            image, response, observer, (profile.strength,),
            wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta)
        shuffled_source = query_slot_wide_observations(
            image, response, observer, (profile.strength,),
            wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta,
            shuffle_query=True)

        predictions = {
            'VIPWide': predict(probs(vip_source, vip_scorer, fusion='coupled'), record, entry['background_index']),
            'Raw24Slots': predict(probs(raw_source, raw_scorer, fusion='coupled'), record, entry['background_index']),
            'QuerySlot_Raw': predict(probs(query_raw, raw_scorer, fusion='coupled'), record, entry['background_index']),
            'DonorBefore': predict(probs(donor_source, vip_scorer, fusion='coupled'), record, entry['background_index']),
            'LocalEndpoint': predict(probs(full_source, raw_scorer, fusion='local'), record, entry['background_index']),
            'WideEndpoint': predict(probs(full_source, raw_scorer, fusion='wide'), record, entry['background_index']),
            'EqualFusion': predict(probs(full_source, raw_scorer, fusion='equal_mean'), record, entry['background_index']),
            'Geov2_FixedTarget': predict(probs(full_source, raw_scorer, fusion='coupled'), record, entry['background_index']),
            'Geov2_AdaptiveGate': predict(probs(full_source, raw_scorer, fusion='adaptive'), record, entry['background_index']),
            'AdaptiveMarginOnly': predict(probs(full_source, raw_scorer, fusion='adaptive_margin'), record, entry['background_index']),
            'AdaptiveSupportOnly': predict(probs(full_source, raw_scorer, fusion='adaptive_support'), record, entry['background_index']),
            'AdaptiveShuffledGate': predict(probs(full_source, raw_scorer, fusion='adaptive_shuffled'), record, entry['background_index']),
            'AdaptiveConstantGate': predict(probs(full_source, raw_scorer, fusion='adaptive_constant'), record, entry['background_index']),
            'Geov2_CanonicalOnly': predict(probs(full_source, raw_scorer, selected_reader=canonical_reader), record, entry['background_index']),
            'Geov2_ShuffledQuery': predict(probs(shuffled_source, raw_scorer, fusion='adaptive'), record, entry['background_index']),
        }
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=24)
    parser.add_argument('--query-temperature', type=float, default=.1)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--arm-set', choices=('primary', 'supplement', 'active-local'), default='primary')
    main(parser.parse_args())
