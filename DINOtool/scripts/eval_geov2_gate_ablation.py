"""Same-input Geov2 structure-writer and gate ablation.

The script deliberately keeps one frozen profile, vocabulary, image list and
backbone for every row.  ``OfficialVIP`` is the external VIP-style endpoint
with the pinned upstream wide observer; it is not a Geov2 component.
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
from eval_context_memory import save
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary
from eval_kev_alias_search import task_entry, load_models, predict


METHODS = (
    'Geov2_VIPWide', 'Raw32', 'QuerySlot_Raw', 'DonorBefore',
    'Geov2_Full', 'Geov2_Target', 'Geov2_FixedG2',
    'Geov2_AdaptiveGate', 'Geov2_NoAlias_Canonical',
    'Geov2_EqualFusion', 'Geov2_ShuffledQuery',
)


def canonical_pair(banks, queries, bank_name):
    """Return a bank/query pair with exactly one canonical phrase per class."""
    bank, query = banks[bank_name], queries[bank_name]
    keep = bank.canonical_mask
    if int(keep.sum()) != bank.class_count:
        raise RuntimeError('Canonical alias mask does not contain one phrase per class.')
    cb = TCPRTextBank(
        bank.features[keep], bank.parent_indices[keep],
        torch.ones(int(keep.sum()), dtype=torch.bool, device=bank.features.device),
        bank.class_names, tuple(alias for alias, flag in zip(bank.alias_names, keep.tolist()) if flag),
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
    args.mode, args.device, args.num_shards, args.shard_index = 'full', 'cuda', 1, 0
    # The pinned remote VIP checkout's zero-argument tokenizer needs the local
    # checkpoint vocabulary when the script is run outside its original repo.
    from dinov3.eval.text import tokenizer as _dino_tokenizer
    _dino_tokenizer.get_tokenizer = lambda *_a, **_k: _dino_tokenizer.Tokenizer(
        vocab_path=str(Path(args.checkpoint_dir) / 'bpe_simple_vocab_16e6.txt.gz'))

    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve completed experiment: ' + str(output))
    output.mkdir(parents=True)
    _, entry, _ = task_entry(args)
    validation, _, loader, target_loader, protocols = load_samples(args, entry)
    lookup = {s.key: s for s in validation}
    protocol = record.get('evaluation_protocol', next(iter(protocols)))
    keys = record['sample_keys'][:args.limit] if args.limit else record['sample_keys']
    profile = StaticProfile(**record['profile'])
    geometry, vip, banks, queries, _, residual = load_models(args, entry, (profile.bank,))
    if residual is not None:
        raise ValueError('Residual ontology is outside this isolated ablation.')

    compact = CompactGeometry(geometry)
    response = ResponseGeometry(compact)
    observer = ContextMemoryObserver(vip.backbone, ContextMemoryConfig(center=False,
                                                                        capacity=args.capacity,
                                                                        temperature=.1,
                                                                        iterations=5))
    reader = StaticReader(banks, queries, entry['background_index'])
    canonical_bank, canonical_query = canonical_pair(banks, queries, profile.bank)
    canonical_reader = StaticReader({profile.bank: canonical_bank},
                                    {profile.bank: canonical_query},
                                    entry['background_index'])
    vip_scorer, raw_scorer, query_scorer = CachedWideScorer(), CompactContextScorer(), CompactContextScorer()
    names = banks[profile.bank].class_names
    matrices = {m: np.zeros((len(names), len(names)), np.int64) for m in METHODS}
    per_image = {m: [] for m in METHODS}

    def probs(source, scorer, *, selected_reader=reader, fusion='coupled'):
        return selected_reader.probabilities(source, profile, {}, wide_scorer=scorer,
                                             fusion=fusion)

    for number, key in enumerate(keys, 1):
        image = loader(lookup[key])
        # Official VIP endpoint: Geometry local readout + pinned VIP wide view.
        vip_source = compact_observations(image, compact, vip, (profile.strength,),
                                          wide_policy=profile.wide_policy)
        raw_wide = memory_wide_observations(image, observer, wide_policy=profile.wide_policy)
        raw_source = dict(vip_source, **raw_wide)
        query_raw = query_slot_wide_observations(
            image, compact, observer, (profile.strength,), wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta)
        donor_source = compact_observations(image, response, vip, (profile.strength,),
                                            wide_policy=profile.wide_policy)
        full_source = query_slot_wide_observations(
            image, response, observer, (profile.strength,), wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta)
        shuffled_source = query_slot_wide_observations(
            image, response, observer, (profile.strength,), wide_policy=profile.wide_policy,
            temperature=args.query_temperature, beta=args.beta, shuffle_query=True)

        predictions = {
            'Geov2_VIPWide': predict(probs(vip_source, vip_scorer), record, entry['background_index']),
            'Raw32': predict(probs(raw_source, raw_scorer), record, entry['background_index']),
            'QuerySlot_Raw': predict(probs(query_raw, query_scorer), record, entry['background_index']),
            'DonorBefore': predict(probs(donor_source, vip_scorer), record, entry['background_index']),
            'Geov2_Full': predict(probs(full_source, query_scorer), record, entry['background_index']),
            'Geov2_Target': predict(
                probs(full_source, query_scorer, fusion='target'), record, entry['background_index']),
            'Geov2_FixedG2': predict(
                probs(full_source, query_scorer, fusion='fixed_g2'), record, entry['background_index']),
            'Geov2_AdaptiveGate': predict(
                probs(full_source, query_scorer, fusion='adaptive'), record, entry['background_index']),
            'Geov2_NoAlias_Canonical': predict(
                probs(full_source, query_scorer, selected_reader=canonical_reader),
                record, entry['background_index']),
            'Geov2_EqualFusion': predict(
                probs(full_source, query_scorer, fusion='equal_mean'),
                record, entry['background_index']),
            'Geov2_ShuffledQuery': predict(
                probs(shuffled_source, query_scorer), record, entry['background_index']),
        }
        target = target_loader(lookup[key], protocol, tuple(image.shape[-2:]))
        for method in METHODS:
            cm = confusion(predictions[method], target, len(names))
            matrices[method] += cm
            per_image[method].append(cm)
        save(output / 'results.json', dict(
            status='complete' if number == len(keys) else 'running', dataset=args.dataset,
            processed_images=number, total_images=len(keys), sample_keys=keys,
            protocol=protocol, capacity=args.capacity, query_temperature=args.query_temperature,
            beta=args.beta, methods=METHODS,
            metrics={m: summary(cm, names, 0) for m, cm in matrices.items()},
            external_baseline_note='Geov2_VIPWide is the internal VIP-wide endpoint. Official VIP full reproduction is an external baseline and is reported separately.',
        ))
        print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(keys))), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{m: np.stack(v) for m, v in per_image.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'experiment', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--capacity', type=int, default=24)
    p.add_argument('--query-temperature', type=float, default=.1)
    p.add_argument('--beta', type=float, default=.5)
    main(p.parse_args())
