"""Matched full-validation fixed-step sweep with controlled alias interfaces.

Encode each image once per view, hold observations fixed, and reuse exact
local/wide scores across all declared gains. No mask is read until inference
has completed for every arm of that image.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.frozen_deployment import FrozenGeometry
from dinotool.natural_static_search import StaticReader
from dinotool.efficient_deployment import GroupedLocalScorer, reconstruction, write_dense, restore_and_finalize
from dinotool.query_slot_memory import query_slot_wide_observations
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_alias_finalization import load_samples
from eval_development_readout import confusion


GAINS = (0., .25, .5, .75, 1., 1.5, 2., 3.)
ALIAS_MODES = ('retained', 'canonical', 'count_normalized')


def canonical_bank_query(bank, query):
    keep = bank.canonical_mask
    cb = TCPRTextBank(bank.features[keep], bank.parent_indices[keep],
                     torch.ones(int(keep.sum()), dtype=torch.bool, device=keep.device),
                     bank.class_names, tuple(a for a,k in zip(bank.alias_names, keep.tolist()) if k))
    cq = VIPQueries(query.features[keep], query.parents[keep], query.class_names, cb.alias_names)
    cb.validate()
    return cb, cq


class CountNormalizedWide:
    """Remove only the class-specific log alias count / tau offset."""
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, source, query, tau, tem):
        broad = self.scorer(source, query, tau, tem)
        counts = torch.bincount(query.parents, minlength=len(query.class_names)).float()
        offset = counts.log() / tau
        if isinstance(broad, dict):
            return {k: v-offset[None] for k,v in broad.items()}
        return broad-offset[:, None, None]


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(path)


def metric(cm):
    tp = cm.diagonal()
    union = cm.sum(0)+cm.sum(1)-tp
    return float(np.nanmean(np.divide(tp, union, out=np.full(len(tp), np.nan), where=union>0))*100)


@torch.inference_mode()
def main(args):
    root = Path(args.experiment)
    protocol = json.loads((root/'protocol.json').read_text())
    record = protocol['datasets'][args.dataset]
    out = root/args.mode/(args.dataset+f'_s{args.shard_index}')
    if out.exists():
        raise RuntimeError('Preserve output: '+str(out))
    out.mkdir(parents=True)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index.')
    from dinov3.eval.text import tokenizer
    tokenizer.get_tokenizer = lambda *_a, **_k: tokenizer.Tokenizer(
        vocab_path=str(Path(protocol['checkpoint_dir'])/'bpe_simple_vocab_16e6.txt.gz'))
    model = FrozenGeometry(record['config_path'], checkpoint_dir=protocol['checkpoint_dir'],
        dinov3_repo=protocol['dinov3_repo'], upstream_root=protocol['upstream_root'],
        text_cache=record['text_cache'], efficient=True, wide_branch='query_slot', query_slot_capacity=24)
    if model.strengths != ('original',):
        raise ValueError('DonorBefore must execute for every dataset.')
    if model.profile.background_rule == 'residual_protected':
        raise ValueError('g-dependent foreground protection is outside this matched sweep.')
    args.data_root = record['data_root']
    args.sample_seed, args.vdd_ontology = 20260923, 'official'
    entry = dict(model.config, banks={'original':next(iter(model.vocabulary['banks'].values()))})
    args.mode = 'full' if args.mode == 'full' else 'smoke'
    validation, samples, images, masks, names_map = load_samples(args, entry)
    if len(validation) != record['total_images']:
        raise ValueError('Full dataset size changed.')
    if args.mode == 'smoke':
        samples = samples[:1]
    names = names_map[args.dataset]
    bank_name = model.profile.bank
    cb, cq = canonical_bank_query(model.banks[bank_name], model.queries[bank_name])
    canonical_reader = StaticReader({bank_name:cb}, {bank_name:cq},
        model.config['background_index'], model.reader.residual_features)
    canonical_scorers = {bank_name:GroupedLocalScorer(cb)}
    normalized_scorer = CountNormalizedWide(model.wide_scorer)
    arms = [(mode, 'g'+format(g, 'g'), g) for mode in ALIAS_MODES for g in GAINS]
    arms += [(mode, fusion, None) for mode in ALIAS_MODES for fusion in ('equal_mean','wide')]
    methods = [mode+'__'+key for mode,key,_ in arms]
    matrices = np.zeros((len(arms), len(names), len(names)), dtype=np.int64)
    tps, predictions, targets, keys = [], [], [], []
    diagnostics = {mode:dict(displacement_energy=0., correction_energy=0., patches=0) for mode in ALIAS_MODES}
    started = time.perf_counter()

    for number, sample in enumerate(samples, 1):
        image = images(sample)
        source = query_slot_wide_observations(image, model.compact_geometry, model.query_slot_observer,
            ('original',), wide_policy=model.profile.wide_policy, temperature=.1, beta=.5,
            reconstruction=reconstruction)
        caches = {mode:{} for mode in ALIAS_MODES}
        actual = []
        for alias_mode, key, gain in arms:
            selected_reader = canonical_reader if alias_mode == 'canonical' else model.reader
            local_scorers = canonical_scorers if alias_mode == 'canonical' else model.efficient_local_scorers
            wide_scorer = normalized_scorer if alias_mode == 'count_normalized' else model.wide_scorer
            selected_profile = model.profile if gain is None else replace(model.profile, coupling=gain)
            actual.append(selected_reader.probabilities(
                source, selected_profile, caches[alias_mode], wide_scorer=wide_scorer,
                fusion='coupled' if gain is not None else key,
                dense_writer=write_dense, local_scorers=local_scorers,
                output_finalizer=lambda normalized,size:restore_and_finalize(normalized,size,
                    model.config['background_bias'], model.config['background_threshold'],
                    model.config['background_index'])))
        # Score-space diagnostics use no labels and share the same branch
        # responses. They are explanatory covariates, not fitted selectors.
        from eval_geometry_vip_reliability import sample_broad
        for alias_mode in ALIAS_MODES:
            cache = caches[alias_mode]
            broad = cache[('wide', bank_name, model.profile.tau, model.profile.tem)]
            if model.profile.background_rule.startswith('residual_'):
                broad = broad.clone()
                broad[model.config['background_index']] = cache[('residual_wide',)]
            for i,tile in enumerate(source['local']):
                local = cache[('local',bank_name,'original',model.profile.background_rule,i)]/model.profile.temperature
                top,left = tile['top'],tile['left']
                wide = sample_broad(broad,top,left,*source['size']).reshape_as(local)
                difference = wide.double()-local.double()
                correction = tile['operator'].double() @ difference
                diagnostics[alias_mode]['displacement_energy'] += float(difference.square().sum())
                diagnostics[alias_mode]['correction_energy'] += float(correction.square().sum())
                diagnostics[alias_mode]['patches'] += len(local)
        target = masks(sample, args.dataset, tuple(image.shape[-2:]))
        cms = np.stack([confusion(p,target,len(names)) for p in actual])
        matrices += cms
        tps.append(np.diagonal(cms,axis1=1,axis2=2).copy())
        predictions.append(cms.sum(1))
        targets.append(cms[0].sum(1))
        if not np.all(cms.sum(2) == cms[0].sum(1)[None]):
            raise RuntimeError('Target support differs between arms.')
        keys.append(sample.key)
        if number == 1 or number % 25 == 0 or number == len(samples):
            progress = dict(dataset=args.dataset, processed=number, total=len(samples),
                            shard_index=args.shard_index, num_shards=args.num_shards)
            write(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    np.savez_compressed(out/'statistics.npz', sample_keys=np.asarray(keys),
        tp=np.stack(tps), predicted=np.stack(predictions), target=np.stack(targets))
    write(out/'results.json',dict(status='complete', dataset=args.dataset,
        processed_images=len(keys), global_sample_count=len(validation), sample_keys=keys,
        shard_index=args.shard_index,num_shards=args.num_shards, methods=methods,
        gains=GAINS,alias_modes=ALIAS_MODES,profile=model.config['profile'],
        config=model.config, cache_identity=model.cache_identity,
        class_names=names, diagnostics=diagnostics,
        alias_counts=torch.bincount(model.banks[bank_name].parent_indices).cpu().tolist(),
        residual_concepts=0 if model.reader.residual_features is None else len(model.reader.residual_features),
        metrics={m:dict(mean_iou_percent=metric(cm),confusion_matrix=cm.tolist()) for m,cm in zip(methods,matrices)},
        masks_used_only_for_scoring=True, both_modules_active=True,
        wall_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--mode',choices=('full','smoke'),default='full')
    main(p.parse_args())
