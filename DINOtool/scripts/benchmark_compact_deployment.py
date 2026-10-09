"""Paired implementation-only latency/memory and prediction verification."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import eval_development_readout as base
from eval_alias_finalization import load_samples, foreground_banks
from eval_kev_alias_search import task_entry, load_models, predict, PREVIOUS
from dinotool.natural_static_search import StaticReader, StaticProfile, projected
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.compact_deployment import CompactGeometry, compact_observations, CachedWideScorer, cached_wide, offload_text_towers
from eval_rival_fine_full import save


@torch.inference_mode()
def main(args):
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Preserve existing deployment evidence.')
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol, entry, _ = task_entry(args)
    validation, _, loader, masks, names = load_samples(args, entry)
    choice = json.loads((Path(args.suite_root) / 'search' / args.dataset / 'selection.json').read_text())
    profile = StaticProfile(**choice['profile'])
    needed = ('semantic_segmentation',) if profile.bank == PREVIOUS else (profile.bank,)
    geometry, vip, banks, queries, _, residual = load_models(args, entry, needed)
    model = (TaxonomyInference.for_finalization(geometry, vip, banks, queries, ordinary_alias_policy='uniform',
        family=entry['family'], background=entry['background_index'], residual_features=residual)
        if profile.bank == PREVIOUS else None)
    reader = StaticReader(banks, queries, entry['background_index'], residual)
    strengths = model.strengths if model else (profile.strength,)
    compact = CompactGeometry(geometry)
    scorer = CachedWideScorer()

    def forward(image, fast):
        if fast:
            source = compact_observations(image, compact, vip, strengths, wide_policy=profile.wide_policy)
            with cached_wide(scorer):
                probability = model.predict_observations(source, return_probability=True,return_prediction=args.mode!='final')[2] if model else reader.probabilities(source, profile, {})
        elif model:
            probability = model.predict(image, return_probability=True)[2]
        else:
            old = base.projected
            base.projected = projected
            try:
                source = base.observations(image, geometry, vip, strengths, wide_policy=profile.wide_policy)
            finally:
                base.projected = old
            probability = reader.probabilities(source, profile, {})
        if fast and args.mode=='final':
            from dinotool.development_readout import biased_probabilities
            bg=entry['background_index']
            probability=biased_probabilities(probability,choice['background_bias'] if bg is not None else 0.,bg)
            confidence,prediction=probability.max(0)
            if bg is not None:prediction=prediction.masked_fill(confidence<choice['background_threshold'],bg)
            if probability.shape[0]>256:raise ValueError('Byte labels require <=256 classes.')
            return prediction.to(torch.uint8).cpu().numpy()
        return predict(probability, choice, entry['background_index'])

    if args.mode == 'interleaved':
        offload_text_towers(geometry,vip)
        groups={'reference':[], 'compact':[]}
        selected=[validation[int(i)] for i in np.unique(np.linspace(0,len(validation)-1,7,dtype=int))]
        for sample in selected:
            image=loader(sample)
            expected=forward(image,False)
            for _ in range(2):forward(image,False);forward(image,True)
            times={False:[],True:[]};peaks={False:[],True:[]}
            for repeat in range(7):
                for fast in ((False,True) if repeat%2==0 else (True,False)):
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                    actual=forward(image,fast);torch.cuda.synchronize()
                    times[fast].append((time.perf_counter()-started)*1000)
                    peaks[fast].append(torch.cuda.max_memory_allocated()/1048576)
                    if not np.array_equal(actual,expected):raise RuntimeError('Interleaved prediction differs.')
            for fast in (False,True):
                groups['compact' if fast else 'reference'].append(dict(key=sample.key,milliseconds=times[fast],
                    peak_allocated_mib=max(peaks[fast]),changed_pixels=0))
        save(output,dict(status='complete',dataset=args.dataset,images=len(selected),exact_sample_predictions=True,
            both_text_towers_on_cpu=True,alternating_order=True,results={k:dict(images=rows,
                median_ms=float(np.median([t for r in rows for t in r['milliseconds']])),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows)) for k,rows in groups.items()}))
        return

    if args.mode == 'full':
        offload_text_towers(geometry, vip)
        old = np.load(Path(args.suite_root) / 'full' / args.dataset / 's0' / 'per_image_confusions.npz')
        # Full runs may have multiple shards. Decompress each array once.
        expected = {}
        for path in sorted((Path(args.suite_root) / 'full' / args.dataset).glob('s*/per_image_confusions.npz')):
            with np.load(path) as arrays:
                keys = arrays['sample_keys'].tolist()
                matrices = {p: arrays[p+'__KevTuned'] for p in names}
                for i, key in enumerate(keys):
                    if key in expected:
                        raise RuntimeError('Duplicate baseline image key.')
                    expected[key] = {p: matrices[p][i] for p in names}
        del old
        if args.dataset == 'loveda':
            raise ValueError('LoveDA P requires independent six-class prediction; use sampled primary verification here.')
        matrices = {p: np.zeros((len(n), len(n)), np.int64) for p, n in names.items()}
        start = time.perf_counter()
        for index, sample in enumerate(validation, 1):
            image = loader(sample)
            actual = forward(image, True)
            for p, n in names.items():
                target = masks(sample, p, tuple(image.shape[-2:]))
                cm = base.confusion(actual, target, len(n))
                if not np.array_equal(cm, expected[sample.key][p]):
                    raise RuntimeError('Full per-image confusion differs: '+sample.key)
                matrices[p] += cm
            if index % 50 == 0:
                print(json.dumps(dict(dataset=args.dataset, processed=index, total=len(validation))), flush=True)
        from eval_geometry_vip_reliability import summary
        save(output, dict(status='complete', mode='full', dataset=args.dataset, images=len(validation),
            exact_per_image_confusion=True, sample_keys=[s.key for s in validation],
            metrics={p: summary(cm, names[p], 0) for p, cm in matrices.items()}, wall_seconds=time.perf_counter()-start))
        return

    selected = [validation[int(i)] for i in np.unique(np.linspace(0, len(validation)-1, 7, dtype=int))]
    reference, results = {}, {}
    text_offload = None
    for fast in (False, True):
        if fast:
            text_offload = offload_text_towers(geometry, vip,release=args.mode=='final')
            torch.cuda.empty_cache()
        rows = []
        for sample in selected:
            image = loader(sample)
            forward(image, fast); forward(image, fast); torch.cuda.synchronize()
            times, allocated, reserved = [], [], []
            for _ in range(7):
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start = time.perf_counter()
                actual = forward(image, fast)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                allocated.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            if fast:
                changed = int(np.count_nonzero(actual != reference[sample.key]))
                if changed:
                    raise RuntimeError(f'Compact prediction changed {changed} pixels on {sample.key}')
            else:
                reference[sample.key] = actual.copy()
                changed = 0
            rows.append(dict(key=sample.key, shape=list(image.shape[-2:]), milliseconds=times,
                peak_allocated_mib=max(allocated), peak_reserved_mib=max(reserved), changed_pixels=changed))
        results['compact' if fast else 'reference'] = dict(images=rows,
            median_ms=float(np.median([t for row in rows for t in row['milliseconds']])),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in rows),
            peak_reserved_mib=max(row['peak_reserved_mib'] for row in rows))
    save(output, dict(status='complete', dataset=args.dataset, choice=choice, exact_sample_predictions=True,
        images=len(selected), text_offload=text_offload, results=results,
        boundary='Same frozen language, parameters, visual views and dtypes; whole RGB input to CPU prediction. '
                 'Decode/init/text encoding excluded. Two warmups, seven fixed images, seven synchronized repeats.'))
    print(json.dumps(dict(dataset=args.dataset, **{k: {f:v[f] for f in ('median_ms','peak_allocated_mib')} for k,v in results.items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','data-root','suite-root','original-cache','dinov3-repo','checkpoint-dir','upstream-root','output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--mode', choices=('sample','full','interleaved','final'), default='sample')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-shards', type=int, default=1); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--sample-seed', type=int, default=20260923); p.add_argument('--vdd-ontology', default='official')
    main(p.parse_args())
