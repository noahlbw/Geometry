"""Time the actual QuerySlot/DonorBefore StaticReader inference path."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dinotool.compact_deployment import CompactGeometry, compact_observations, CachedWideScorer, offload_text_towers
from dinotool.context_memory_observer import ContextMemoryConfig, ContextMemoryObserver, CompactContextScorer
from dinotool.response_geometry_deployment import ResponseGeometry, EfficientResponseGeometry
from dinotool.efficient_deployment import reconstruction, write_dense, restore_and_finalize
from dinotool.natural_static_search import StaticReader, StaticProfile
from dinotool.query_slot_memory import query_slot_wide_observations
from eval_alias_finalization import load_samples
from eval_kev_alias_search import task_entry, load_models, predict


@torch.inference_mode()
def main(args):
    out = Path(args.output)
    if out.exists():
        raise RuntimeError('Preserve measured output: ' + str(out))
    manifest = json.loads(Path(args.experiment).read_text())
    records = []
    for ds in args.datasets:
        record = manifest['datasets'][ds]
        cfg = SimpleNamespace(**record['loader'], mode='full', device='cuda',
                              num_shards=1, shard_index=0)
        from dinov3.eval.text import tokenizer as tokenizer
        tokenizer.get_tokenizer = lambda *_a, **_k: tokenizer.Tokenizer(
            vocab_path=str(Path(cfg.checkpoint_dir) / 'bpe_simple_vocab_16e6.txt.gz'))
        _, entry, _ = task_entry(cfg)
        validation, _, loader, _, _ = load_samples(cfg, entry)
        chosen = [validation[i] for i in np.linspace(0, len(validation)-1, args.images, dtype=int)]
        profile = StaticProfile(**record['profile'])
        geometry, vip, banks, queries, _, residual = load_models(cfg, entry, (profile.bank,))
        if residual is not None:
            raise RuntimeError('This benchmark requires no residual ontology.')
        text_offload = offload_text_towers(geometry, vip, release=True)
        reader = StaticReader(banks, queries, entry['background_index'])
        compact = CompactGeometry(geometry)
        response = ResponseGeometry(compact)
        efficient = EfficientResponseGeometry(geometry)
        observer = ContextMemoryObserver(vip.backbone, ContextMemoryConfig(
            center=False, capacity=24, temperature=.1, iterations=5))
        wide_scorer, vip_scorer = CompactContextScorer(), CachedWideScorer()
        methods = ('fixed', 'adaptive', 'equal_mean', 'adaptive_efficient', 'vipwide_fixed')

        def inference(image, method):
            opt = method == 'adaptive_efficient'
            if method == 'vipwide_fixed':
                source = compact_observations(image, compact, vip, (profile.strength,),
                                              wide_policy=profile.wide_policy)
                scorer = vip_scorer
                fusion = 'coupled'
            else:
                source = query_slot_wide_observations(
                    image, efficient if opt else response, observer, (profile.strength,),
                    wide_policy=profile.wide_policy, temperature=.1, beta=.5,
                    reconstruction=reconstruction if opt else None)
                scorer = wide_scorer
                fusion = {'fixed':'coupled', 'adaptive_efficient':'adaptive'}.get(method, method)
            probability = reader.probabilities(
                source, profile, {}, wide_scorer=scorer, fusion=fusion,
                dense_writer=write_dense if opt else None,
                output_finalizer=(lambda normalized, size: restore_and_finalize(
                    normalized, size, record['background_bias'],
                    record['background_threshold'], entry['background_index'])) if opt else None)
            return probability if isinstance(probability, np.ndarray) else predict(probability, record, entry['background_index'])

        gc.collect()
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()/1048576
        measurements = {m: [] for m in methods}
        peaks = {m: [] for m in methods}
        comparisons = []
        for sample in chosen:
            image = loader(sample)
            predictions = {}
            for method in methods:
                for _ in range(args.warmups):
                    prediction = inference(image, method)
                torch.cuda.synchronize()
                for _ in range(args.repeats):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    prediction = inference(image, method)
                    torch.cuda.synchronize()
                    measurements[method].append((time.perf_counter()-start)*1000)
                    peaks[method].append(torch.cuda.max_memory_allocated()/1048576)
                predictions[method] = prediction
            mismatch = int(np.count_nonzero(predictions['adaptive'] != predictions['adaptive_efficient']))
            comparisons.append(dict(key=sample.key, pixels=int(predictions['adaptive'].size),
                                    efficient_vs_reference_changed_pixels=mismatch))
            print(json.dumps(dict(dataset=ds, key=sample.key, optimized_changed_pixels=mismatch)), flush=True)
        records.append(dict(
            dataset=ds, images=len(chosen), sample_keys=[s.key for s in chosen],
            profile=record['profile'], text_offload=text_offload,
            live_baseline_allocated_mib=baseline_memory, comparisons=comparisons,
            methods={m: dict(mean_ms=float(np.mean(measurements[m])),
                             median_ms=float(np.median(measurements[m])),
                             p95_ms=float(np.quantile(measurements[m], .95)),
                             peak_allocated_mib=max(peaks[m]), samples_ms=measurements[m])
                     for m in methods}))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
            inference_scope='CPU RGB tensor -> CPU label map; load/encode text and disk I/O excluded; includes resize, both visual encoders, H solve, aliases, writer, softmax, interpolation and transfer',
            capacity=24, beta=.5, query_temperature=.1,
            warmups_per_image_method=args.warmups, repeats_per_image_method=args.repeats,
            records=records), indent=2))
        del geometry, vip, banks, queries, reader, compact, response, efficient, observer, wide_scorer, vip_scorer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--datasets', nargs='+', default=['vdd','potsdam','ade150','voc20'])
    p.add_argument('--images', type=int, default=8)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    main(p.parse_args())
