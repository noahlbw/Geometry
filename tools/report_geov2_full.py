"""Audit full-shard coverage and compute paired dataset-mIoU intervals.

No model, labels or dataset installation is needed after collection.  Dataset
mIoU is recomputed from summed pixel counts, never from averaged image mIoUs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SHARDS = {'vdd': 1, 'potsdam': 1, 'ade150': 4, 'voc20': 2}
COUNTS = {'vdd': 80, 'potsdam': 504, 'ade150': 2000, 'voc20': 1449}


def statistics(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(np.float64)
    return tp, cm.sum(-1) + cm.sum(-2) - tp


def miou(tp, union):
    values = np.divide(tp, union, out=np.full_like(tp, np.nan, dtype=float), where=union > 0)
    return np.nanmean(values, axis=-1) * 100


def paired_bootstrap(arrays, reference, controls, repetitions, seed):
    n = len(arrays[reference])
    rng = np.random.default_rng(seed)
    sufficient = {m: statistics(a) for m, a in arrays.items()}
    draws = {m: [] for m in (reference, *controls)}
    # Shared multinomial image weights preserve pairing and make resampling
    # the aggregate confusion matrix cheap even for 150 classes.
    for start in range(0, repetitions, 100):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(100, repetitions-start)).astype(float)
        for method in draws:
            tp, union = sufficient[method]
            draws[method].extend(miou(weights @ tp, weights @ union).tolist())
    ref_tp, ref_union = sufficient[reference]
    ref_value = float(miou(ref_tp.sum(0), ref_union.sum(0)))
    result = {}
    for method in controls:
        tp, union = sufficient[method]
        delta = np.asarray(draws[reference])-np.asarray(draws[method])
        result[method] = dict(
            difference_pp=ref_value-float(miou(tp.sum(0), union.sum(0))),
            paired_dataset_miou_ci95_pp=np.quantile(delta, [.025, .975]).tolist(),
            bootstrap_images=n, repetitions=repetitions)
    return result


def merge(root, dataset, count):
    rows, archives = [], []
    for shard in range(count):
        folder = root / f'{dataset}_shard{shard}'
        row = json.loads((folder/'results.json').read_text())
        if (row['status'] != 'complete' or row['shard'] != shard or row['shards'] != count
                or row['processed_images'] != row['total_images']
                or row['global_sample_count'] != COUNTS[dataset]):
            raise ValueError('Invalid or incomplete shard: ' + str(folder))
        archive = np.load(folder/'per_image_confusions.npz', allow_pickle=False)
        if archive['sample_keys'].tolist() != row['sample_keys']:
            raise ValueError('Result/array key mismatch: ' + str(folder))
        rows.append(row)
        archives.append(archive)
    keys = [key for row in rows for key in row['sample_keys']]
    if len(keys) != COUNTS[dataset] or len(set(keys)) != len(keys):
        raise ValueError('Missing or duplicated full-split images: ' + dataset)
    methods = rows[0]['methods']
    names = [c['name'] for c in rows[0]['metrics'][methods[0]]['per_class']]
    matrices = {}
    for method in methods:
        parts = [np.asarray(r['metrics'][method]['confusion_matrix'], dtype=np.int64) for r in rows]
        matrices[method] = sum(parts, start=np.zeros_like(parts[0]))
    targets = [cm.sum(1) for cm in matrices.values()]
    if any(not np.array_equal(targets[0], x) for x in targets[1:]):
        raise ValueError('Target support differs across matched rows.')
    arrays = {}
    for method in archives[0].files:
        if method == 'sample_keys':
            continue
        array = np.concatenate([a[method] for a in archives])
        if array.shape[0] != len(keys) or not np.array_equal(array.sum(0), matrices[method]):
            raise ValueError('Per-image sum differs from aggregate: ' + method)
        arrays[method] = array
    metrics = {}
    for method, cm in matrices.items():
        tp, union = statistics(cm)
        metrics[method] = dict(mean_iou_percent=float(miou(tp, union)),
                              class_names=names, confusion_matrix=cm.tolist(),
                              per_class_iou_percent=np.divide(tp*100, union, out=np.full_like(tp, np.nan), where=union>0).tolist())
    return dict(status='complete', dataset=dataset, global_sample_count=len(keys),
                sample_keys=keys, methods=methods, metrics=metrics,
                audit=dict(unique_images=len(set(keys)), all_shards_complete=True,
                           target_support_identical=True, per_image_sums_verified=True)), arrays


def main(args):
    root = Path(args.root)
    all_results = {}
    for ds, shards in SHARDS.items():
        result, arrays = merge(root, ds, shards)
        reference = 'AdaptiveReference' if args.supplement else 'Geov2_AdaptiveGate'
        controls = [m for m in arrays if m != reference]
        result['paired_intervals'] = paired_bootstrap(arrays, reference, controls, args.bootstrap, 20261014)
        (root/(ds+'_audited.json')).write_text(json.dumps(result, indent=2))
        all_results[ds] = result
        print(ds, result['audit'], {m:round(v['mean_iou_percent'],4) for m,v in result['metrics'].items()}, flush=True)
    macro = {m:float(np.mean([all_results[d]['metrics'][m]['mean_iou_percent'] for d in SHARDS]))
             for m in all_results['vdd']['methods']}
    summary = dict(datasets=all_results, equal_dataset_macro_miou=macro,
                   inference_selection_note='Profiles and aliases were already selected during prior development; no validation retuning here. A label-free gate does not make dataset-specific frozen profiles label-free in their development history.')
    (root/'audited_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='research/geov2_full_ablation_20261014')
    p.add_argument('--supplement', action='store_true')
    p.add_argument('--bootstrap', type=int, default=3000)
    main(p.parse_args())
