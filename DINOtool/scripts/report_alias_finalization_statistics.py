"""Paired source-group uncertainty from frozen results; no model inference.

TP and union counts are sufficient for each bootstrap mIoU. Keeping these
instead of C-by-C bootstrap matrices reduces the analysis budget to O(GC).
Filename groups are only acquisition proxies; intervals do not correct for
the historical model/word selection on these domains.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from report_matched_readout_publication import group_key


METHODS = ('Retained', 'Uniform', 'UniformLocal', 'UniformWide', 'UniformMean')
PAIRS = (('Uniform', 'Retained'), ('Uniform', 'UniformMean'),
         ('Uniform', 'UniformLocal'), ('Uniform', 'UniformWide'),
         ('Retained', 'UniformMean'), ('Retained', 'UniformLocal'),
         ('Retained', 'UniformWide'))


def score(tp, union):
    values = np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union > 0)
    return np.nanmean(values, axis=-1) * 100


def sufficient(raw):
    tp = np.diagonal(raw, axis1=-2, axis2=-1).copy()
    return tp, raw.sum(-1) + raw.sum(-2) - tp


def analyze(root, dataset, entry, replicates=2000, seed=20261009):
    folder = root / 'full' / dataset
    path = folder / 'verified_merged.json'
    if not path.exists():
        path = folder / 'merged.json'
    result = json.loads(path.read_text())
    if (result['status'] != 'complete' or not result['coverage_verified']
            or not result.get('paired_scored_targets_equal')
            or not result.get('scalar_agreement_verified')
            or result['processed_images'] != entry['total_images']):
        raise ValueError('Verified complete protocol required: ' + dataset)
    expected = result['signature']['sample_keys']
    if len(expected) != entry['total_images'] or len(set(expected)) != len(expected):
        raise ValueError('Incomplete/duplicate global keys.')
    pieces, keys = {}, []
    for shard in range(entry['shards']):
        with np.load(folder / f's{shard}/per_image_confusions.npz', allow_pickle=False) as arrays:
            shard_keys = arrays['sample_keys'].tolist()
            keys.extend(shard_keys)
            targets = {}
            for protocol in result['metrics']:
                for method in METHODS:
                    raw = arrays[protocol + '__' + method]
                    if (raw.ndim != 3 or raw.shape[0] != len(shard_keys)
                            or raw.shape[1] != raw.shape[2]
                            or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0)):
                        raise ValueError('Invalid per-image confusion.')
                    current = raw.sum(-1)
                    if protocol in targets and not np.array_equal(targets[protocol], current):
                        raise ValueError('Different per-image target counts.')
                    targets[protocol] = current
                    pieces.setdefault((protocol, method), []).append(sufficient(raw))
                    del raw
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Shards do not uniquely cover global keys.')
    lookup = {k: i for i, k in enumerate(keys)}
    order = np.asarray([lookup[k] for k in expected])
    names = [group_key(dataset, k) for k in expected]
    unique = sorted(set(names))
    group_lookup = {name: i for i, name in enumerate(unique)}
    ids = np.asarray([group_lookup[name] for name in names])
    rng = np.random.default_rng(seed)
    # Generate the same paired group weights once for all methods/protocols.
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=replicates)
    fields, differences = {}, {}
    for protocol, metrics in result['metrics'].items():
        values, points = {}, {}
        for method in METHODS:
            tp = np.concatenate([p[0] for p in pieces[protocol, method]])[order]
            union = np.concatenate([p[1] for p in pieces[protocol, method]])[order]
            full_tp, full_union = sufficient(np.asarray(metrics[method]['confusion_matrix'], np.int64))
            if not np.array_equal(tp.sum(0), full_tp) or not np.array_equal(union.sum(0), full_union):
                raise ValueError('Sufficient counts do not reconstruct aggregate.')
            points[method] = float(score(tp.sum(0), union.sum(0)))
            if abs(points[method] - metrics[method]['mean_iou_percent']) > .000051:
                raise ValueError('Recomputed mIoU differs.')
            grouped_tp = np.zeros((len(unique), tp.shape[-1]), np.int64)
            grouped_union = np.zeros_like(grouped_tp)
            np.add.at(grouped_tp, ids, tp)
            np.add.at(grouped_union, ids, union)
            # Float64 matrix products are exact for these integer-count ranges.
            values[method] = score(draws.astype(np.float64) @ grouped_tp,
                                   draws.astype(np.float64) @ grouped_union)
        comparisons = {}
        for a, b in PAIRS:
            key = a + '_minus_' + b
            samples = values[a] - values[b]
            differences[protocol, key] = samples
            comparisons[key] = dict(delta_pp=points[a] - points[b],
                ci95_pp=np.percentile(samples, [2.5, 97.5]).tolist())
        fields[protocol] = dict(images=len(keys), groups=len(unique), comparisons=comparisons)
    return fields, differences


def main(root, output, datasets=None, replicates=2000):
    root = root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    selected = datasets or protocol['order']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'paired_statistics.json').exists():
        raise RuntimeError('Preserve existing statistics; use a new explicit output directory.')
    stats, draws = {}, {}
    for i, dataset in enumerate(protocol['order']):
        if dataset not in selected:
            continue
        stats[dataset], draws[dataset] = analyze(root, dataset, protocol['datasets'][dataset],
            replicates, 20261009 + i)
    panels = dict(primary=protocol['decision']['primary_panel'],
                  eight_rs=[d for d in protocol['order'] if protocol['datasets'][d]['family'] == 'remote_sensing'])
    means = {}
    for name, panel in panels.items():
        if not all(d in stats for d in panel):
            continue
        means[name] = {}
        for a, b in PAIRS:
            key = a + '_minus_' + b
            p = lambda d: 'D' if d == 'loveda' else d
            samples = np.mean([draws[d][p(d), key] for d in panel], axis=0)
            point = float(np.mean([stats[d][p(d)]['comparisons'][key]['delta_pp'] for d in panel]))
            means[name][key] = dict(delta_pp=point, ci95_pp=np.percentile(samples, [2.5, 97.5]).tolist())
    note = ('Paired source-group bootstrap; filename groups are proxies, not verified independent '
            'acquisitions. VOC20/21 and PC59/60 reuse sources. Conditional on developed domains; '
            'not adjusted for method selection, not untouched validation. All classes with union>0 '
            'participate in each bootstrap draw. Intervals do not replace the frozen model-decision gate.')
    record = dict(replicates=replicates, seed=20261009, domains=stats, panel_means=means,
                  inference_rerun=False, sufficient_count_reconstruction_verified=True, note=note)
    (output / 'paired_statistics.json').write_text(json.dumps(record, indent=2) + '\n')
    lines = ['# Frozen finalization: paired uncertainty', '', note, '',
             f'{replicates} paired bootstrap repetitions; no image/model inference.', '',
             '| Protocol | Groups | Comparison | Delta pp | 95% paired interval |',
             '| --- | ---: | --- | ---: | --- |']
    for d, protocols in stats.items():
        for p, row in protocols.items():
            for key, val in row['comparisons'].items():
                lo, hi = val['ci95_pp']
                lines.append(f'| {d}/{p} | {row["groups"]} | {key} | {val["delta_pp"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', 'Panel means count LoveDA D once; resample groups independently within domains.', '']
    for name, pairs in means.items():
        for key, val in pairs.items():
            lo, hi = val['ci95_pp']
            lines.append(f'{name}/{key}: {val["delta_pp"]:+.4f}pp, [{lo:+.4f}, {hi:+.4f}].')
    (output / 'PAIRED_UNCERTAINTY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(protocols=len(stats), panel_means=means)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error('At least two replicates required.')
    main(args.root, args.output_dir, args.datasets, args.replicates)
