"""Audit coverage and paired effects for the anchored wide-readout experiment.

No measurements are inferred for missing shards. Screen results remain explicitly
separate from full-validation results, and all ablations (including negatives)
are retained.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / 'research/geov2_anchored_wide_20261015'
CORE = ('Coordinate', 'Legacy', 'Anchored')
LABELS = dict(vdd='VDD', potsdam='Potsdam', udd5='UDD5', oem='OEM',
              vaihingen='Vaihingen', ade150='ADE150', voc21='VOC21')
SEED = 20261015


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        # Do not leave an older completed table beside a newly incomplete audit.
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scores(tp, predicted, target):
    union = predicted + target - tp
    iou = np.divide(tp, union, out=np.full(tp.shape, np.nan, dtype=np.float64), where=union > 0)
    return np.nanmean(iou, axis=-1) * 100


def grouping(dataset, keys):
    if dataset in ('potsdam', 'vaihingen'):
        return [re.sub(r'_y\d+_x\d+\.[^.]+$', '', key) for key in keys], 'original aerial scene'
    if dataset == 'oem':
        return [key.split('/')[0] for key in keys], 'geographic region'
    return list(keys), 'image'


def bootstrap(tp, predicted, target, repeats):
    n, m, c = tp.shape
    if n == 0:
        raise ValueError('Cannot bootstrap zero sampling units.')
    stats = np.stack((tp, predicted + target[:, None] - tp), axis=-1).reshape(n, -1).astype(np.float64)
    rng = np.random.default_rng(SEED)
    result = np.empty((repeats, m))
    for start in range(0, repeats, 50):
        count = min(50, repeats - start)
        draws = rng.integers(0, n, size=(count, n))
        weights = np.stack([np.bincount(draw, minlength=n) for draw in draws]).astype(np.float64)
        total = (weights @ stats).reshape(count, m, c, 2)
        iou = np.divide(total[..., 0], total[..., 1], out=np.full((count, m, c), np.nan), where=total[..., 1] > 0)
        result[start:start + count] = np.nanmean(iou, axis=-1) * 100
    return result


def stage_keys(protocol, expected, dataset, stage):
    record = protocol['datasets'][dataset]
    if stage == 'full':
        keys = expected[dataset]
    else:
        keys = record.get(stage + '_keys')
        if keys is None:
            keys = protocol.get(stage + '_keys', {}).get(dataset)
    if isinstance(keys, dict):
        keys = keys.get('sample_keys', keys.get('keys'))
    if keys is None:
        return None
    if len(keys) != len(set(keys)):
        raise RuntimeError(f'Duplicate expected {stage} keys: {dataset}')
    if not set(keys).issubset(expected[dataset]):
        raise RuntimeError(f'Unexpected {stage} sample keys: {dataset}')
    return list(keys)


def compare(index, reference, cells, values, boot, image_boot, unit):
    delta = boot[:, reference] - boot[:, index]
    return dict(**cells[index], mean_iou_percent=float(values[index]),
                anchored_minus_arm_pp=float(values[reference] - values[index]),
                paired_95ci_pp=np.quantile(delta, [.025, .975]).tolist(),
                image_paired_95ci_pp=np.quantile(image_boot[:, reference] - image_boot[:, index], [.025, .975]).tolist(),
                interval_unit=unit)


def cross_validated_gain(cells, arrays, groups):
    """Supplementary scalar selection, retaining scene/region groups in folds."""
    unique = sorted(set(groups))
    if len(unique) < 5:
        return dict(status='insufficient_groups', unique_groups=len(unique))
    rng = np.random.default_rng(SEED)
    assignment = {unique[index]: fold % 5 for fold, index in enumerate(rng.permutation(len(unique)))}
    folds = np.asarray([assignment[group] for group in groups])
    rows = []
    for method in (*CORE, 'UniformBlend'):
        candidates = [i for i, cell in enumerate(cells) if cell['method'] == method]
        if not candidates:
            continue
        pooled_tp = np.zeros(arrays['tp'].shape[-1], np.int64)
        pooled_predicted = pooled_tp.copy()
        selected = []
        for fold in range(5):
            train, test = folds != fold, folds == fold
            values = scores(arrays['tp'][train][:, candidates].sum(0),
                            arrays['predicted'][train][:, candidates].sum(0), arrays['target'][train].sum(0))
            index = candidates[int(np.nanargmax(values))]
            pooled_tp += arrays['tp'][test, index].sum(0)
            pooled_predicted += arrays['predicted'][test, index].sum(0)
            selected.append(dict(fold=fold, train_images=int(train.sum()), test_images=int(test.sum()),
                                 key=cells[index]['key'], g0=cells[index].get('g0'), alpha=cells[index].get('alpha')))
        rows.append(dict(method=method, mean_iou_percent=float(scores(pooled_tp, pooled_predicted, arrays['target'].sum(0))),
                         fold_choices=selected))
    return dict(status='complete', folds=5, seed=SEED, unique_groups=len(unique), rows=rows,
                limitation='Only new scalar selection is separated by folds. Historical aliases/calibration and validation inspection are not made independent by this analysis.')


def audit_dataset(root, protocol, expected, dataset, stage, repeats):
    record = protocol['datasets'][dataset]
    cells = record['cells']
    cell_keys = [cell['key'] for cell in cells]
    if len(cell_keys) != len(set(cell_keys)):
        raise RuntimeError('Duplicate cell keys: ' + dataset)
    wanted = stage_keys(protocol, expected, dataset, stage)
    folders = sorted((root / stage / dataset).glob('s*'))
    if not folders:
        return dict(status='missing', expected_images=len(wanted) if wanted is not None else None)
    first, arrays, keys, missing = None, [], [], []
    for folder in folders:
        result_path, statistics_path = folder / 'results.json', folder / 'statistics.npz'
        if not result_path.exists() or not statistics_path.exists():
            missing.append(str(folder.relative_to(root)))
            continue
        row = read(result_path)
        if row.get('status') != 'complete':
            missing.append(str(folder.relative_to(root)))
            continue
        with np.load(statistics_path, allow_pickle=False) as data:
            a = {key: data[key].copy() for key in ('tp', 'predicted', 'target')}
            local_keys = data['sample_keys'].tolist()
            if 'methods' in data and data['methods'].tolist() != cell_keys:
                raise RuntimeError('NPZ cell order differs: ' + str(folder))
        if row.get('cells', cells) != cells or row['sample_keys'] != local_keys or row['processed_images'] != len(local_keys):
            raise RuntimeError('Result sample/cell metadata differs: ' + str(folder))
        if not set(local_keys).issubset(expected[dataset]):
            raise RuntimeError('Unknown sample keys: ' + str(folder))
        class_names = row['class_names']
        shape = (len(local_keys), len(cells), len(class_names))
        if a['tp'].shape != shape or a['predicted'].shape != shape or a['target'].shape != (shape[0], shape[2]):
            raise RuntimeError('Invalid statistics shape: ' + str(folder))
        for key, value in a.items():
            if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                raise RuntimeError('Invalid count array: ' + str(folder) + ' ' + key)
        if (np.any(a['tp'] > a['predicted']) or np.any(a['tp'] > a['target'][:, None])
                or not np.all(a['predicted'].sum(-1) == a['target'].sum(-1)[:, None])):
            raise RuntimeError('Invalid per-image support: ' + str(folder))
        if first is not None:
            for field in ('class_names', 'profile', 'config', 'cache_identity', 'alias_counts', 'cells', 'num_shards'):
                if field in first and field in row and first[field] != row[field]:
                    raise RuntimeError('Shard metadata differs: ' + field)
        first = first or row
        for i, key in enumerate(cell_keys):
            metric = row['metrics'][key]
            cm = np.asarray(metric['confusion_matrix'], dtype=np.int64)
            if (cm.shape != (shape[2], shape[2]) or not np.array_equal(a['tp'][:, i].sum(0), cm.diagonal())
                    or not np.array_equal(a['predicted'][:, i].sum(0), cm.sum(0))
                    or not np.array_equal(a['target'].sum(0), cm.sum(1))):
                raise RuntimeError('Statistics differ from confusion matrix: ' + str(folder) + ' ' + key)
            recomputed = float(scores(cm.diagonal(), cm.sum(0), cm.sum(1)))
            if not np.isclose(recomputed, metric['mean_iou_percent'], rtol=0, atol=1e-7):
                raise RuntimeError('Recorded shard mIoU differs: ' + str(folder) + ' ' + key)
        arrays.append(a)
        keys += local_keys
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate shard coverage: ' + dataset + ' ' + stage)
    if wanted is None:
        # Smoke's tiny set may be absent from a legacy protocol. Audit its
        # recorded subset, but never label that coverage as predeclared.
        if stage != 'smoke':
            raise RuntimeError('Protocol is missing predeclared ' + stage + '_keys: ' + dataset)
        wanted = sorted(keys)
        predeclared = False
    else:
        predeclared = True
    if set(keys) - set(wanted):
        raise RuntimeError('Shard keys outside declared stage: ' + dataset + ' ' + stage)
    if missing or set(keys) != set(wanted):
        return dict(status='incomplete', processed_images=len(keys), expected_images=len(wanted),
                    missing_shards=missing, missing_sample_keys=sorted(set(wanted) - set(keys)))
    if stage == 'full' and len(wanted) != record['total_images']:
        raise RuntimeError('Full expected count differs from protocol: ' + dataset)
    lookup = {key: i for i, key in enumerate(keys)}
    order = [lookup[key] for key in wanted]
    a = {key: np.concatenate([piece[key] for piece in arrays])[order] for key in ('tp', 'predicted', 'target')}
    values = scores(a['tp'].sum(0), a['predicted'].sum(0), a['target'].sum(0))
    groups, unit = grouping(dataset, wanted)
    image_boot = bootstrap(a['tp'], a['predicted'], a['target'], repeats)
    boot = image_boot
    if unit != 'image':
        members = [np.asarray(groups) == group for group in sorted(set(groups))]
        boot = bootstrap(np.stack([a['tp'][member].sum(0) for member in members]),
                         np.stack([a['predicted'][member].sum(0) for member in members]),
                         np.stack([a['target'][member].sum(0) for member in members]), repeats)
    ref_candidates = [i for i, cell in enumerate(cells) if cell['method'] == 'Anchored' and cell.get('g0') == record['reference_g0']]
    if len(ref_candidates) != 1:
        raise RuntimeError('No unique Anchored reference: ' + dataset)
    ref = ref_candidates[0]
    methods = list(dict.fromkeys(cell['method'] for cell in cells))
    matched = []
    for method in methods:
        candidates = [i for i, cell in enumerate(cells) if cell['method'] == method
                      and (cell.get('g0') == record['reference_g0'] or cell.get('g0') is None)]
        if method == 'UniformBlend':
            candidates = [max((i for i, cell in enumerate(cells) if cell['method'] == method), key=lambda i: values[i])]
        if len(candidates) != 1:
            raise RuntimeError('No unique matched arm: ' + dataset + ' ' + method)
        matched.append(compare(candidates[0], ref, cells, values, boot, image_boot, unit))
    best = {method: max((i for i, cell in enumerate(cells) if cell['method'] == method), key=lambda i: values[i])
            for method in CORE if method in methods}
    core_gains = [[cell.get('g0') for cell in cells if cell['method'] == method] for method in CORE if method in methods]
    equal_budget = len(core_gains) == len(CORE) and all(gains == core_gains[0] for gains in core_gains)
    retuned = [compare(best[method], best['Anchored'], cells, values, boot, image_boot, unit) for method in CORE if method in best]
    if 'UniformBlend' in methods:
        uniform = max((i for i, cell in enumerate(cells) if cell['method'] == 'UniformBlend'), key=lambda i: values[i])
        retuned.append(compare(uniform, best['Anchored'], cells, values, boot, image_boot, unit))
    output = root / stage / dataset
    np.savez_compressed(output / 'per_image_statistics.npz', sample_keys=np.asarray(wanted), methods=np.asarray(cell_keys), **a)
    record_out = dict(status='complete', images=len(wanted), reference_g0=record['reference_g0'],
                      class_names=first['class_names'], coverage_verified=True, predeclared_coverage=predeclared,
                      statistics_match_confusion=True, interval_unit=unit, unique_groups=len(set(groups)),
                      matched_rows=matched, retuned_rows=retuned, equal_gain_search_budget_verified=equal_budget,
                      all_rows=[dict(dataset=dataset, **cell, mean_iou_percent=float(value)) for cell, value in zip(cells, values)],
                      cross_validated_gain=cross_validated_gain(cells, a, groups))
    for field in ('profile', 'config', 'cache_identity', 'alias_counts'):
        if field in first:
            record_out[field] = first[field]
    delta = {row['method']: boot[:, ref] - boot[:, cell_keys.index(row['key'])] for row in matched if row['method'] != 'Anchored'}
    return record_out, delta


def collect_stage(root, protocol, expected, stage, repeats):
    result = dict(stage=stage, status='missing', datasets={}, bootstrap_replicates=repeats, seed=SEED)
    deltas = {}
    for dataset in protocol['datasets']:
        row = audit_dataset(root, protocol, expected, dataset, stage, repeats)
        if isinstance(row, tuple):
            row, delta = row
            deltas[dataset] = delta
        result['datasets'][dataset] = row
    completed = {dataset: row for dataset, row in result['datasets'].items() if row['status'] == 'complete'}
    if len(completed) == len(protocol['datasets']):
        result['status'] = 'complete'
    elif completed or any(row['status'] != 'missing' for row in result['datasets'].values()):
        result['status'] = 'incomplete'
    result['completed_datasets'] = list(completed)
    result['expected_datasets'] = list(protocol['datasets'])
    result['equal_domain_effects'] = []
    if completed:
        common = set.intersection(*(set(deltas[dataset]) for dataset in completed))
        rng = np.random.default_rng(15102026)
        for method in sorted(common):
            observed = [next(row['anchored_minus_arm_pp'] for row in r['matched_rows'] if row['method'] == method)
                        for r in completed.values()]
            draws = [deltas[dataset][method][rng.permutation(repeats)] for dataset in completed]
            result['equal_domain_effects'].append(dict(method=method, datasets=list(completed),
                anchored_minus_arm_pp=float(np.mean(observed)), paired_95ci_pp=np.quantile(np.mean(draws, axis=0), [.025, .975]).tolist(),
                positive_point_datasets=sum(value > 0 for value in observed), negative_point_datasets=sum(value < 0 for value in observed)))
    rows = [row for record in completed.values() for row in record['all_rows']]
    (root / stage).mkdir(parents=True, exist_ok=True)
    csv_write(root / stage / 'ALL_RESULTS.csv', rows)
    dump(root / stage / 'SUMMARY.json', result)
    report_stage(root / stage / 'REPORT_ZH.md', result, protocol)
    return result


def collect_costs(root, protocol):
    rows, missing = [], []
    for dataset, record in protocol['datasets'].items():
        wanted = record.get('cost_methods', protocol.get('cost_methods', list(CORE)))
        present = set()
        for path in sorted((root / 'cost' / dataset).glob('*/results.json')):
            row = read(path)
            if row.get('status', 'complete') != 'complete':
                continue
            if row.get('target_masks_loaded', False):
                raise RuntimeError('Cost evaluation loaded target masks: ' + str(path))
            if row.get('dataset', dataset) != dataset:
                raise RuntimeError('Cost dataset differs: ' + str(path))
            method = row.get('method', row.get('cells', [{}])[0].get('method', path.parent.name))
            if '__' in method:
                method = method.split('__')[0]
            cell = row.get('cells', [None])[0]
            if cell is not None and cell not in record['cells']:
                raise RuntimeError('Cost cell not in frozen protocol: ' + str(path))
            if cell is not None and method in CORE and cell.get('g0') != record['reference_g0']:
                raise RuntimeError('Core cost must use reference gain: ' + str(path))
            values = {key: float(row[key]) for key in ('median_ms', 'peak_allocated_mib', 'peak_reserved_mib')}
            if not all(np.isfinite(value) and value > 0 for value in values.values()):
                raise RuntimeError('Invalid cost values: ' + str(path))
            if 'images' in row and record.get('cost_keys'):
                if [image['key'] for image in row['images']] != record['cost_keys']:
                    raise RuntimeError('Cost sample set differs: ' + str(path))
                measurements = [time for image in row['images'] for time in image['milliseconds']]
                if len(row['images']) != 7 or any(len(image['milliseconds']) != 7 for image in row['images']):
                    raise RuntimeError('Incomplete cost repetitions: ' + str(path))
                if not np.isclose(values['median_ms'], np.median(measurements)):
                    raise RuntimeError('Cost median differs from samples: ' + str(path))
            present.add(method)
            rows.append(dict(dataset=dataset, method=method, g0=cell.get('g0') if cell else None,
                             **values, source=str(path.relative_to(root)), boundary=row.get('boundary', 'See frozen protocol.')))
        missing += [dict(dataset=dataset, method=method) for method in wanted if method not in present]
    csv_write(root / 'COST_RESULTS.csv', rows)
    return dict(status='complete' if not missing else 'incomplete' if rows else 'missing', rows=rows, missing=missing)


def source_signature(root, protocol, stage, repeats):
    """Use source file metadata to avoid repeating unchanged statistical work."""
    paths = [root / 'protocol.json', root / 'expected_keys.json']
    for dataset in protocol['datasets']:
        for folder in sorted((root / stage / dataset).glob('s*')):
            paths += [folder / name for name in ('results.json', 'statistics.npz')]
    return dict(repeats=repeats, files=[dict(path=str(path.relative_to(root)),
        size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) for path in paths if path.exists()])


def promotion_decision(summary, protocol):
    """Apply the predeclared checks; missing evidence remains pending."""
    full = summary['stages'].get('full', {})
    budgets = protocol.get('promotion', {})
    checks = []
    complete = full.get('status') == 'complete'
    macros = {row['method']: row for row in full.get('equal_domain_effects', [])}
    for method in ('Coordinate', 'Legacy'):
        value = macros.get(method) if complete else None
        low = value['paired_95ci_pp'][0] if value else None
        checks.append(dict(check='macro_vs_' + method, status='pending' if low is None else 'passed' if low > 0 else 'failed',
                           observed=low, criterion='grouped paired macro 95% lower bound > 0 pp'))
    margin = budgets.get('voc21_noninferiority_margin_pp', -.2)
    voc = full.get('datasets', {}).get('voc21', {})
    comparison = next((row for row in voc.get('matched_rows', []) if row['method'] == 'Legacy'), None) if complete else None
    low = comparison['paired_95ci_pp'][0] if comparison else None
    checks.append(dict(check='VOC21_vs_Legacy_noninferiority', status='pending' if low is None else 'passed' if low > margin else 'failed',
                       observed=low, criterion=f'paired 95% lower bound > {margin:g} pp'))
    for dataset in ('potsdam', 'vaihingen'):
        data = full.get('datasets', {}).get(dataset, {})
        comparison = next((row for row in data.get('matched_rows', []) if row['method'] == 'Coordinate'), None) if complete else None
        gain = comparison['anchored_minus_arm_pp'] if comparison else None
        checks.append(dict(check=dataset + '_vs_Coordinate', status='pending' if gain is None else 'passed' if gain >= 0 else 'failed',
                           observed=gain, criterion='reference-gain point improvement >= 0 pp'))
    costs = {(row['dataset'], row['method']): row for row in summary['costs']['rows']}
    ratios, increments = {}, {}
    for dataset in protocol['datasets']:
        anchored, legacy = costs.get((dataset, 'Anchored')), costs.get((dataset, 'Legacy'))
        if anchored is not None and legacy is not None:
            ratios[dataset] = anchored['median_ms'] / legacy['median_ms']
            increments[dataset] = anchored['peak_allocated_mib'] - legacy['peak_allocated_mib']
    all_costs = len(ratios) == len(protocol['datasets'])
    ratio_max = budgets.get('latency_ratio_max', 1.05)
    increment_max = budgets.get('peak_allocated_increment_mib_max', 50)
    for name, values, limit, criterion in (
            ('latency_vs_Legacy', ratios, ratio_max, f'all datasets: Anchored / Legacy median latency <= {ratio_max:g}'),
            ('allocated_memory_vs_Legacy', increments, increment_max, f'all datasets: Anchored - Legacy peak allocated <= {increment_max:g} MiB')):
        check = 'pending' if not all_costs else 'passed' if max(values.values()) <= limit else 'failed'
        checks.append(dict(check=name, status=check, observed=max(values.values()) if values else None,
                           per_dataset=values, criterion=criterion))
    decision = 'not_promoted' if any(row['status'] == 'failed' for row in checks) else 'pending' if any(row['status'] == 'pending' for row in checks) else 'promote'
    return dict(decision=decision, checks=checks, cost_baseline='Legacy at frozen reference g0',
                interpretation='These development criteria select whether to replace the incumbent. They do not establish CVPR acceptance, independent-test generalization, or novelty against untested methods.')


def global_gain_analysis(full, protocol):
    """Compare one shared scalar across domains using already recorded cells."""
    if full.get('status') != 'complete':
        return dict(status='pending', rows=[], best_shared_gain=[])
    gains = protocol.get('gains', [.25, .5, .75, 1.])
    datasets = list(protocol['datasets'])
    rows = []
    for gain in gains:
        for method in CORE:
            values = {}
            for dataset in datasets:
                candidates = [row for row in full['datasets'][dataset]['all_rows']
                              if row['method'] == method and row.get('g0') == gain]
                if len(candidates) != 1:
                    raise RuntimeError(f'No unique global-gain cell: {dataset} {method} g0={gain}')
                values[dataset] = candidates[0]['mean_iou_percent']
            rows.append(dict(method=method, g0=gain, equal_domain_mean_iou_percent=float(np.mean(list(values.values()))),
                             per_dataset=values))
    best = [max((row for row in rows if row['method'] == method),
                key=lambda row: row['equal_domain_mean_iou_percent']) for method in CORE]
    return dict(status='complete', datasets=datasets, gains=gains, rows=rows, best_shared_gain=best,
                selection='One shared g0 for all declared domains; chosen using development labels over the same four frozen gains. Equal weight per domain, not per image. This analysis does not replace frozen-reference promotion criteria or constitute untouched-test evidence.')


def scientific_plots(root, full, costs):
    """Render only measured complete-validation data with standard matplotlib."""
    if full.get('status') != 'complete':
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = list(full['datasets'])
    records = full['datasets']
    color = dict(Coordinate='#238b45', Legacy='#e6550d', Anchored='#2171b5')
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
        for axis, method in zip(axes, ('Coordinate', 'Legacy')):
            for i, dataset in enumerate(datasets):
                row = next(row for row in records[dataset]['matched_rows'] if row['method'] == method)
                low, high = row['paired_95ci_pp']
                value = row['anchored_minus_arm_pp']
                shade = '#2166ac' if low > 0 else '#b2182b' if high < 0 else '#666666'
                axis.plot([low, high], [i, i], color=shade, linewidth=2)
                axis.plot(value, i, 'o', color=shade, markersize=6)
            macro = next(row for row in full['equal_domain_effects'] if row['method'] == method)
            low, high = macro['paired_95ci_pp']
            axis.plot([low, high], [len(datasets), len(datasets)], color='black', linewidth=2.5)
            axis.plot(macro['anchored_minus_arm_pp'], len(datasets), 'D', color='black', markersize=6)
            axis.axvline(0, color='black', linewidth=.8, linestyle='--')
            axis.set_title('Anchored minus ' + method)
            axis.set_xlabel('mIoU difference (percentage points)')
            axis.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(len(datasets) + 1), [LABELS.get(dataset, dataset) for dataset in datasets] + ['Equal-domain mean'])
        axes[0].invert_yaxis()
        fig.suptitle('Full validation: paired 95% intervals at frozen reference gains')
        fig.text(.5, .015, 'Scene groups: Potsdam/Vaihingen; region groups: OEM; images elsewhere. Conditional on frozen parameters.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .94))
        for suffix in ('png', 'pdf'):
            fig.savefig(root / ('paired_readout_intervals.' + suffix), dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)
        lookup = {(row['dataset'], row['method']): row for row in costs['rows']}
        for column, (axis, field, title, label) in enumerate(zip(axes,
                ('mean_iou_percent', 'median_ms', 'peak_allocated_mib'),
                ('Segmentation accuracy', 'Median latency', 'Peak allocated GPU memory'),
                ('mIoU (%)', 'Milliseconds', 'MiB'))):
            any_cost = False
            for i, dataset in enumerate(datasets):
                values = []
                for offset, method in zip((-.18, 0, .18), CORE):
                    source = next(row for row in records[dataset]['matched_rows'] if row['method'] == method) if column == 0 else lookup.get((dataset, method))
                    if source is None:
                        continue
                    value = source[field]
                    values.append(value)
                    axis.plot(value, i + offset, 'o', color=color[method], markersize=5, label=method if i == 0 else None)
                    any_cost = True
                if len(values) == 3:
                    axis.plot([min(values), max(values)], [i, i], color='#aaaaaa', linewidth=.7, zorder=0)
            if not any_cost:
                axis.text(.5, .5, 'Cost measurements pending', ha='center', va='center', transform=axis.transAxes, color='#666666')
            axis.set_title(title)
            axis.set_xlabel(label)
            axis.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(len(datasets)), [LABELS.get(dataset, dataset) for dataset in datasets])
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .95), frameon=False)
        fig.suptitle('Matched 24-slot readouts: complete validation and independent costs', y=.99)
        fig.text(.5, .012, 'Accuracy: frozen reference g0. Cost: fixed 7 images, 49 synchronized timings per method; decode/init/offline text excluded.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .88))
        for suffix in ('png', 'pdf'):
            fig.savefig(root / ('core_accuracy_cost.' + suffix), dpi=180)
        plt.close(fig)
    return ['paired_readout_intervals.png', 'paired_readout_intervals.pdf', 'core_accuracy_cost.png', 'core_accuracy_cost.pdf']


def fmt(row):
    low, high = row['paired_95ci_pp']
    return f"{row['anchored_minus_arm_pp']:+.4f} [{low:+.4f}, {high:+.4f}]"


def report_stage(path, summary, protocol):
    complete = {dataset: row for dataset, row in summary['datasets'].items() if row['status'] == 'complete'}
    title = dict(smoke='冒烟检查', screen='预声明子集筛查', full='完整验证集实验')[summary['stage']]
    lines = [f'# 空间锚定宽视野：{title}', '', f"状态：**{summary['status']}**。已完成{len(complete)}/{len(summary['datasets'])}个数据集。", '',
             '局部DonorBefore、Alias、类别计分和背景校准沿用冻结配置；本轮只改变宽视野读取。固定24槽、聚类5次、温度0.1、残差系数0.5。Coordinate/Legacy/Anchored使用相同步长搜索预算。', '',
             '这里按预声明ID核对唯一覆盖，逐图TP/predicted/target与分片混淆矩阵一致；mIoU由累计计数重新计算，未平均分片mIoU。缺失或未完成的数据集不填入精度数字。', '']
    if summary['stage'] != 'full':
        lines += ['**本页不是完整验证集结果，不能替代全量精度结论。**', '']
    if complete:
        methods = list(dict.fromkeys(row['method'] for r in complete.values() for row in r['matched_rows']))
        lines += ['## 相同参考g0', '', '| 配置 | ' + ' | '.join(LABELS.get(d, d) for d in complete) + ' |',
                  '| --- | ' + ' | '.join('---:' for _ in complete) + ' |',
                  '| g0 | ' + ' | '.join(format(r['reference_g0'], 'g') for r in complete.values()) + ' |']
        for method in methods:
            values = [next((f"{row['mean_iou_percent']:.4f}" for row in r['matched_rows'] if row['method'] == method), '—') for r in complete.values()]
            lines.append('| ' + ('最佳普通加权融合' if method == 'UniformBlend' else method) + ' | ' + ' | '.join(values) + ' |')
        lines += ['', '普通融合若提供多个alpha，以上一行显示其独立最优alpha；不是只比较等权融合。', '',
                  '## 配对净收益', '', '各项为Anchored−对照，单位pp。区间覆盖0时不称为稳定提升。Potsdam/Vaihingen按原航片、OEM按地理区域分组，其余按图像重采样。', '',
                  '| 数据集 | 对照 | Anchored−对照及95%区间 | 重采样单元 |', '| --- | --- | ---: | --- |']
        for dataset, record in complete.items():
            for row in record['matched_rows']:
                if row['method'] != 'Anchored':
                    lines.append(f"| {LABELS.get(dataset, dataset)} | {row['method']} | {fmt(row)} | {record['interval_unit']} ({record['unique_groups']}) |")
        lines += ['', '## 相同预算各自重选g0', '', '| 数据集 | 方法 | 最优g0或alpha | mIoU | Anchored最佳−对照最佳及95%区间 |', '| --- | --- | ---: | ---: | ---: |']
        for dataset, record in complete.items():
            for row in record['retuned_rows']:
                parameter = row.get('alpha') if row['method'] == 'UniformBlend' else row.get('g0')
                lines.append(f"| {LABELS.get(dataset, dataset)} | {row['method']} | {parameter} | {row['mean_iou_percent']:.4f} | {fmt(row)} |")
        lines += ['', '## 已完成域等权平均', '', '| 对照 | 平均净收益pp | 95%区间 | 正/负点估计域数 |', '| --- | ---: | --- | ---: |']
        for row in summary['equal_domain_effects']:
            low, high = row['paired_95ci_pp']
            lines.append(f"| {row['method']} | {row['anchored_minus_arm_pp']:+.4f} | [{low:+.4f}, {high:+.4f}] | {row['positive_point_datasets']}/{row['negative_point_datasets']} |")
        lines += ['', '上述平均只覆盖已完成域；若阶段未完成，不能称全体数据集平均。', '']
    lines += ['## 解释与边界', '',
              '- Coordinate：保留24槽语义头，使用坐标读出；仍有局部和宽视野两路以及结构写回。',
              '- Legacy：已有局部raw-query槽读取及绝对特征混合。',
              '- Anchored：空间槽归属先验＋相对同位置宽视野raw的证据更新＋槽混合残差。',
              '- NoPrior、NoReference、AbsoluteMix分别检查先验、相对参照和残差减项；Shuffled破坏query与输出位置对应。打乱下降不自动证明相对Coordinate有收益。',
              '- 参考g0与独立最优g0均保留。最优步长用开发标签选取；推理不读取标签。配对bootstrap条件于选定参数，不含完整的参数选择不确定性或多重比较校正。',
              '- 历史Alias、背景校准和验证集查看仍有开发偏差。五折标量选择结果保存在SUMMARY中，仅隔离本轮标量选择。',
              '- 所有负结果保留；本轮对照不能单独证明新方法优于LPOSS等未测方法。', '',
              '完整配置见[ALL_RESULTS.csv](ALL_RESULTS.csv)，配对区间、覆盖审计与五折分析见[SUMMARY.json](SUMMARY.json)。', '']
    if protocol.get('limitations'):
        lines += ['协议记录的限制：', '', str(protocol['limitations']), '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def report_root(root, summary, protocol):
    top = ['# 空间锚定宽视野实验', '', f"总体状态：**{summary['status']}**。", '',
           '| 阶段 | 状态 | 已完成数据集 |', '| --- | --- | --- |']
    for stage, value in summary['stages'].items():
        top.append(f"| [{stage}]({stage}/REPORT_ZH.md) | {value['status']} | {', '.join(value['completed_datasets']) or '无'} |")
    full = summary['stages'].get('full', {})
    if full.get('status') == 'complete':
        datasets = full['datasets']
        methods = list(dict.fromkeys(row['method'] for record in datasets.values() for row in record['matched_rows']))
        top += ['', '## 完整验证集：相同参考g0', '',
                '单位为mIoU%。所有方法共享局部DonorBefore、24槽语义头、Alias、计分及背景规则。普通加权融合独立选取alpha；其余各行使用表中相同g0。', '',
                '| 配置 | ' + ' | '.join(LABELS.get(dataset, dataset) for dataset in datasets) + ' |',
                '| --- | ' + ' | '.join('---:' for _ in datasets) + ' |',
                '| 图像数 | ' + ' | '.join(str(record['images']) for record in datasets.values()) + ' |',
                '| g0 | ' + ' | '.join(format(record['reference_g0'], 'g') for record in datasets.values()) + ' |']
        for method in methods:
            values = [next((f"{row['mean_iou_percent']:.4f}" for row in record['matched_rows'] if row['method'] == method), '—') for record in datasets.values()]
            top.append('| ' + ('最佳普通加权融合' if method == 'UniformBlend' else method) + ' | ' + ' | '.join(values) + ' |')
        top += ['', '## 主要配对比较', '', '| 对照 | Anchored平均净收益pp | 分组配对95%区间 | 正/负点估计域数 |', '| --- | ---: | --- | ---: |']
        for row in full['equal_domain_effects']:
            if row['method'] not in ('Coordinate', 'Legacy'):
                continue
            low, high = row['paired_95ci_pp']
            top.append(f"| {row['method']} | {row['anchored_minus_arm_pp']:+.4f} | [{low:+.4f}, {high:+.4f}] | {row['positive_point_datasets']}/{row['negative_point_datasets']} |")
        top += ['', '## 各自相同预算选择g0', '',
                '三个核心方法均在{0.25,0.5,0.75,1}中各自选择。这是开发标签上的最佳配置分析，不能代替上面的冻结参考步长结论。', '',
                '| 数据集 | Coordinate最佳g0 / mIoU | Legacy最佳g0 / mIoU | Anchored最佳g0 / mIoU |', '| --- | ---: | ---: | ---: |']
        for dataset, record in datasets.items():
            values = [next(f"{row['g0']:g} / {row['mean_iou_percent']:.4f}" for row in record['retuned_rows'] if row['method'] == method) for method in CORE]
            top.append('| ' + LABELS.get(dataset, dataset) + ' | ' + ' | '.join(values) + ' |')
        global_gain = summary['global_gain']
        top += ['', '## 所有数据集共用一个g0', '',
                '只使用已声明并完成的四个步长，不新增评测。每个数据集等权，以下为各域mIoU的算术平均；不是把所有图像合并后的mIoU。', '',
                '| 统一g0 | Coordinate平均mIoU | Legacy平均mIoU | Anchored平均mIoU |', '| ---: | ---: | ---: | ---: |']
        for gain in global_gain['gains']:
            values = [next(f"{row['equal_domain_mean_iou_percent']:.4f}" for row in global_gain['rows']
                           if row['method'] == method and row['g0'] == gain) for method in CORE]
            top.append(f'| {gain:g} | ' + ' | '.join(values) + ' |')
        top += ['', '| 方法 | 最佳统一g0 | 七域平均mIoU |', '| --- | ---: | ---: |']
        for row in global_gain['best_shared_gain']:
            top.append(f"| {row['method']} | {row['g0']:g} | {row['equal_domain_mean_iou_percent']:.4f} |")
        top += ['', '最佳统一g0用完整开发标签选择，是减少逐域参数的开发分析；不是未查看测试集上的结论，也不改变冻结参考g0的模型替换判定。']
    decision = summary['promotion']
    translations = dict(promote='达到预声明替换条件', not_promoted='未达到预声明替换条件', pending='证据尚不完整，暂不判定')
    top += ['', '## 是否替换当前模型', '', '**' + translations[decision['decision']] + '。**', '',
            '以下条件在新评测前声明。精度按冻结参考g0判断；成本基线为同配置Legacy，要求七个数据集全部满足。Coordinate成本同时报告，作为更简单的替代方案。', '',
            '| 条件 | 实测 | 预声明要求 | 状态 |', '| --- | ---: | --- | --- |']
    names = dict(macro_vs_Coordinate='平均优于Coordinate', macro_vs_Legacy='平均优于Legacy',
                 VOC21_vs_Legacy_noninferiority='VOC21相对Legacy不劣', potsdam_vs_Coordinate='Potsdam不退化',
                 vaihingen_vs_Coordinate='Vaihingen不退化', latency_vs_Legacy='最差域时延比',
                 allocated_memory_vs_Legacy='最差域allocated增量')
    for row in decision['checks']:
        value = '待完成' if row['observed'] is None else f"{row['observed']:+.4f}"
        top.append(f"| {names[row['check']]} | {value} | {row['criterion']} | {dict(passed='通过', failed='未通过', pending='待完成')[row['status']]} |")
    failed = [names[row['check']] for row in decision['checks'] if row['status'] == 'failed']
    if failed:
        top += ['', '未通过的条件：' + '、'.join(failed) + '。保留本轮候选及负结果；当前证据不支持按预声明规则替换原模型。']
    top += ['', '这些是开发阶段的替换条件，不是CVPR录用或创新性的充分条件。区间条件于历史开发配置，不能据此声称独立测试泛化；未测LPOSS等方法的优劣仍未知。', '',
            '## 独立成本', '', f"成本状态：{summary['costs']['status']}。缺失项不会填入推测值。", '',
            '单位为ms与MiB；allocated、reserved均不是nvidia-smi驻留显存。固定7图，每图2次预热、7次同步测量，49次合并中位数；CPU RGB tensor到原尺寸CPU uint8标签，排除解码、初始化及离线文本编码。', '']
    if summary['costs']['rows']:
        top += ['| 数据集 | 方法 | ms | allocated MiB | reserved MiB |', '| --- | --- | ---: | ---: | ---: |']
        for row in summary['costs']['rows']:
            top.append(f"| {LABELS.get(row['dataset'], row['dataset'])} | {row['method']} | {row['median_ms']:.2f} | {row['peak_allocated_mib']:.2f} | {row['peak_reserved_mib']:.2f} |")
        top.append('')
    if summary.get('figures'):
        top += ['## 完整验证集图表', '', '![配对净收益及区间](paired_readout_intervals.png)', '',
                '![精度与独立成本](core_accuracy_cost.png)', '',
                '矢量版：[配对区间PDF](paired_readout_intervals.pdf)、[精度与成本PDF](core_accuracy_cost.pdf)。', '']
    top += ['筛查子集、冒烟和完整验证集分开报告，全部负结果保留。配对分组区间、同预算最优g0以及五折标量选择见[完整报告](full/REPORT_ZH.md)。', '',
            '机器汇总：[SUMMARY.json](SUMMARY.json)；全部精度：[ALL_RESULTS.csv](ALL_RESULTS.csv)；成本：[COST_RESULTS.csv](COST_RESULTS.csv)。', '']
    (root / 'REPORT_ZH.md').write_text('\n'.join(top), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--stage', choices=('all', 'screen', 'full', 'smoke'), default='all')
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--refresh', action='store_true', help='Recompute statistics even when source file metadata is unchanged.')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = args.root.resolve()
    protocol, expected = read(root / 'protocol.json'), read(root / 'expected_keys.json')
    stages = ('smoke', 'screen', 'full') if args.stage == 'all' else (args.stage,)
    summary = read(root / 'SUMMARY.json') if (root / 'SUMMARY.json').exists() else dict(stages={})
    summary.setdefault('stages', {})
    for stage in stages:
        signature = source_signature(root, protocol, stage, args.repeats)
        previous = read(root / stage / 'SUMMARY.json') if (root / stage / 'SUMMARY.json').exists() else {}
        if not args.refresh and previous.get('source_signature') == signature:
            summary['stages'][stage] = previous
            report_stage(root / stage / 'REPORT_ZH.md', previous, protocol)
        else:
            summary['stages'][stage] = collect_stage(root, protocol, expected, stage, args.repeats)
            summary['stages'][stage]['source_signature'] = signature
            dump(root / stage / 'SUMMARY.json', summary['stages'][stage])
    summary['costs'] = collect_costs(root, protocol)
    full = summary['stages'].get('full', {})
    summary['status'] = ('complete' if full.get('status') == 'complete' and summary['costs']['status'] == 'complete'
                         else 'full_complete_cost_pending' if full.get('status') == 'complete'
                         else 'screen_complete_full_pending' if summary['stages'].get('screen', {}).get('status') == 'complete'
                         else 'incomplete')
    summary['limitations'] = protocol.get('limitations', [])
    summary['global_gain'] = global_gain_analysis(full, protocol)
    summary['promotion'] = promotion_decision(summary, protocol)
    summary['figures'] = scientific_plots(root, full, summary['costs'])
    dump(root / 'SUMMARY.json', summary)
    all_rows = [dict(stage=stage, **row) for stage, value in summary['stages'].items()
                for record in value['datasets'].values() if record['status'] == 'complete' for row in record['all_rows']]
    csv_write(root / 'ALL_RESULTS.csv', all_rows)
    report_root(root, summary, protocol)
    print(json.dumps(dict(status=summary['status'], stages={key: value['status'] for key, value in summary['stages'].items()},
                          costs=summary['costs']['status']), ensure_ascii=False))


if __name__ == '__main__':
    main()
