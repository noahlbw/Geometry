"""Continue the frozen bounded896 candidate from its verified two-domain gate."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

from dinotool.bounded_physical_coupling import IMPLEMENTATION, VIEW_PROTOCOL
from dinotool.shared_rival_soft_alias import CONFIG, METHODS, PRIMARY
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import TOOL, idle
from run_sat_geometry_transport_suite import read_json
from run_shared_rival_soft_full import TOTALS, launch as original_launch, monitor


SHARDS = dict(vdd=2, potsdam=4, loveda=4, flair1=4)


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False):
    if vip or phase not in ('full', 'benchmark'):
        raise ValueError('Only frozen accuracy and matched singleton cost are allowed.')
    return original_launch(root, dataset, gpu, phase, shard, shards,
        evaluator='scripts/eval_bounded_physical_coupling.py', session_prefix='gbpc05')


def check_dataset(path, dataset):
    row = read_json(path)
    old = read_json(TOOL / 'results/shared_rival_soft20_full_20261005/full' / dataset / 'merged.json')
    signature = row['signature']
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or row['total_images'] != TOTALS[dataset]
            or signature['sample_keys'] != old['signature']['sample_keys']
            or signature['implementation'] != IMPLEMENTATION
            or signature['gear']['input_protocol'] != VIEW_PROTOCOL
            or signature['vocabulary'] != old['signature']['vocabulary']
            or signature['checkpoints'] != old['signature']['checkpoints']):
        raise RuntimeError('Frozen full coverage/source differs: ' + dataset)
    for values in row['diagnostics'].values():
        if (values['native_resolution_encodings'] or values['fine_forwards']
                or values['geometry_encodings'] > 4 or values['wide_encodings'] > 4):
            raise RuntimeError('Actual encoding budget exceeded: ' + dataset)
    return row


def verify_dataset(root, dataset, vip=False):
    output = root / 'full' / dataset / 'merged.json'
    if output.exists():
        raise RuntimeError('Preserve existing merged output: ' + dataset)
    merge([str(root / 'full' / dataset / ('s' + str(s)))
           for s in range(SHARDS.get(dataset, 1))], str(output))
    check_dataset(output, dataset)
    return str(output)


def vip_source(dataset):
    if dataset == 'vaihingen':
        return TOOL / 'results/shared_rival_soft20_full_20261005/full/vip_all20_vaihingen/merged.json'
    parent = 'vip_official20_distillation_full_20261003' if dataset in ('vdd', 'potsdam') else 'vip_paper_distillation_finite_full_20261003'
    manifest = read_json(TOOL / 'results' / parent / 'suite_results.json')
    return Path(manifest['completed'][dataset])


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if (not root.is_dir() or (root / 'suite_status.json').exists()
            or (root / 'suite_results.json').exists() or any(not idle(g) for g in range(8))):
        raise RuntimeError('Existing full controller or occupied GPUs; do not relaunch.')
    completed, timings, failures = {}, {}, {}
    for dataset in ('vdd', 'potsdam'):
        path = root / 'full' / dataset / 'merged.json'
        row = check_dataset(path, dataset)
        vip = read_json(vip_source(dataset))['metrics'][dataset]['VIP_All20']
        timing = root / 'benchmark' / dataset / 's0/results.json'
        costs = read_json(timing)
        if (row['metrics'][dataset][PRIMARY]['mean_iou_percent'] <= vip['mean_iou_percent']
                or costs['status'] != 'complete'
                or any(x['timings'][PRIMARY]['median_seconds'] >= 1 for x in costs['images'])):
            raise RuntimeError('Two-domain full gate not passed: ' + dataset)
        completed[dataset], timings[dataset] = str(path), str(timing)
    save(root / 'protocol.json', dict(implementation=IMPLEMENTATION, input_protocol=VIEW_PROTOCOL,
        methods=METHODS, primary=PRIMARY, admission=asdict(CONFIG), totals=TOTALS,
        fixed_aliases_per_class=20, target_label_tuning=False,
        comparison='Existing finite All20 VIP; official short/distilled as separate comparators.',
        development='Frozen after the declared two-domain gate; exploratory full validation.'))
    start = time.time()
    active = {g: launch(root, d, g, 'full')
              for g, d in enumerate(('udd5', 'oem', 'vaihingen', 'landcoverai'))}
    active.update({g: launch(root, 'flair1', g, 'full', g - 4, 4) for g in range(4, 8)})
    pending = [('loveda', s, 4, False) for s in range(4)]
    monitor(root, active, pending, 'full', completed, failures, launcher=launch,
            dataset_verifier=verify_dataset)
    save(root / 'full_summary.json', dict(completed=completed, failures=failures))
    if not failures:
        pending = [(d, 0, 1, False) for d in ('udd5', 'oem', 'vaihingen', 'landcoverai', 'flair1', 'loveda')]
        monitor(root, {}, pending, 'benchmark', timings, failures, launcher=launch,
                dataset_verifier=verify_dataset)
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete',
        completed=completed, timing=timings, failures=failures,
        unique_full_images=sum(TOTALS.values()), remaining_suite_wall_seconds=time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
