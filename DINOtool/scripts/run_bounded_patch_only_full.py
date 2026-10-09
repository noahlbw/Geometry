"""Finish the frozen patch-only pilot: serial timings, then remaining full domains."""
import argparse
from pathlib import Path
import time

import numpy as np

from dinotool.bounded_patch_only import IMPLEMENTATION, METHODS, PRIMARY, PROTOCOL
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import TOOL, idle
from run_sat_geometry_transport_suite import alive, read_json
from run_shared_rival_soft_full import TOTALS, launch as original_launch, monitor, verify_job


PREFIX = 'gbpo05'
PILOT = ('vdd', 'potsdam', 'udd5', 'oem')
SHARDS = dict(loveda=4, flair1=4)
BASELINE = TOOL / 'results/bounded_physical_coupling_20261005/full'


def check_dataset(path, dataset):
    row, baseline = read_json(path), read_json(BASELINE / dataset / 'merged.json')
    signature = row['signature']
    if (row['status'] != 'complete' or not row['coverage_verified']
            or row['processed_images'] != TOTALS[dataset] or row['total_images'] != TOTALS[dataset]
            or signature['sample_keys'] != baseline['signature']['sample_keys']
            or signature['implementation'] != IMPLEMENTATION
            or signature['gear']['input_protocol'] != PROTOCOL
            or signature['vocabulary'] != baseline['signature']['vocabulary']
            or signature['checkpoints'] != baseline['signature']['checkpoints']
            or any(n != 20 for counts in signature['vocabulary']['counts'].values() for n in counts)):
        raise RuntimeError('Frozen coverage/source changed: ' + dataset)
    for protocol, group in row['metrics'].items():
        for method in METHODS[:2]:
            if not np.array_equal(group[method]['confusion_matrix'], baseline['metrics'][protocol][method]['confusion_matrix']):
                raise RuntimeError('Original endpoint changed: ' + dataset + '/' + protocol + '/' + method)
        diagnostics = row['diagnostics'][protocol]
        if (diagnostics['native_resolution_encodings'] or diagnostics['fine_forwards']
                or diagnostics['geometry_encodings'] > 4 or diagnostics['wide_encodings'] > 4):
            raise RuntimeError('Encoding budget exceeded: ' + dataset)
    return row


def verify_dataset(root, dataset, vip=False):
    if vip:
        raise ValueError('No new VIP accuracy run is part of this suite.')
    output = root / 'full' / dataset / 'merged.json'
    paths = [str(root / 'full' / dataset / f's{s}') for s in range(SHARDS.get(dataset, 1))]
    if not output.exists():
        merge(paths, str(output))
    check_dataset(output, dataset)
    return str(output)


def job_info(root, dataset, phase, gpu=7):
    output = root / phase / dataset / 's0'
    return dict(dataset=dataset, name=dataset, gpu=gpu, phase=phase, shard=0, shards=1,
                output=str(output), log=str(output.with_suffix('.log')),
                session=f'{PREFIX}_{phase}_{dataset}_s0', vip=False)


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False):
    if phase == 'benchmark':
        # Timing uses the same all-idle condition as the completed pilot panels.
        while not all(idle(g) for g in range(8)):
            time.sleep(15)
    return original_launch(root, dataset, gpu, phase, shard, shards, vip,
                          evaluator='scripts/eval_bounded_patch_only.py', session_prefix=PREFIX)


def check_timing(path, dataset):
    row = read_json(Path(path))
    baseline = read_json(BASELINE / dataset / 'merged.json')
    if (row['status'] != 'complete' or row['target_masks_loaded']
            or not row['weights_frozen'] or not row['head_weights_unchanged']
            or row['implementation'] != IMPLEMENTATION or row['input_protocol'] != PROTOCOL
            or len(row['images']) != 3
            or any(PRIMARY not in image['timings'] or 'VIP_All20' not in image['timings'] for image in row['images'])):
        raise RuntimeError('Unmatched or incomplete timing: ' + str(path))
    if row['source_vocabulary'] != baseline['signature']['vocabulary']:
        raise RuntimeError('Timing vocabulary changed.')
    return row


def main(root, resume_after_timing=False):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if not root.is_dir() or (root / 'suite_results.json').exists():
        raise RuntimeError('Existing controller or missing pilot; do not relaunch.')
    if resume_after_timing:
        state = read_json(root / 'suite_status.json')
        if (not state or state['phase'] != 'benchmark' or state['status'] != 'complete'
                or state['active'] or state['pending'] or state['failures']
                or set(state['completed']) != set(TOTALS)):
            raise RuntimeError('Recovery only allowed after all eight timings completed.')
        for dataset in set(TOTALS) - set(PILOT):
            if (root / 'full' / dataset).exists():
                raise RuntimeError('Accuracy outputs already exist; preserve them: ' + dataset)
    elif (root / 'suite_status.json').exists():
        raise RuntimeError('Existing controller state; no relaunch.')
    completed = {}
    for dataset in PILOT:
        verify_job(job_info(root, dataset, 'full'))
        completed[dataset] = verify_dataset(root, dataset)
    if not resume_after_timing:
        save(root / 'full_summary.json', dict(completed=completed, failures={}))
    protocol = dict(implementation=IMPLEMENTATION, primary=PRIMARY, methods=METHODS,
        input_protocol=PROTOCOL, totals=TOTALS, shards=SHARDS, fixed_aliases_per_class=20,
        target_label_tuning=False, alias_admission='none',
        declaration='Retain strength2 for every domain; no per-domain winner switching.',
        novelty='Patch/prefix routes and strength are configurations, not an independent innovation.',
        development='Previously developed domains; exploratory, not independent validation.')
    if resume_after_timing:
        import json
        if read_json(root / 'protocol.json') != json.loads(json.dumps(protocol)):
            raise RuntimeError('Frozen declaration changed; do not resume.')
    else:
        save(root / 'protocol.json', protocol)
    start = time.time()
    failures, timing, active, pending = {}, {}, {}, []
    for dataset in TOTALS:
        job = job_info(root, dataset, 'benchmark')
        path = Path(job['output']) / 'results.json'
        if alive(job['session']):
            if active:
                raise RuntimeError('Overlapping timing workers are not allowed.')
            active[7] = job
        elif path.exists():
            verify_job(job)
            check_timing(path, dataset)
            timing[dataset] = str(path)
        elif Path(job['output']).exists() or Path(job['log']).exists():
            raise RuntimeError('Terminal incomplete timing; preserve output: ' + job['session'])
        else:
            pending.append((dataset, 0, 1, False))
    monitor(root, active, pending, 'benchmark', timing, failures, launcher=launch,
            dataset_verifier=verify_dataset)
    if failures:
        save(root / 'suite_results.json', dict(status='failed', completed=completed, timing=timing, failures=failures))
        return
    for dataset, path in timing.items():
        check_timing(path, dataset)
    save(root / 'timing_summary.json', dict(completed=timing))
    active = {}
    for gpu in range(4, 8):
        while not idle(gpu):
            time.sleep(15)
        active[gpu] = launch(root, 'flair1', gpu, 'full', gpu - 4, 4)
    pending = [('loveda', s, 4, False) for s in range(4)] + [
        ('vaihingen', 0, 1, False), ('landcoverai', 0, 1, False)]
    monitor(root, active, pending, 'full', completed, failures, launcher=launch,
            dataset_verifier=verify_dataset)
    save(root / 'full_summary.json', dict(completed=completed, failures=failures))
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete',
        completed=completed, timing=timing, failures=failures,
        unique_full_images=sum(TOTALS.values()), remaining_suite_wall_seconds=time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-after-timing', action='store_true')
    args = parser.parse_args()
    main(args.root, args.resume_after_timing)
