"""Frozen same-field full ablation; never tune or promote a control by dataset."""
import argparse
from pathlib import Path
import time

import numpy as np

from dinotool.bounded_coupling_controls import IMPLEMENTATION, METHODS, PROTOCOL
from dinotool.bounded_patch_only import PRIMARY
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import TOOL, idle
from run_sat_geometry_transport_suite import read_json
from run_shared_rival_soft_full import TOTALS, launch as original_launch, monitor


SHARDS = dict(loveda=4, flair1=4)
SOURCE = TOOL / 'results/bounded_patch_only_20261005/full'


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False):
    return original_launch(root, dataset, gpu, phase, shard, shards, vip,
        evaluator='scripts/eval_bounded_coupling_controls_full.py', session_prefix='gbcf05')


def verify_dataset(root, dataset, vip=False):
    paths = [str(root / 'full' / dataset / f's{s}') for s in range(SHARDS.get(dataset, 1))]
    output = root / 'full' / dataset / 'merged.json'
    if not output.exists():
        merge(paths, str(output))
    row, original = read_json(output), read_json(SOURCE / dataset / 'merged.json')
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or row['signature']['sample_keys'] != original['signature']['sample_keys']
            or row['signature']['implementation'] != IMPLEMENTATION
            or row['signature']['vocabulary'] != original['signature']['vocabulary']
            or row['signature']['checkpoints'] != original['signature']['checkpoints']):
        raise RuntimeError('Frozen full source or coverage changed: ' + dataset)
    for protocol, group in row['metrics'].items():
        for name in ('Geometry', 'NoAdmission_Exact', PRIMARY):
            if not np.array_equal(group[name]['confusion_matrix'], original['metrics'][protocol][name]['confusion_matrix']):
                raise RuntimeError('Frozen primary/control endpoint changed: ' + dataset + '/' + name)
        diagnostic = row['diagnostics'][protocol]
        if (diagnostic['fine_forwards'] or diagnostic['native_resolution_encodings']
                or diagnostic['geometry_encodings'] > 4 or diagnostic['wide_encodings'] > 4):
            raise RuntimeError('Encoding budget changed.')
    return str(output)


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if root.exists():
        raise RuntimeError('Existing full ablation; preserve outputs.')
    root.mkdir()
    save(root / 'protocol.json', dict(implementation=IMPLEMENTATION, input_protocol=PROTOCOL,
        methods=METHODS, primary=PRIMARY, totals=TOTALS, shards=SHARDS,
        fixed_aliases_per_class=20, coefficients_unchanged_from_96_image_audit=True,
        controls_are_not_new_candidates=True, target_label_tuning=False,
        development='Developed domains; frozen full module ablation, not independent validation.'))
    active = {}
    for gpu in range(4, 8):
        while not idle(gpu):
            time.sleep(15)
        active[gpu] = launch(root, 'flair1', gpu, 'full', gpu - 4, 4)
    pending = [(d, 0, 1, False) for d in ('vdd', 'potsdam', 'udd5', 'oem')]
    pending += [('loveda', s, 4, False) for s in range(4)]
    pending += [('vaihingen', 0, 1, False), ('landcoverai', 0, 1, False)]
    completed, failures = {}, {}
    started = time.time()
    monitor(root, active, pending, 'full', completed, failures, launcher=launch,
            dataset_verifier=verify_dataset)
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete',
        completed=completed, failures=failures, unique_full_images=sum(TOTALS.values()),
        all_arm_suite_wall_seconds=time.time() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
