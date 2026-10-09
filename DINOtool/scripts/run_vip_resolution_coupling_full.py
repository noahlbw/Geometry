"""Frozen resized-input eight-domain accuracy evaluation, without new timing."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

from dinotool.shared_rival_soft_alias import CONFIG, METHODS, PRIMARY
from dinotool.vip_resolution_coupling import IMPLEMENTATION, VIEW_PROTOCOL
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import TOOL, idle
from run_sat_geometry_transport_suite import read_json
from run_shared_rival_soft_full import SHARDS, TOTALS, launch as original_launch, monitor


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False):
    if phase != 'full' or vip:
        raise ValueError('Only the requested accuracy evaluation is allowed.')
    return original_launch(root, dataset, gpu, phase, shard, shards,
        evaluator='scripts/eval_vip_resolution_coupling.py', session_prefix='gvrf05')


def verify_dataset(root, dataset, vip=False):
    if vip:
        raise ValueError('Reuse existing VIP results, do not rerun it.')
    output = root / 'full' / dataset / 'merged.json'
    if output.exists():
        raise RuntimeError('Preserve existing merged output.')
    row = merge([str(root / 'full' / dataset / ('s' + str(s)))
                 for s in range(SHARDS.get(dataset, 1))], str(output))
    old = read_json(TOOL / 'results/shared_rival_soft20_full_20261005/full' / dataset / 'merged.json')
    signature = row['signature']
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or signature['sample_keys'] != old['signature']['sample_keys']
            or signature['implementation'] != IMPLEMENTATION
            or signature['gear']['input_protocol'] != VIEW_PROTOCOL
            or signature['vocabulary'] != old['signature']['vocabulary']
            or signature['checkpoints'] != old['signature']['checkpoints']):
        raise RuntimeError('Resized full coverage/frozen identities differ: ' + dataset)
    for p, values in row['diagnostics'].items():
        if (values['native_resolution_encodings'] or values['fine_forwards']
                or values['geometry_encodings'] > 4 or values['wide_encodings'] > 4):
            raise RuntimeError('Visual budget exceeded: ' + dataset + '/' + p)
    return str(output)


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if root.exists() or any(not idle(g) for g in range(8)):
        raise RuntimeError('Existing root or occupied GPU; preserve all work.')
    root.mkdir(parents=True)
    save(root / 'protocol.json', dict(implementation=IMPLEMENTATION, methods=METHODS,
        primary=PRIMARY, input_protocol=VIEW_PROTOCOL, admission=asdict(CONFIG), totals=TOTALS,
        fixed_aliases_per_class=20, parameter_tuning=False, timing_requested=False,
        comparison='Existing finite VIP All20; same words/input layout, not identical text/readout/background.'))
    start = time.time()
    completed, failures = {}, {}
    active = {g: launch(root, d, g, 'full') for g, d in enumerate(('vdd', 'potsdam', 'udd5', 'oem'))}
    active.update({g: launch(root, 'flair1', g, 'full', g - 4, 4) for g in range(4, 8)})
    pending = [('loveda', s, 4, False) for s in range(4)] + [
        ('vaihingen', 0, 1, False), ('landcoverai', 0, 1, False)]
    monitor(root, active, pending, 'full', completed, failures, launcher=launch,
            dataset_verifier=verify_dataset)
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete',
        completed=completed, failures=failures, unique_full_images=sum(TOTALS.values()),
        suite_wall_seconds=time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
