"""Recover aggregation only from complete canonical-rival predictions.

The frozen evaluator records per-image means; the legacy merger defaults to
tile-weighted means. This finalizer never loads models or reruns predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from merge_gear_ov_shards import merge


def finalize(root, dataset):
    protocol = json.loads((root / 'protocol.json').read_text())
    entry = protocol['datasets'][dataset]
    output = root / dataset
    shard = output / 's0'
    row = json.loads((shard / 'results.json').read_text())
    signature = row['signature']
    if (row['status'] != 'complete' or row['processed_images'] != entry['total_images']
            or row['total_images'] != entry['total_images']
            or not row.get('weights_frozen') or not row.get('head_weights_unchanged')
            or not row.get('target_masks_used_only_after_prediction')):
        raise RuntimeError('Only complete, frozen saved predictions can be finalized.')
    if signature['gear']['method_sources'] != protocol['method_sources']:
        raise RuntimeError('Frozen evaluator identity differs from protocol.')
    vocabulary = Path(entry['vocabulary_config'])
    if signature['vocabulary']['sha256'] != hashlib.sha256(vocabulary.read_bytes()).hexdigest():
        raise RuntimeError('Original20 vocabulary identity changed.')
    previous = json.loads(Path(entry['matched_patch_result']).read_text())
    if (signature['sample_keys'] != previous['signature']['sample_keys']
            or signature['checkpoints'] != previous['signature']['checkpoints']):
        raise RuntimeError('Full original20 sample sequence or checkpoint identity differs.')
    with np.load(shard / 'per_image_confusions.npz', allow_pickle=False) as archive:
        if archive['sample_keys'].tolist() != signature['sample_keys']:
            raise RuntimeError('Per-image archive does not cover the saved sequence.')
        for partition, group in row['metrics'].items():
            for name, metrics in group.items():
                matrices = archive[partition + '__' + name]
                if (len(matrices) != entry['total_images']
                        or not np.array_equal(matrices.sum(0), metrics['confusion_matrix'])):
                    raise RuntimeError('Per-image confusion does not reproduce full result.')
            targets = [np.asarray(item['confusion_matrix']).sum(-1) for item in group.values()]
            if any(not np.array_equal(targets[0], value) for value in targets[1:]):
                raise RuntimeError('Paired scored target counts differ.')
            if entry.get('exact_task_reference'):
                archived = json.loads(Path(entry['exact_task_reference']).read_text())
                if (archived['signature']['sample_keys'] != signature['sample_keys']
                        or not np.array_equal(group['Fixed20_TaskCoupled']['confusion_matrix'],
                                              archived['metrics'][partition]['DomainShared']['confusion_matrix'])):
                    raise RuntimeError('Full VDD task-profile confusion replay differs.')
    timing = json.loads((output / 'timing.json').read_text())
    if timing['status'] != 'complete' or not timing.get('exclusive_gpu_verified'):
        raise RuntimeError('Exclusive timing is incomplete.')
    if (output / 'merged.json').exists():
        raise RuntimeError('Existing merged output; preserve it.')
    merged = merge([str(shard)], str(output / 'merged.json'), diagnostic_weight='images')
    merged.update(paired_scored_targets_equal=True, original20_identity_verified=True,
                  exact_task_reference_replayed=bool(entry.get('exact_task_reference')),
                  weights_frozen=True, head_weights_unchanged=True,
                  per_image_confusions_verified=True,
                  aggregation_recovery='Explicit image-weighted diagnostic means; predictions unchanged.')
    (output / 'merged.json').write_text(json.dumps(merged, indent=2) + '\n')
    old_state = json.loads((output / 'worker_status.json').read_text())
    (output / 'aggregation_recovery.json').write_text(json.dumps(dict(
        previous_worker_state=old_state, predictions_rerun=False,
        original_results_sha256=hashlib.sha256((shard / 'results.json').read_bytes()).hexdigest(),
        recovered_result=str(output / 'merged.json')), indent=2) + '\n')
    (output / 'worker_status.json').write_text(json.dumps(dict(
        status='complete', phase='complete', processed=entry['total_images'],
        total=entry['total_images'], aggregation_recovered=True), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('vdd', 'ade150'), required=True)
    args = parser.parse_args()
    finalize(args.root, args.dataset)
