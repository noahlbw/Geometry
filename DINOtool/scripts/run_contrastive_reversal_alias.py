"""Run and verify one frozen contrastive-reversal source on eight idle GPUs."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.alias_spatial_allocation import ROUNDING_GUARD
from dinotool.contrastive_reversal_alias import IMPLEMENTATION, METHODS, ALIAS_SHUFFLES, CONFIG
from eval_pair_context_reader import SOURCE, RETAINED
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(SOURCE/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['signature'] != prior['signature'] or row['implementation'] != IMPLEMENTATION
            or row['config'] != asdict(CONFIG) or row['source_config'] != asdict(CONFIG)
            or not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_loaded']
            or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Invalid complete reversal source pilot.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as arrays, np.load(
            SOURCE/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        if arrays['sample_keys'].tolist() != prior['sample_keys']:
            raise RuntimeError('Changed per-image sequence.')
        for p, group in row['metrics'].items():
            if set(group) != set(METHODS):
                raise RuntimeError('Missing matched source controls.')
            for m, metric in group.items():
                if not np.array_equal(arrays[p+'__'+m].sum(0), metric['confusion_matrix']):
                    raise RuntimeError('Confusion reconstruction mismatch.')
                if m in RETAINED and not np.array_equal(arrays[p+'__'+m], previous[p+'__'+m]):
                    raise RuntimeError('Historical per-image controls changed.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for p, group in row['transitions'].items():
            base = np.asarray(row['metrics'][p]['Anchored_Exact']['confusion_matrix'])
            for m, entry in group.items():
                counts = arrays[p+'__'+m].sum(0)
                if (not np.array_equal(counts, entry['counts']) or not np.array_equal(counts.sum(1).T, base)
                        or not np.array_equal(counts.sum(0).T, row['metrics'][p][m]['confusion_matrix'])):
                    raise RuntimeError('Transition endpoints changed.')
    with np.load(root/dataset/'per_image_source_scores.npz', allow_pickle=False) as arrays:
        if arrays['sample_keys'].tolist() != prior['sample_keys'] or any(not np.isfinite(arrays[name]).all() for name in arrays.files if name != 'sample_keys'):
            raise RuntimeError('Changed/nonfinite source scores.')
    for image in row['diagnostics'].values():
        if image['operator_replay_max_error'] > 1e-10 or image['new_intervention_forwards'] != 0:
            raise RuntimeError('Changed Geometry/source observations.')
        for p in row['metrics']:
            diag = image[p]
            if max(diag['replay_max_errors'].values()) > 1e-4:
                raise RuntimeError('Historical reader replay failed.')
            for m, stats in diag['source'].items():
                if stats['canonical_rejection_max'] != 0 or stats['capacity_excess_max'] > ROUNDING_GUARD:
                    raise RuntimeError('Protected source capacity failed.')
                if m in ALIAS_SHUFFLES and stats['alias_spectrum_max_error'] != 0:
                    raise RuntimeError('Alias shuffle spectrum changed.')
            for m, stats in diag['allocation_controls'].items():
                if (stats['budget_max_error'] > 1e-10 or stats['capacity_excess_max'] > ROUNDING_GUARD
                        or (m.startswith('SpatialShuffle') and stats['spectrum_max_error'] > 1e-12)):
                    raise RuntimeError('Matched action controls changed.')
    if any(not (root/dataset/'numerical_cache'/f'{i}.npz').is_file() for i in range(8)):
        raise RuntimeError('Missing pre-mask fields.')
    row['coverage_verified'] = True
    row['exact_per_image_controls'] = list(RETAINED)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_contrastive_reversal_alias.py', session_prefix='gcrv03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify)
