"""Frozen same-path alias reliability pilot on authorized idle physical GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.matched_contribution_alias import IMPLEMENTATION, CONFIG, METHODS, LEGACY, ALIAS_SHUFFLES, SPATIAL_SHUFFLES, GATE
from dinotool.target_context_alias import TargetContextConfig
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_region_semantic_suite_a800 import TOOL
from run_sat_geometry_transport_suite import read_json


PREVIOUS = TOOL/'results/rival_preserving_alias_reader_screen_20261003'


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['signature'] != prior['signature'] or row['implementation'] != IMPLEMENTATION
            or row['config'] != asdict(CONFIG) or row['source_config'] != asdict(TargetContextConfig())
            or not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_loaded']
            or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Invalid matched contribution suite/config/coverage.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        if current['sample_keys'].tolist() != prior['sample_keys']:
            raise RuntimeError('Changed per-image sequence.')
        for p, group in row['metrics'].items():
            if set(group) != set(METHODS):
                raise RuntimeError('Missing matched arms.')
            for method, metric in group.items():
                if not np.array_equal(current[p+'__'+method].sum(0), metric['confusion_matrix']):
                    raise RuntimeError('Confusion reconstruction failed.')
                if method in LEGACY and not np.array_equal(current[p+'__'+method], previous[p+'__'+method]):
                    raise RuntimeError('Historical per-image endpoint changed: '+method)
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for p, group in row['transitions'].items():
            for method, entry in group.items():
                counts = arrays[p+'__'+method].sum(0)
                if (not np.array_equal(counts, entry['counts'])
                        or not np.array_equal(counts.sum(1).T, row['metrics'][p]['Anchored_Exact']['confusion_matrix'])
                        or not np.array_equal(counts.sum(0).T, row['metrics'][p][method]['confusion_matrix'])):
                    raise RuntimeError('Transition endpoint mismatch.')
    for index, image in enumerate(row['diagnostics'].values()):
        if (image['operator_replay_max_error'] > 1e-10 or not image['shared_support_replay_exact']
                or image['support_shuffle_spectrum_error'] != 0):
            raise RuntimeError('Geometry/shared source support changed.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as previous:
            for p in row['metrics']:
                for method in LEGACY:
                    if not np.array_equal(current[p+'__'+method+'__scores'], previous[p+'__'+method+'__scores']):
                        raise RuntimeError('Historical numerical field changed.')
        for p in row['metrics']:
            diag = image[p]
            if max(diag['replay_max_errors'].values()) > 1e-4:
                raise RuntimeError('Actual source/readout replay failed.')
            for method, stats in diag['sources'].items():
                if (max(stats['normal_equation_max_error'], stats['gauge_max_error']) > 1e-10
                        or stats['directed_capacity_excess_max'] > 1e-6 or stats['canonical_risk_max'] != 0
                        or stats['zero_risk_writer_max'] != 0 or (method in ALIAS_SHUFFLES and stats['alias_spectrum_max_error'] != 0)):
                    raise RuntimeError('Source/writer numerical constraints failed.')
            for method, stats in diag['controls'].items():
                if (stats['pair_budget_max_error'] > 1e-10 or stats['capacity_excess_max'] > 1e-6
                        or (method in SPATIAL_SHUFFLES and stats['pair_spectrum_max_error'] > 1e-12)):
                    raise RuntimeError('Directional control budgets/capacities/spectra changed.')
    row['coverage_verified'] = True
    row['exact_per_image_controls'] = list(LEGACY)
    row['exact_source_path_replay'] = True
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_matched_contribution_alias.py', session_prefix='gmca03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'measurement': 'crop-profiled alias vs normalized rival class score before stencil'})
