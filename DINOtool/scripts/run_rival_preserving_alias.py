"""Run the frozen rival-preserving writer on eight authorized idle GPUs."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.contrastive_reversal_alias import CONFIG as SOURCE_CONFIG, METHODS as RETAINED
from dinotool.rival_preserving_alias import IMPLEMENTATION, METHODS, ALIAS_SHUFFLES, SPATIAL_SHUFFLES, CONFIG
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_region_semantic_suite_a800 import TOOL
from run_sat_geometry_transport_suite import read_json


PREVIOUS = TOOL/'results/contrastive_reversal_alias_screen_20261003'


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['signature'] != prior['signature'] or row['implementation'] != IMPLEMENTATION
            or row['config'] != asdict(CONFIG) or row['source_config'] != asdict(SOURCE_CONFIG)
            or not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_loaded']
            or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Invalid complete rival-preserving pilot.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as arrays, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        if arrays['sample_keys'].tolist() != prior['sample_keys']:
            raise RuntimeError('Changed per-image sequence.')
        for p, group in row['metrics'].items():
            if set(group) != set(METHODS):
                raise RuntimeError('Missing competitive writer controls.')
            for m, metric in group.items():
                if not np.array_equal(arrays[p+'__'+m].sum(0), metric['confusion_matrix']):
                    raise RuntimeError('Confusion sum mismatch.')
                if m in RETAINED and not np.array_equal(arrays[p+'__'+m], previous[p+'__'+m]):
                    raise RuntimeError('Historical per-image endpoint changed: '+m)
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for p, group in row['transitions'].items():
            base = np.asarray(row['metrics'][p]['Anchored_Exact']['confusion_matrix'])
            for m, entry in group.items():
                counts = arrays[p+'__'+m].sum(0)
                if (not np.array_equal(counts, entry['counts']) or not np.array_equal(counts.sum(1).T, base)
                        or not np.array_equal(counts.sum(0).T, row['metrics'][p][m]['confusion_matrix'])):
                    raise RuntimeError('Transition endpoints changed.')
    for index, image in enumerate(row['diagnostics'].values()):
        if image['operator_replay_max_error'] > 1e-10 or image['new_intervention_forwards'] != 0:
            raise RuntimeError('Changed Geometry/source observations.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as previous:
            for p in row['metrics']:
                for m in RETAINED:
                    if not np.array_equal(current[p+'__'+m+'__scores'], previous[p+'__'+m+'__scores']):
                        raise RuntimeError('Historical numerical score changed: '+m)
                if not np.array_equal(current[p+'__pair_risk'], previous[p+'__pair_reversal_risk']):
                    raise RuntimeError('Frozen reversal source changed.')
        for p in row['metrics']:
            diag = image[p]
            if max(diag['replay_max_errors'].values()) > 1e-4:
                raise RuntimeError('Historical actual-readout replay failed.')
            for m, stats in diag['rival_reader']['sources'].items():
                if (stats['normal_equation_max_error'] > CONFIG.numerical_tolerance
                        or stats['gauge_max_error'] > CONFIG.numerical_tolerance
                        or stats['directed_capacity_excess_max'] > 1e-6 or stats['potential_bound_excess_max'] > 1e-6
                        or stats['canonical_risk_max'] != 0):
                    raise RuntimeError('Competitive source consistency/protection failed.')
                if m in ALIAS_SHUFFLES and stats['risk_spectrum_max_error'] != 0:
                    raise RuntimeError('Alias/rival spectrum changed.')
            for m, stats in diag['rival_reader']['controls'].items():
                if (stats['pair_budget_max_error'] > 1e-10 or stats['potential_budget_max_error'] > 1e-10
                        or stats['capacity_excess_max'] > 1e-6
                        or (m in SPATIAL_SHUFFLES and stats['pair_spectrum_max_error'] > 1e-12)):
                    raise RuntimeError('Directional-pair control budgets/caps/spectra failed.')
    row['coverage_verified'] = True
    row['exact_per_image_controls'] = list(RETAINED)
    row['exact_source_risk_replay'] = True
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_rival_preserving_alias.py', session_prefix='grpa03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify)
