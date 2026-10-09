"""Frozen physical8 alias-view screen, with unchanged writer and Geometry."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.fine_alias_view import IMPLEMENTATION, CONFIG, METHODS, REPLAY, GATE
from dinotool.native_alias_noise import CONFIG as SCENARIO_CONFIG
from eval_fine_alias_view import PREVIOUS
from run_pair_context_reader import run
from run_bounded_alias_suite import save
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or len(set(row['sample_keys'])) != 8 or row['sample_keys'] != prior['sample_keys']
            or row['signature'] != prior['signature'] or row['vocabularies'] != prior['vocabularies']
            or row['implementation'] != IMPLEMENTATION or row['config'] != asdict(SCENARIO_CONFIG)
            or not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_loaded']
            or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Changed complete frozen window protocol.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        if current['sample_keys'].tolist() != row['sample_keys']:
            raise RuntimeError('Changed sample order.')
        for scene, protocols in row['metrics'].items():
            for p, arms in protocols.items():
                if set(arms) != set(METHODS):
                    raise RuntimeError('Missing prediction endpoint.')
                for name, metric in arms.items():
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Confusion reconstruction differs.')
                    if name in REPLAY and not np.array_equal(current[key], old[scene+'__'+p+'__'+REPLAY[name]]):
                        raise RuntimeError('Historical per-image endpoint differs.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for scene, protocols in row['transitions'].items():
            for p, arms in protocols.items():
                for name, entry in arms.items():
                    counts = arrays[scene+'__'+p+'__'+name].sum(0)
                    if (not np.array_equal(counts, entry['counts'])
                            or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p]['NoAdmission_Exact']['confusion_matrix'])
                            or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p][name]['confusion_matrix'])):
                        raise RuntimeError('Transition reconstruction differs.')
    for index, image in enumerate(row['diagnostics'].values()):
        if (image['operator_replay_max_error'] > 1e-10 or max(image['local_replay_max_error'].values()) > 1e-4
                or image['fine_token_original_pixels'] != 8 or image['new_image_observation_forwards'] != 0
                or not image['cached_fine_source_precedes_masks']):
            raise RuntimeError('Changed frozen image source.')
        for protocols in image['scenarios'].values():
            for diag in protocols.values():
                if (diag['baseline_readout_replay_max_error'] > 1e-4 or diag['native16_control_score_replay_max_error'] != 0
                        or diag['native16_control_risk_replay_max_error'] != 0
                        or diag['zero_risk_writer_max_error'] != 0 or diag['canonical_view_risk_max'] != 0
                        or diag['class_mean_view_budget_error'] > 1e-10 or max(diag['alias_spectrum_errors'].values()) != 0
                        or diag['spatial_view_spectrum_error'] != 0
                        or any(s['capacity_excess_max'] > 1e-10 for s in diag['sources'].values())
                        or any(s['count_max_error'] != 0 or s['canonical_risk_max'] != 0
                            for s in diag['independent_observer'].values())):
                    raise RuntimeError('Changed physical8 source/writer/budget contract.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as old:
            for scene, protocols in row['metrics'].items():
                for p in protocols:
                    for name, earlier in REPLAY.items():
                        if not np.array_equal(current[scene+'__'+p+'__'+name+'__scores'], old[scene+'__'+p+'__'+earlier+'__scores']):
                            raise RuntimeError('Archived numerical endpoint differs.')
    row.update(coverage_verified=True, exact_previous_controls=REPLAY, alias_view_config=asdict(CONFIG),
        source_fields_frozen_before_prediction_masks=True, original_observer_and_geometry_unchanged=True)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_fine_alias_view.py', session_prefix='gfav03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'alias_view_config': asdict(CONFIG),
            'historical_prediction_reference': str(PREVIOUS), 'original_geometry_writer_unchanged': True,
            'only_alias_view_source_changed': True, 'fine_rgb_observations_cached': True,
            'independent_observer_controls': 'same wide observer; no Geometry anchor or reconstruction',
            'real_raw_llm_provenance_available': False})
