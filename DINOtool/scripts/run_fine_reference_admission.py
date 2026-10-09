"""Frozen physical8 reference experiment on idle authorized physical GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.fine_reference_admission import IMPLEMENTATION, CONFIG, METHODS, REPLAY, GATE
from eval_fine_reference_admission import PREVIOUS, SEMANTIC
from dinotool.native_alias_noise import CONFIG as SCENARIO_CONFIG
from run_pair_context_reader import run
from run_bounded_alias_suite import save
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or len(set(row['sample_keys'])) != 8 or row['sample_keys'] != prior['sample_keys']
            or row['signature'] != prior['signature'] or row['vocabularies'] != prior['vocabularies']
            or row['implementation'] != IMPLEMENTATION or row['config'] != asdict(SCENARIO_CONFIG)
            or not row['weights_frozen'] or not row['head_weights_unchanged']
            or not row['target_masks_loaded'] or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Changed full fixed-window/weight/input contract.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        if current['sample_keys'].tolist() != row['sample_keys']:
            raise RuntimeError('Changed unique sample order.')
        for scene, protocols in row['metrics'].items():
            for p, arms in protocols.items():
                if set(arms) != set(METHODS):
                    raise RuntimeError('Missing fixed endpoint.')
                for name, metric in arms.items():
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Per-image confusion reconstruction differs.')
                    if name in REPLAY and not np.array_equal(current[key], old[scene+'__'+p+'__'+REPLAY[name]]):
                        raise RuntimeError('Historical endpoint replay differs.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for scene, protocols in row['transitions'].items():
            for p, arms in protocols.items():
                for name, entry in arms.items():
                    counts = arrays[scene+'__'+p+'__'+name].sum(0)
                    if (not np.array_equal(counts, entry['counts'])
                            or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p]['NoAdmission_Exact']['confusion_matrix'])
                            or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p][name]['confusion_matrix'])):
                        raise RuntimeError('Transition endpoints differ.')
    for index, image in enumerate(row['diagnostics'].values()):
        if (image['operator_replay_max_error'] > 1e-10 or max(image['local_replay_max_error'].values()) > 1e-4
                or image['observer336_replay_max_error'] != 0 or image['fine_coverage_max_error'] > 1e-6
                or not 1 <= image['fine_forwards'] <= 4 or image['fine_token_original_pixels'] != 8
                or not image['original_head_layout_unchanged']):
            raise RuntimeError('Fine observation/layout identity differs.')
        for protocols in image['scenarios'].values():
            for diag in protocols.values():
                if (diag['baseline_readout_replay_max_error'] > 1e-4 or diag['held_family_independence_max_error'] != 0
                        or diag['zero_semantic_identity_max_error'] not in (None, 0.)
                        or diag['zero_risk_writer_max_error'] != 0 or diag['extra_canonical_risk_max'] != 0
                        or max(diag['semantic_weight_spectrum_errors'].values()) != 0
                        or diag['spatial_risk_spectrum_error'] != 0
                        or any(s['capacity_excess_max'] > 1e-10 for s in diag['sources'].values())):
                    raise RuntimeError('Fine reference/source/writer invariant differs.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as old:
            for scene, protocols in row['metrics'].items():
                for p in protocols:
                    for name, earlier in REPLAY.items():
                        if not np.array_equal(current[scene+'__'+p+'__'+name+'__scores'], old[scene+'__'+p+'__'+earlier+'__scores']):
                            raise RuntimeError('Historical numerical endpoint differs.')
    row.update(coverage_verified=True, exact_previous_controls=REPLAY, fine_reference_config=asdict(CONFIG),
        source_fields_frozen_before_prediction_masks=True, semantic_source_text_only=True,
        original_observer_and_geometry_unchanged=True)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_fine_reference_admission.py',
        session_prefix='gfref03', implementation=IMPLEMENTATION, methods=METHODS,
        config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'fine_reference_config': asdict(CONFIG),
            'semantic_source': str(SEMANTIC), 'historical_prediction_reference': str(PREVIOUS),
            'original_geometry_writer_unchanged': True, 'only_reference_physical_observation_changed': True,
            'real_raw_llm_provenance_available': False})
