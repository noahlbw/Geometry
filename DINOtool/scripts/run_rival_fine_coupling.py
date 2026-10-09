"""Evaluate the frozen pair-preserving candidate with count-matched controls."""
import argparse
from pathlib import Path

import numpy as np

from dinotool.rival_fine_coupling import IMPLEMENTATION, METHODS, REPLAY, GATE
from dinotool.fine_alias_view import CONFIG
from eval_rival_fine_coupling import PREVIOUS
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['signature'] != prior['signature'] or row['vocabularies'] != prior['vocabularies']
            or row['implementation'] != IMPLEMENTATION or not row['weights_frozen'] or not row['head_weights_unchanged']
            or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Invalid complete unchanged cached-source evaluation.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        for scene, protocols in row['metrics'].items():
            for p, arms in protocols.items():
                if set(arms) != set(METHODS):
                    raise RuntimeError('Missing prediction endpoint.')
                for name, metric in arms.items():
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Confusion reconstruction differs.')
                    if name in REPLAY and not np.array_equal(current[key], old[key]):
                        raise RuntimeError('Source per-image endpoint differs.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as current:
        for scene, protocols in row['transitions'].items():
            for p, arms in protocols.items():
                for name, entry in arms.items():
                    counts = current[scene+'__'+p+'__'+name].sum(0)
                    if (not np.array_equal(counts, entry['counts'])
                            or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p]['NoAdmission_Exact']['confusion_matrix'])
                            or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p][name]['confusion_matrix'])):
                        raise RuntimeError('Transition endpoints differ.')
    for index, diag in enumerate(row['diagnostics'].values()):
        if diag['new_image_observation_forwards'] != 0 or not diag['source_risks_and_predictions_precede_masks']:
            raise RuntimeError('Changed frozen observations.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as old:
            for scene, protocols in diag['scenarios'].items():
                for p, detail in protocols.items():
                    if detail['zero_risk_identity_max_error'] != 0:
                        raise RuntimeError('Original zero-risk coupling changed.')
                    for name in REPLAY:
                        key = scene+'__'+p+'__'+name+'__scores'
                        if not np.array_equal(current[key], old[key]):
                            raise RuntimeError('Source numerical scores differ.')
    row.update(coverage_verified=True, exact_previous_controls=list(REPLAY),
        original_geometry_writer_unchanged=True, source_fields_frozen_before_prediction_masks=True)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_rival_fine_coupling.py',
        session_prefix='grfc03', implementation=IMPLEMENTATION, methods=METHODS,
        config=CONFIG, verifier=verify, protocol_extra={'promotion_gate': GATE,
            'historical_prediction_reference': str(PREVIOUS), 'source_observations_cached': True,
            'only_pair_preserving_geometry_writeback_changed': True})
