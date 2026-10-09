"""Run and verify one frozen native admission vocabulary screen on idle GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.native_alias_noise import (IMPLEMENTATION, CONFIG, METHODS, PRIMARY, SCENARIOS,
    STYLE_FILES, ALIAS_SHUFFLES, RANDOM_DELETIONS, SPATIAL_SHUFFLES, GATE)
from eval_native_alias_noise import HISTORY, PREVIOUS
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset, implementation=IMPLEMENTATION, methods=METHODS, extension_verifier=None):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    expected = list(SCENARIOS)+(['llm_style'] if dataset in STYLE_FILES else [])
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['signature'] != prior['signature'] or row['implementation'] != implementation
            or row['config'] != asdict(CONFIG) or row['scenarios'] != expected
            or not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_loaded']
            or not row['numerical_caches_precede_masks'] or not row['local_anchor_unchanged']):
        raise RuntimeError('Changed coverage, weights or frozen protocol.')
    vocabularies = read_json(root/dataset/'vocabularies.json')
    if vocabularies != row['vocabularies'] or set(vocabularies) != set(expected):
        raise RuntimeError('Changed pre-image vocabulary manifest.')
    for scene, entry in vocabularies.items():
        for p, classes in entry['vocabularies'].items():
            if ([c['name'] for c in classes] != prior['signature']['classes'][p]
                    or any(len(c['synonyms']) != 20 or len(set(c['synonyms'])) != 20 or c['name'] not in c['synonyms'] for c in classes)):
                raise RuntimeError('Changed20-count/classes/canonical contract.')
            if scene == 'clean' and [a for c in classes for a in c['synonyms']] != prior['signature']['vocabulary']['aliases'][p]:
                raise RuntimeError('Changed clean vocabulary.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        if current['sample_keys'].tolist() != prior['sample_keys']:
            raise RuntimeError('Changed sample order.')
        for scene, protocols in row['metrics'].items():
            for p, group in protocols.items():
                if set(group) != set(methods):
                    raise RuntimeError('Missing robustness arms.')
                for m, metric in group.items():
                    values = current[scene+'__'+p+'__'+m]
                    if not np.array_equal(values.sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Changed confusion endpoint.')
                    if m == 'Geometry' and not np.array_equal(values, previous[p+'__Geometry']):
                        raise RuntimeError('Scenario changed local anchor predictions.')
                    if scene == 'clean' and m in HISTORY and not np.array_equal(values, previous[p+'__'+HISTORY[m]]):
                        raise RuntimeError('Changed historical clean prediction.')
                    if scene == 'llm_style' and vocabularies[scene]['identical_to_clean'] and not np.array_equal(values, current['clean__'+p+'__'+m]):
                        raise RuntimeError('Style identity control differs.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for scene, protocols in row['transitions'].items():
            for p, group in protocols.items():
                for m, entry in group.items():
                    counts = arrays[scene+'__'+p+'__'+m].sum(0)
                    if (not np.array_equal(counts, entry['counts'])
                            or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p]['NoAdmission_Exact']['confusion_matrix'])
                            or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p][m]['confusion_matrix'])):
                        raise RuntimeError('Changed dense transition endpoint.')
    for index, image in enumerate(row['diagnostics'].values()):
        if (image['operator_replay_max_error'] > 1e-10 or image['new_intervention_forwards'] != 0
                or image['native_coverage_max_error'] > 1e-6 or not 1 <= image['native_forwards'] <= 4
                or image['native_token_original_pixels'] != 16 or max(image['local_replay_max_error'].values()) > 1e-4):
            raise RuntimeError('Changed Geometry or physical observation.')
        with np.load(root/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as current, np.load(
                PREVIOUS/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as previous:
            for p in row['metrics']['clean']:
                for m, old in HISTORY.items():
                    if not np.array_equal(current['clean__'+p+'__'+m+'__scores'], previous[p+'__'+old+'__scores']):
                        raise RuntimeError('Historical clean numerical endpoint differs.')
        for scene, protocols in image['scenarios'].items():
            for p, stats in protocols.items():
                if stats['profile_replay_max_error'] > 1e-4:
                    raise RuntimeError('Profiled observation path replay differs.')
                if scene == 'clean' and (max(stats['clean_score_replay_max_errors'].values()) > 1e-4
                        or max(stats['clean_source_replay_max_errors'].values()) > 1e-4):
                    raise RuntimeError('Historical source path replay differs.')
                for m, s in stats['sources'].items():
                    if (max(s['normal_equation_max_error'], s['gauge_max_error']) > 1e-10
                            or s['capacity_excess_max'] > 1e-6 or s['canonical_risk_max'] != 0
                            or m in ALIAS_SHUFFLES and s['risk_spectrum_max_error'] != 0):
                        raise RuntimeError('Soft source/writer constraints differ.')
                for s in stats['hard'].values():
                    if max(s['normal_equation_max_error'], s['gauge_max_error']) > 1e-10 or s['count_max_error'] != 0 or s['canonical_risk_max'] != 0:
                        raise RuntimeError('Hard/count-matched constraints differ.')
                for m, s in stats['controls'].items():
                    if (s['pair_budget_max_error'] > 1e-10 or s['capacity_excess_max'] > 1e-6
                            or m in SPATIAL_SHUFFLES and s['pair_spectrum_max_error'] > 1e-12):
                        raise RuntimeError('Directional budget/spectrum controls differ.')
    if extension_verifier is not None:
        extension_verifier(root, dataset, row)
    row['coverage_verified'] = True
    row['exact_historical_clean_controls'] = HISTORY
    row['exact_local_anchor_all_scenarios'] = True
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_native_alias_noise.py', session_prefix='gnan03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'scenarios': SCENARIOS, 'style_files': STYLE_FILES,
            'historical_clean_controls': HISTORY, 'witness_source': 'native-query-only frozen from prior declared control',
            'real_raw_llm_provenance_available': False})
