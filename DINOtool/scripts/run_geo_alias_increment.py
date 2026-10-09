"""Run the frozen canonical-witness pilot; verify every archived control."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.geo_alias_increment import CONFIG, COUNTS, IMPLEMENTATION, METHODS
from eval_geo_alias_increment import PAIRS, PREVIOUS, VOCAB
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def replay_source(scene, method):
    if method == 'Geometry':
        return scene, 'Geometry'
    if method == 'AllCount_Exact':
        return scene, 'NoAdmission_Exact'
    if method == 'RivalFineHard_Exact':
        return scene, method
    return 'k20', 'NoAdmission_Exact' if method == 'Fixed20_Exact' else 'RivalFineHard_Exact'


def verify(root, dataset):
    row, old = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != old['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['base_signature'] != old['base_signature'] or row['implementation'] != IMPLEMENTATION
            or row['config'] != asdict(CONFIG) or row['context_alias_counts'] != list(COUNTS)
            or row['context_vocabularies'] != old['context_vocabularies']
            or row['vocabulary_source_sha256'] != old['vocabulary_source_sha256']
            or not row['weights_frozen'] or not row['head_weights_unchanged']
            or not row['target_masks_loaded'] or not row['numerical_caches_precede_masks']):
        raise RuntimeError('Incomplete or changed canonical-witness protocol.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        if current['sample_keys'].tolist() != row['sample_keys']:
            raise RuntimeError('Changed image sequence.')
        for scene, ps in row['metrics'].items():
            for p, methods in ps.items():
                if set(methods) != set(METHODS):
                    raise RuntimeError('Missing declared endpoint.')
                for method, metric in methods.items():
                    value = current[scene+'__'+p+'__'+method]
                    if not np.array_equal(value.sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Confusion reconstruction differs.')
                    if method in METHODS[:5]:
                        old_scene, old_method = replay_source(scene, method)
                        if not np.array_equal(value, previous[old_scene+'__'+p+'__'+old_method]):
                            raise RuntimeError('Archived endpoint changed.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as current:
        for scene, ps in row['transitions'].items():
            for p, pairs in ps.items():
                for pair, entry in pairs.items():
                    counts = current[scene+'__'+p+'__'+pair].sum(0)
                    before, after = PAIRS[pair]
                    if (not np.array_equal(counts, entry['counts'])
                            or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p][before]['confusion_matrix'])
                            or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p][after]['confusion_matrix'])):
                        raise RuntimeError('Transition endpoints differ.')
    for diag in row['diagnostics'].values():
        if (diag['operator_replay_max_error'] > 1e-10 or max(diag['local_replay_max_error'].values()) > 1e-4
                or max(diag['base_wide_replay_max_error'].values()) > 1e-4):
            raise RuntimeError('Original Geometry or fixed20 anchor changed.')
        for scene, ps in diag['scenarios'].items():
            for entry in ps.values():
                if entry['weight_shuffle_spectrum_error'] or (scene == 'k20' and entry['uniform20_increment_identity_max_error']):
                    raise RuntimeError('Matched writer control failed.')
                if any(max(v['normal_equation_max_error'], v['gauge_max_error']) > 1e-10 for v in entry['sources'].values()):
                    raise RuntimeError('Competitive potential failed.')
    row.update(coverage_verified=True, archived_controls_verified=True, original_geometry_and_anchor_unchanged=True)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    source = read_json(VOCAB)
    if not source or source['status'] != 'complete' or source['target_masks_loaded'] or source['target_images_loaded']:
        raise RuntimeError('Frozen text-only source required.')
    run(parser.parse_args().root, evaluator='scripts/eval_geo_alias_increment.py', session_prefix='ggai03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'context_alias_counts': list(COUNTS), 'local_geometry_alias_count': 20,
            'anchor': 'original unscreened20 coupling', 'source_count_root': str(PREVIOUS),
            'vocabulary_source': str(VOCAB), 'no_count_dependent_old_profile': True,
            'canonical_only_witness_selection': True, 'writer': 'bounded maximum excess increment'})
