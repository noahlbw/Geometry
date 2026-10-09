"""Verify and run nested context-size evaluation on idle A800 GPUs0-7."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from dinotool.fine_alias_view import CONFIG
from dinotool.rival_alias_count import IMPLEMENTATION, COUNTS, METHODS
from eval_rival_alias_count import PREVIOUS, VOCAB, HISTORY
from run_bounded_alias_suite import save
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify(root, dataset):
    row, prior = read_json(root/dataset/'results.json'), read_json(PREVIOUS/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or len(set(row['sample_keys'])) != 8
            or row['base_signature'] != prior['signature'] or row['implementation'] != IMPLEMENTATION
            or row['config'] != asdict(CONFIG) or row['local_geometry_alias_count'] != 20
            or row['context_alias_counts'] != list(COUNTS) or not row['weights_frozen'] or not row['head_weights_unchanged']
            or not row['target_masks_loaded'] or not row['numerical_caches_precede_masks']
            or row['vocabulary_source_sha256'] != hashlib.sha256(VOCAB.read_bytes()).hexdigest()):
        raise RuntimeError('Changed complete variable-count evaluation.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        if current['sample_keys'].tolist() != row['sample_keys']:
            raise RuntimeError('Changed sample order.')
        for scene, protocols in row['metrics'].items():
            for p, arms in protocols.items():
                if set(arms) != set(METHODS):
                    raise RuntimeError('Missing count-study endpoint.')
                for name, metric in arms.items():
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Confusion reconstruction differs.')
                    if scene == 'k20' and not np.array_equal(current[key], old['clean__'+p+'__'+HISTORY[name]]):
                        raise RuntimeError('Historical20 image endpoint differs.')
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as current:
        for scene, protocols in row['transitions'].items():
            for p, entry in protocols.items():
                counts = current[scene+'__'+p].sum(0)
                if (not np.array_equal(counts, entry['counts'])
                        or not np.array_equal(counts.sum(1).T, row['metrics'][scene][p]['NoAdmission_Exact']['confusion_matrix'])
                        or not np.array_equal(counts.sum(0).T, row['metrics'][scene][p]['RivalFineHard_Exact']['confusion_matrix'])):
                    raise RuntimeError('Transition reconstruction differs.')
    for diag in row['diagnostics'].values():
        if diag['operator_replay_max_error'] > 1e-10 or max(diag['local_replay_max_error'].values()) > 1e-4:
            raise RuntimeError('Original Geometry changed.')
        for scene, protocols in diag['scenarios'].items():
            for detail in protocols.values():
                if detail['random_count_max_error'] or detail['canonical_risk_max']:
                    raise RuntimeError('Matched deletion/canonical protection changed.')
                if scene == 'k20' and max(detail['historical20_replay_max_errors'].values()) > 1e-4:
                    raise RuntimeError('Fresh historical20 score replay failed.')
    row.update(coverage_verified=True, exact_historical20_per_image_replay=True,
        original_geometry_and_rule_unchanged=True)
    save(root/dataset/'merged.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    source = read_json(VOCAB)
    if not source or source['status'] != 'complete' or source['target_masks_loaded'] or source['target_images_loaded']:
        raise RuntimeError('Complete label/image-free Qwen extensions required.')
    run(parser.parse_args().root, evaluator='scripts/eval_rival_alias_count.py', session_prefix='grac03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'context_alias_counts': list(COUNTS), 'local_geometry_alias_count': 20,
            'vocabulary_source': str(VOCAB), 'vocabulary_source_sha256': hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
            'nested_historical20_prefix': True, 'all_model_rules_frozen': True,
            'source_has_images_or_labels': False, 'source_has_semantic_or_visual_filtering': False})
