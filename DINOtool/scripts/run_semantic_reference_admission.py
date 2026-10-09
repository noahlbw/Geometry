"""Frozen semantic-reference gate on idle physical GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.semantic_reference_admission import IMPLEMENTATION, CONFIG, METHODS, LEGACY, GATE
from dinotool.native_query_alias import CONFIG as OBSERVATION_CONFIG
from eval_semantic_reference_admission import PREVIOUS, SEMANTIC
from run_native_alias_noise import verify as verify_noise
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify_extension(root, dataset, row):
    prior = read_json(PREVIOUS/dataset/'merged.json')
    if row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature'] or row['vocabularies'] != prior['vocabularies']:
        raise RuntimeError('Historical69 inputs changed.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        for scene, protocols in row['metrics'].items():
            for p in protocols:
                for name in LEGACY:
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key], old[key]):
                        raise RuntimeError('Historical69 per-image endpoint changed.')
    for image in row['diagnostics'].values():
        for protocols in image['scenarios'].values():
            for stats in protocols.values():
                diag = stats['semantic_reference']
                if (max(diag['legacy69_score_replay_max_errors'].values()) != 0
                        or diag['neutral_held_reference_replay_max_error'] > 1e-12
                        or diag['old_base_replay_max_error'] != 0
                        or diag['held_family_independence_max_error'] != 0
                        or diag['zero_semantic_identity_max_error'] not in (None, 0.)
                        or diag['extra_canonical_risk_max'] != 0
                        or max(diag['semantic_weight_spectrum_errors'].values()) != 0
                        or any(s['capacity_excess_max'] > 1e-10 for s in diag['sources'].values())):
                    raise RuntimeError('Weighted reference, protected extra role or writer identity failed.')
    row['exact_previous_all_regime_controls'] = list(LEGACY)
    row['source_fields_frozen_before_prediction_masks'] = True
    row['admission_config'] = asdict(CONFIG)
    row['semantic_source_text_only'] = True
    row['original_observer_and_geometry_unchanged'] = True


def verify(root, dataset):
    verify_noise(root, dataset, IMPLEMENTATION, METHODS, verify_extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_semantic_reference_admission.py',
        session_prefix='gsrf03', implementation=IMPLEMENTATION, methods=METHODS,
        config=OBSERVATION_CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'admission_config': asdict(CONFIG),
            'semantic_source': str(SEMANTIC), 'historical_prediction_reference': str(PREVIOUS),
            'original_geometry_writer_unchanged': True, 'weighted_reference_only_source_change': True,
            'real_raw_llm_provenance_available': False})
