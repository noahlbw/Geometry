"""Frozen coherent-admission prediction gate on the authorized idle GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.coherent_native_admission import (IMPLEMENTATION, CONFIG, METHODS, LEGACY,
    GATE, ALIAS_SHUFFLES, ACTION_SHUFFLES)
from dinotool.native_query_alias import CONFIG as OBSERVATION_CONFIG
from dinotool.native_class_ownership import CONFIG as SOURCE_CONFIG
from eval_coherent_native_admission import PREVIOUS, SOURCE
from run_native_alias_noise import verify as verify_noise
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify_extension(root, dataset, row):
    prior = read_json(PREVIOUS/dataset/'merged.json')
    if row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature'] or row['vocabularies'] != prior['vocabularies']:
        raise RuntimeError('Historical33 prediction inputs changed.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        for scene, protocols in row['metrics'].items():
            for p in protocols:
                for name in LEGACY:
                    key = scene+'__'+p+'__'+name
                    if not np.array_equal(current[key], old[key]):
                        raise RuntimeError('Historical33 per-image endpoints changed: '+key)
    for image in row['diagnostics'].values():
        for protocols in image['scenarios'].values():
            for stats in protocols.values():
                diag = stats['coherent']
                if max(diag['legacy33_score_replay_max_errors'].values()) != 0 or diag['zero_risk_action_max'] != 0:
                    raise RuntimeError('Numerical replay/identity failed.')
                for name, entry in diag['sources'].items():
                    if entry['capacity_excess_max'] > 1e-10 or name in ALIAS_SHUFFLES and entry['alias_spectrum_max_error'] != 0:
                        raise RuntimeError('Alias allocation spectrum/bounds failed.')
                for name, entry in diag['controls'].items():
                    if entry['class_budget_max_error'] > 1e-10 or name in ACTION_SHUFFLES and entry['class_spectrum_max_error'] != 0:
                        raise RuntimeError('Matched action budget/spectrum failed.')
    row['exact_previous_all_regime_controls'] = list(LEGACY)
    row['source_fields_frozen_before_prediction_masks'] = True
    row['admission_config'] = asdict(CONFIG)


def verify(root, dataset):
    verify_noise(root, dataset, IMPLEMENTATION, METHODS, verify_extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, evaluator='scripts/eval_coherent_native_admission.py', session_prefix='gcn03',
        implementation=IMPLEMENTATION, methods=METHODS, config=OBSERVATION_CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'admission_config': asdict(CONFIG),
            'ownership_source': str(SOURCE), 'ownership_source_config': asdict(SOURCE_CONFIG),
            'historical_prediction_reference': str(PREVIOUS), 'original_geometry_writer_unchanged': True,
            'coherent_class_action': True, 'canonical_protected_in_primary': False,
            'real_raw_llm_provenance_available': False})
