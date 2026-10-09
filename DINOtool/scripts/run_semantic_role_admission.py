"""Frozen semantic-role prediction gate on idle authorized GPUs0-7."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dinotool.semantic_role_admission import IMPLEMENTATION, CONFIG, METHODS, LEGACY, GATE, ALIAS_SHUFFLES
from dinotool.native_query_alias import CONFIG as OBSERVATION_CONFIG
from eval_semantic_role_admission import PREVIOUS, SEMANTIC
from run_native_alias_noise import verify as verify_noise
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify_extension(root, dataset, row):
    prior, source = read_json(PREVIOUS/dataset/'merged.json'), read_json(SEMANTIC/(dataset+'.json'))
    if (row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature']
            or row['vocabularies'] != prior['vocabularies'] or not source['sentinel_contract_passed']):
        raise RuntimeError('Frozen source or prior inputs changed.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        for scene, protocols in row['metrics'].items():
            for protocol in protocols:
                for name in LEGACY:
                    key = scene+'__'+protocol+'__'+name
                    if not np.array_equal(current[key], old[key]):
                        raise RuntimeError('Historical61 per-image endpoint changed.')
    for image in row['diagnostics'].values():
        for protocols in image['scenarios'].values():
            for stats in protocols.values():
                diag = stats['semantic_role']
                if (max(diag['legacy61_score_replay_max_errors'].values()) != 0
                        or not diag['source_score_replay_exact'] or not diag['no_per_image_language_forwards']):
                    raise RuntimeError('Historical prediction/source identity failed.')
                for name, entry in diag['sources'].items():
                    if (entry['capacity_excess_max'] > 1e-10
                            or name in ALIAS_SHUFFLES and entry['extra_risk_spectrum_max_error'] > 1e-8):
                        raise RuntimeError('Writer bounds/semantic spectra failed.')
    row['exact_previous_all_regime_controls'] = list(LEGACY)
    row['source_fields_frozen_before_prediction_masks'] = True
    row['admission_config'] = asdict(CONFIG)
    row['semantic_source_text_only'] = True


def verify(root, dataset):
    verify_noise(root, dataset, IMPLEMENTATION, METHODS, verify_extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    if read_json(SEMANTIC/'suite_results.json')['status'] != 'complete':
        raise RuntimeError('Mask-free source contract must pass before evaluation.')
    run(parser.parse_args().root, evaluator='scripts/eval_semantic_role_admission.py',
        session_prefix='gsra03', implementation=IMPLEMENTATION, methods=METHODS,
        config=OBSERVATION_CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'admission_config': asdict(CONFIG),
            'semantic_source': str(SEMANTIC), 'historical_prediction_reference': str(PREVIOUS),
            'original_geometry_writer_unchanged': True, 'text_role_source_frozen_before_prediction': True,
            'real_raw_llm_provenance_available': False})
