"""Run the predeclared ownership-source prediction screen on idle GPUs0-7."""
import argparse
from pathlib import Path

import numpy as np

from dinotool.native_ownership_reader import IMPLEMENTATION, CONFIG, METHODS, LEGACY, GATE, SPATIAL_SHUFFLES
from dinotool.native_class_ownership import CONFIG as SOURCE_CONFIG
from eval_native_ownership_reader import SOURCE, PREVIOUS_NOISE
from run_native_alias_noise import verify as verify_noise
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify_extension(root, dataset, row):
    previous = read_json(PREVIOUS_NOISE/dataset/'merged.json')
    source = read_json(SOURCE/dataset/'results.json')
    if (row['sample_keys'] != previous['sample_keys'] or row['signature'] != source['signature']
            or row['vocabularies'] != previous['vocabularies']):
        raise RuntimeError('Frozen ownership prediction inputs differ.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS_NOISE/dataset/'per_image_confusions.npz', allow_pickle=False) as old:
        for scene, protocols in row['metrics'].items():
            for p in protocols:
                for method in LEGACY:
                    key = scene+'__'+p+'__'+method
                    if not np.array_equal(current[key], old[key]):
                        raise RuntimeError('Previous17 per-image endpoints differ.')
    for image in row['diagnostics'].values():
        for protocols in image['scenarios'].values():
            for stats in protocols.values():
                s = stats['ownership']
                if (max(s['legacy_score_replay_max_errors'].values()) != 0
                        or s['attachment_risk_replay_max_error'] != 0 or s['zero_risk_writer_max'] != 0):
                    raise RuntimeError('Frozen source/unchanged writer replay differs.')
                for method, v in {**s['sources'], **s['controls']}.items():
                    if (max(v['normal_equation_max_error'], v['gauge_max_error']) > 1e-10
                            or v['capacity_excess_max'] > 1e-6):
                        raise RuntimeError('Writer/consistency constraints differ.')
                    if method in s['controls'] and (v['pair_budget_max_error'] > 1e-10
                            or method in SPATIAL_SHUFFLES and v['pair_spectrum_max_error'] > 1e-12):
                        raise RuntimeError('Matched action controls differ.')
    row['exact_previous_all_regime_controls'] = list(LEGACY)
    row['source_fields_frozen_before_prediction_masks'] = True


def verify(root, dataset):
    verify_noise(root, dataset, IMPLEMENTATION, METHODS, verify_extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not read_json(SOURCE/'suite_results.json')['decision']['passed']:
        raise RuntimeError('Source gate must pass before pixel predictions.')
    run(args.root, evaluator='scripts/eval_native_ownership_reader.py', session_prefix='gnor03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'ownership_source': str(SOURCE),
            'ownership_source_config': vars(SOURCE_CONFIG), 'historical_noise_reference': str(PREVIOUS_NOISE),
            'unchanged_excess_writer': True, 'real_raw_llm_provenance_available': False})
