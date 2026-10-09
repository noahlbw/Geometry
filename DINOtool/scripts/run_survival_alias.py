"""Run a frozen same-source soft-survival writer comparison on GPUs0-7."""
import argparse
from pathlib import Path

import numpy as np

from dinotool.survival_alias import IMPLEMENTATION, CONFIG, METHODS, LEGACY, SCENARIOS, STYLE_FILES, GATE, SPATIAL
from eval_survival_alias import PREVIOUS_NOISE
from run_native_alias_noise import verify as verify_noise
from run_pair_context_reader import run
from run_sat_geometry_transport_suite import read_json


def verify_extension(root, dataset, row):
    prior = read_json(PREVIOUS_NOISE/dataset/'merged.json')
    if (row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature']
            or row['vocabularies'] != prior['vocabularies']):
        raise RuntimeError('Changed source vocabulary or sequence.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            PREVIOUS_NOISE/dataset/'per_image_confusions.npz', allow_pickle=False) as previous:
        for scene, protocols in row['metrics'].items():
            for p in protocols:
                for method in LEGACY:
                    key = scene+'__'+p+'__'+method
                    if not np.array_equal(current[key], previous[key]):
                        raise RuntimeError('Previous17 per-image prediction endpoints differ.')
    for image in row['diagnostics'].values():
        for protocols in image['scenarios'].values():
            for stats in protocols.values():
                s = stats['survival']
                if (max(s['legacy_score_replay_max_errors'].values()) != 0 or s['risk_replay_max_error'] != 0
                        or s['zero_risk_writer_max'] != 0 or s['binary_hard_score_max_error'] > 1e-10):
                    raise RuntimeError('Writer-only contract differs.')
                for name, info in {**s['sources'], **s['controls']}.items():
                    if max(info['normal_equation_max_error'], info['gauge_max_error']) > 1e-10:
                        raise RuntimeError('Unconverged signed consistency.')
                    if name in s['controls'] and (info['pair_budget_max_error'] > 1e-10
                            or name in SPATIAL and (info['pair_spectrum_max_error'] != 0 or info['pair_absolute_budget_max_error'] > 1e-10)):
                        raise RuntimeError('Signed control budgets differ.')
    row['exact_previous_all_regime_controls'] = list(LEGACY)
    row['same_native_risk_verified'] = True


def verify(root, dataset):
    verify_noise(root, dataset, IMPLEMENTATION, METHODS, verify_extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, evaluator='scripts/eval_survival_alias.py', session_prefix='gsur03',
        implementation=IMPLEMENTATION, methods=METHODS, config=CONFIG, verifier=verify,
        protocol_extra={'promotion_gate': GATE, 'scenarios': SCENARIOS, 'style_files': STYLE_FILES,
            'same_source_reference': str(PREVIOUS_NOISE), 'real_raw_llm_provenance_available': False,
            'signed_shuffle_preserves_spectrum_not_alias_feasibility': True})
