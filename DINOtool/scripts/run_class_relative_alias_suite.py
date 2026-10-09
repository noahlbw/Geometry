"""Frozen no-absolute-gain successor with exact clean and stress history replay."""
import argparse
from pathlib import Path

from dinotool.class_relative_alias_rejection import IMPLEMENTATION, METHODS, PRIMARY, REPLAY, SHUFFLED
from run_excess_alias_suite import run
from run_region_semantic_suite_a800 import TOOL


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root, source=TOOL / 'results/excess_alias_rejection_screen_20261003',
        implementation=IMPLEMENTATION, methods=METHODS, primary=PRIMARY, shuffled=SHUFFLED,
        replay=REPLAY, replay_stress=True, evaluator='scripts/eval_class_relative_alias_rejection.py',
        session_prefix='gcrar03')
