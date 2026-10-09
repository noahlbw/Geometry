"""Run one CPU-only full analysis after the existing evaluation/cost pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from report_alias_finalization_statistics import main as analyze


MAIN_SESSION = 'gaf09_controller_verification_loader_queue'
COST_SESSION = 'gaf09_controller_natural_vip_cost'


def alive(session):
    return subprocess.run(['tmux', 'has-session', '-t', session],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run(root,main_result='verification_results.json',main_session=MAIN_SESSION,
        cost_result='natural_vip_cost_results.json',cost_session=COST_SESSION):
    output = root / 'statistics_full'
    if output.exists():
        raise RuntimeError('Existing full analysis output; preserve it.')
    # Timing retains exclusive pipeline access: analysis starts only after both
    # existing finite stages complete, avoiding CPU/I/O interference in timing.
    stages = ((main_result, main_session), (cost_result, cost_session))
    while True:
        ready = True
        for filename, session in stages:
            result = read(root / filename)
            if result is not None:
                if result['status'] != 'complete':
                    raise RuntimeError('Prerequisite stage failed: ' + filename)
            else:
                ready = False
                if not alive(session):
                    # A short final-save boundary is observed twice before it
                    # is treated as missing; observation never relaunches work.
                    time.sleep(2)
                    result = read(root / filename)
                    if result is None and not alive(session):
                        raise RuntimeError('Prerequisite handle missing without terminal result: ' + session)
        if ready:
            break
        time.sleep(15)
    analyze(root, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--main-result',default='verification_results.json')
    parser.add_argument('--main-session',default=MAIN_SESSION)
    parser.add_argument('--cost-result',default='natural_vip_cost_results.json')
    parser.add_argument('--cost-session',default=COST_SESSION)
    args=parser.parse_args()
    run(args.root,args.main_result,args.main_session,args.cost_result,args.cost_session)
