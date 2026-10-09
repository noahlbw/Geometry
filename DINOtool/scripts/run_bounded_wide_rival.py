import argparse
from pathlib import Path
import run_evidence_envelope as runner
from eval_bounded_wide_rival import METHODS

if __name__=='__main__':
    p=argparse.ArgumentParser(description='One bounded-rival rule on idle GPUs.');p.add_argument('--root',required=True,type=Path)
    runner.EVALUATOR='scripts/eval_bounded_wide_rival.py';runner.PREFIX='gbwr07';runner.METHODS=METHODS
    runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=runner.verify_job;runner.queue.verify_dataset=runner.verify_dataset
    runner.queue.main(p.parse_args().root)
