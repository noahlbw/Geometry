"""Queue the fixed paired consistency hypothesis only on idle GPUs."""
import argparse
from pathlib import Path
import run_evidence_envelope as runner
from eval_geometry_consistency_gain import METHODS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    runner.EVALUATOR='scripts/eval_geometry_consistency_gain.py';runner.PREFIX='ggcg07';runner.METHODS=METHODS
    runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=runner.verify_job;runner.queue.verify_dataset=runner.verify_dataset
    runner.queue.main(p.parse_args().root)
