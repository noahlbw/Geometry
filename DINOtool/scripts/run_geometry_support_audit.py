import argparse
from pathlib import Path
import run_correction_reliability_audit as runner

runner.launcher.EVALUATOR='scripts/eval_geometry_support_audit.py'
runner.launcher.PREFIX='ggsa07'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    runner.main(p.parse_args().root)
