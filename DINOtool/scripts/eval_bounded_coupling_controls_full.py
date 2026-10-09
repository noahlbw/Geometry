"""Full-population entry point; the diagnosed source and score rules stay fixed."""
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    sys.argv.append('--full-domain')
    runpy.run_path(str(Path(__file__).with_name('eval_bounded_coupling_controls.py')), run_name='__main__')
