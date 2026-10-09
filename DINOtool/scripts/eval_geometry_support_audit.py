"""Enable Geometry-relation diagnostic, without changing prediction rules."""
import runpy
import sys

if __name__=='__main__':
    sys.argv.append('--geometry-support')
    runpy.run_module('eval_correction_reliability_audit',run_name='__main__')
