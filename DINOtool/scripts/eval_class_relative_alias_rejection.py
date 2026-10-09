"""Matched no-absolute-gain successor using the existing complete-image evaluator."""
import sys

from dinotool.class_relative_alias_rejection import IMPLEMENTATION, METHODS, PRIMARY, ClassRelativeAliasReader
from eval_excess_alias_rejection import main, smoke
from eval_geometry_semantic_innovation import parse_args


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
        smoke(parse_args(), ClassRelativeAliasReader, IMPLEMENTATION, PRIMARY)
    else:
        main(parse_args(), ClassRelativeAliasReader, IMPLEMENTATION, METHODS)
