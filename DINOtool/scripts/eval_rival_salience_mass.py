"""Frozen developed64 replay for retained observer salience-mass normalization."""
import argparse
from types import FunctionType

import torch

from dinotool.rival_salience_mass import (IMPLEMENTATION, METHODS, PRIMARY,
    CANDIDATES, settings, salience_mass_scores)
import eval_rival_alias_attribution as audit


@torch.inference_mode()
def tile(*args, methods=METHODS, **kwargs):
    original = audit.source.reference.predict_tile.__wrapped__
    def reader(*inputs): return salience_mass_scores(*inputs, methods=methods)
    scope = dict(original.__globals__, retained_scores=reader)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


@torch.inference_mode()
def window(*args):
    original = audit.source.window.__wrapped__
    scope = dict(original.__globals__, tile=tile)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    original = audit.screen
    scope = dict(original.__globals__, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS,
                 PRIMARY=PRIMARY, settings=settings, window=window)
    FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(parser.parse_args(), CANDIDATES[1:], True)
