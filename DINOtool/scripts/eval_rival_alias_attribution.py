"""Exact-source replay plus frozen alias/location/rival attribution controls."""
import argparse
import json
import sys
from types import FunctionType

import numpy as np
import torch

from dinotool.rival_alias_attribution import IMPLEMENTATION, METHODS, PRIMARY, settings, attribution_scores
import eval_rival_competition_admission as source


PREVIOUS = source.TOOL/'results/rival_competition_admission_20261005'
FROZEN_CHECKS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', 'FineBudgetOnly_Exact')


@torch.inference_mode()
def tile(*args, methods=METHODS, **kwargs):
    original = source.reference.predict_tile.__wrapped__
    def reader(*inputs): return attribution_scores(*inputs, methods=methods)
    scope = dict(original.__globals__, retained_scores=reader)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


@torch.inference_mode()
def window(*args):
    original = source.window.__wrapped__
    scope = dict(original.__globals__, tile=tile)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args)


def screen(args, benchmark_alternatives=(), check_singletons=False):
    previous = json.loads((PREVIOUS/args.dataset/'results.json').read_text())
    historical = json.loads((source.SOURCE/args.dataset/'results.json').read_text())
    if previous['status'] != 'complete' or previous['sample_keys'] != historical['sample_keys']:
        raise RuntimeError('Complete fixed-candidate source required.')
    index = 0
    singleton_methods = (PRIMARY, *benchmark_alternatives)
    first_scores, singleton_checks = {}, dict.fromkeys(singleton_methods, 0)
    def checked_window(*inputs):
        nonlocal index
        result = window(*inputs)
        if inputs[-1] == METHODS:
            if check_singletons and index == 0:
                first_scores.update({p: {m: g[m].cpu().numpy().copy() for m in singleton_methods}
                                     for p, g in result[1].items()})
            with np.load(PREVIOUS/args.dataset/'scores'/f'{index}.npz', allow_pickle=False) as old:
                for p, group in result[1].items():
                    for m in FROZEN_CHECKS:
                        if not np.array_equal(group[m].cpu().numpy(), old[p+'__'+m]):
                            raise RuntimeError('Frozen candidate score differs: '+p+'/'+m)
            index += 1
        elif check_singletons and len(inputs[-1]) == 1 and inputs[-1][0] in singleton_methods:
            method = inputs[-1][0]
            for p, group in result[1].items():
                if not np.array_equal(group[method].cpu().numpy(), first_scores[p][method]):
                    raise RuntimeError('Standalone candidate score differs: '+p+'/'+method)
            singleton_checks[method] += 1
        return result
    def save(path, result):
        result['frozen_candidate_scores_exact'] = True
        if check_singletons and result['status'] == 'complete':
            if any(count != 4 for count in singleton_checks.values()):
                raise RuntimeError('Expected warmup plus three standalone score checks per candidate.')
            result['standalone_candidate_scores_exact'] = True
            result['standalone_candidate_checks'] = singleton_checks
        source.reference.save(path, result)
    original = source.screen.__wrapped__
    scope = dict(original.__globals__, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS,
        PRIMARY=PRIMARY, ALTERNATIVES=benchmark_alternatives, settings=settings, window=checked_window)
    # Keep the source module untouched; only this isolated function sees the extra checks.
    proxy = type('Reference', (), {'frozen_state': staticmethod(source.reference.frozen_state),
        'check_frozen': staticmethod(source.reference.check_frozen), 'save': staticmethod(save)})
    scope['reference'] = proxy
    with torch.inference_mode():
        FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(args)
    if index != 8: raise RuntimeError('Candidate replay did not cover all eight source windows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    screen(parser.parse_args())
