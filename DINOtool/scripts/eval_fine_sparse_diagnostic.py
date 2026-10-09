"""Same developed64 windows, fixed fine evidence; restrict only pair actions."""
import argparse

import torch

from dinotool.fine_sparse_diagnostic import IMPLEMENTATION, METHODS, PRIMARY, diagnostic_scores
from eval_rival_fine_graph import bind
import eval_rival_competition_admission as reference


def settings():
    return dict(scope='diagnostic only; dense reader and four fine observations retained',
                only_change='zero non-top2 pair actions before the original posterior solver',
                contender_source='stable top2 unchanged baseline posterior',
                fine_evidence_alias_support_allocation_projection_and_full_posterior_unchanged=True,
                target_labels_used_by_rule=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    tile = bind(reference.tile, retained_competition_scores=diagnostic_scores)
    window = bind(reference.window, tile=tile)
    with torch.inference_mode():
        bind(reference.screen, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
             ALTERNATIVES=(), settings=settings, window=window)(parser.parse_args())
