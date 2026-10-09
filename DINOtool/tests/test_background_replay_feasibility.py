import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from background_replay_feasibility import retention_frontier, union_shift


def witness(probabilities, labels):
    return dict(probabilities={'default': np.asarray(probabilities)}, labels=np.asarray(labels),
                quality=np.ones(len(labels)))


class BackgroundReplayFeasibilityTest(unittest.TestCase):
    def test_threshold_ties_do_not_delete_the_foreground_guard(self):
        rows = [witness([[.2, .8], [.4, .6], [.3, .7]], [1, 1, 0])]
        result = retention_frontier(rows, ['background', 'object'], [0, .5], 'default')[0]
        self.assertEqual(result['foreground95_guard'], .6)
        self.assertEqual(result['pseudo_tp_loss_percent'], 0.)
        self.assertEqual(result['pseudo_background_fp_removed_percent'], 0.)

    def test_different_point_counts_do_not_destroy_image_balance(self):
        a = witness([[.4, .6], [.3, .7]], [1, 0])
        b = witness([[.1, .9], [.2, .8]], [1, 0])
        first = retention_frontier([a, b], ['background', 'object'], [0, .5], 'default')[0]
        repeated = witness(np.repeat(a['probabilities']['default'], 100, axis=0), np.repeat(a['labels'], 100))
        second = retention_frontier([repeated, b], ['background', 'object'], [0, .5], 'default')[0]
        self.assertEqual(first['foreground95_guard'], second['foreground95_guard'])
        self.assertEqual(first['pseudo_background_fp_removed_percent'], second['pseudo_background_fp_removed_percent'])

    def test_no_positive_support_does_not_invent_a_guard(self):
        result = retention_frontier([witness([[.2, .8]], [0])], ['background', 'object'], [0, .5], 'default')[0]
        self.assertEqual(result['foreground95_guard'], .5)
        self.assertIsNone(result['pseudo_tp_loss_percent'])
        self.assertFalse(result['positive_image_support_at_least8'])

    def test_residual_union_delta_matches_the_existing_linear_reader(self):
        operator = torch.tensor([[.4, .1], [.1, .3]], dtype=torch.float64)
        local = torch.tensor([[.3, .7], [.1, .2]], dtype=torch.float64)
        wide = torch.tensor([[.9, .2], [.6, .4]], dtype=torch.float64)
        potential = torch.tensor([[.1, -.1], [.2, -.2]], dtype=torch.float64)
        original = local+operator@(wide-local+potential)
        adjusted = local.clone()
        adjusted[:, 0] += np.log(4)
        control = adjusted+operator@(wide-adjusted+potential)
        torch.testing.assert_close(union_shift(original, operator.sum(-1), 4), control)


if __name__ == '__main__':
    unittest.main()
