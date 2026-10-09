import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from task_scope_witness_replay import choose_profile, crossfit, errors, replay


def inputs():
    row = dict(labels=np.array([0, 1, 0]), quality=np.array([1., .25, 1.]),
        coordinates=np.array([[0, 0], [1, 0], [2, 0]]),
        probabilities=dict(default=np.array([[.8, .2], [.2, .8], [.8, .2]]),
                           frozen=np.array([[.2, .8], [.8, .2], [.8, .2]])))
    challenge = dict(labels=np.array([0, 2, 2]), quality=np.array([1., 1., 1.]),
        coordinates=row['coordinates'].copy(),
        probabilities={name: np.array([[.8, .1, .1], [.1, .1, .8], [.1, .1, .8]])
                       for name in ('default', 'frozen')})
    rows, other_rows = [copy.deepcopy(row) for _ in range(8)], [copy.deepcopy(challenge) for _ in range(8)]
    metadata = dict(status='complete', target_masks_loaded=False, target_label_tuning=False,
        weights_frozen=True, head_weights_unchanged=True, smoke_only=False,
        dataset='foreground', image_keys=[str(i) for i in range(8)], processed_images=8, total_images=8,
        identity=dict(classes=['cat', 'dog'], checkpoints={'model': 'unchanged'}))
    metadata['profile'] = choose_profile([errors(r, r['quality'], k) for k, r in zip(metadata['image_keys'], rows)])
    other = copy.deepcopy(metadata)
    other.update(dataset='open_world', identity=dict(classes=['background', 'cat', 'dog'], checkpoints={'model': 'unchanged'}))
    return metadata, rows, other, other_rows


class TaskScopeWitnessReplayTest(unittest.TestCase):
    def test_background_and_rival_challenges_are_excluded_without_changing_inputs(self):
        args = inputs()
        before = args[1][0]['quality'].copy()
        result = replay(*args)
        self.assertEqual(result['before_trusted_points'], 24)
        self.assertEqual(result['scoped_trusted_points'], 8)
        self.assertEqual(result['trusted_background_challenges'], 8)
        self.assertEqual(result['trusted_foreground_disagreements'], 8)
        self.assertEqual(result['crossfit']['scored_images'], 8)
        self.assertFalse(result['gt_loaded'])
        np.testing.assert_array_equal(args[1][0]['quality'], before)

    def test_coordinate_or_class_order_mismatch_is_rejected(self):
        args = inputs()
        args[3][0]['coordinates'][0, 0] = 3
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            replay(*args)
        args = inputs()
        args[2]['identity']['classes'] = ['background', 'dog', 'cat']
        with self.assertRaisesRegex(ValueError, 'taxonomy'):
            replay(*args)

    def test_gt_derived_or_unfrozen_records_are_rejected(self):
        args = inputs()
        args[0]['target_masks_loaded'] = True
        with self.assertRaisesRegex(ValueError, 'mask-free'):
            replay(*args)
        args = inputs()
        args[2]['weights_frozen'] = False
        with self.assertRaisesRegex(ValueError, 'mask-free'):
            replay(*args)

    def test_no_scope_support_retains_default_without_inventing_errors(self):
        args = inputs()
        for row in args[3]:
            row['quality'].fill(0)
        result = replay(*args)
        self.assertEqual(result['scoped_profile']['profile'], 'default')
        self.assertEqual(result['scoped_profile']['paired_images'], 0)
        self.assertIsNone(result['crossfit']['scoped_rule_scoped_error'])

    def test_profile_selection_excludes_each_heldout_image_block(self):
        rows = [dict(key=str(i), default_error=.1 if i % 2 == 0 else 0.,
                     frozen_error=0. if i % 2 == 0 else 1.) for i in range(16)]
        result = crossfit(rows, rows, folds=2)
        self.assertEqual(result['folds'][0]['original_rule_profile'], 'default')
        self.assertEqual(result['folds'][1]['original_rule_profile'], 'frozen')
        self.assertAlmostEqual(result['original_rule_scoped_error'], .55)


if __name__ == '__main__':
    unittest.main()
