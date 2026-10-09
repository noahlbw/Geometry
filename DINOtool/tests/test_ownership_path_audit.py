import unittest

import numpy as np

from dinotool.ownership_path_audit import action_decomposition, target_margin_signs


class OwnershipPathAuditTest(unittest.TestCase):
    def test_pure_class_cycle_cannot_be_written_as_class_scores(self):
        old = np.zeros((1, 3, 3))
        new = old.copy()
        new[0, 0, 1] = new[0, 1, 2] = new[0, 2, 0] = -1.
        value, stats = action_decomposition(old, new, np.array([True]))
        np.testing.assert_array_equal(value, 0.)
        self.assertEqual(stats['realized_energy'], 0.)
        self.assertEqual(stats['requested_energy'], stats['cycle_energy'])

    def test_consistent_gradient_is_fully_retained(self):
        potential = np.array([[-1., 0., 1.]])
        edges = potential[:, :, None]-potential[:, None, :]
        directed = np.minimum(edges, 0.)
        value, stats = action_decomposition(np.zeros_like(directed), directed, np.array([True]))
        np.testing.assert_allclose(value, potential, atol=0, rtol=0)
        self.assertEqual(stats['cycle_energy'], 0.)

    def test_symmetric_suppression_is_competitively_neutral(self):
        directed = np.array([[[0., -2.], [-2., 0.]]])
        value, stats = action_decomposition(np.zeros_like(directed), directed, np.array([True]))
        np.testing.assert_array_equal(value, 0.)
        self.assertGreater(stats['mean_added_directed_suppression'], 0.)
        self.assertEqual(stats['requested_energy'], 0.)

    def test_signs_use_one_fixed_old_competitor(self):
        old = np.array([[5., 4., 1.], [1., 2., 4.], [0., 0., 0.]])
        before = np.array([[1., 0., 0.], [0., 0., 2.], [1., 0., 0.]])
        after = -before
        target = np.array([0, 2, -1])
        classes, stats = target_margin_signs(before, after, old, target, np.ones(3, bool))
        self.assertEqual(classes[0, 2, 0], 1)
        self.assertEqual(classes[2, 2, 0], 1)
        self.assertEqual(int(classes.sum()), 2)
        self.assertEqual(stats['valid_labelled_queries'], 2)

    def test_invalid_queries_cannot_carry_actions(self):
        with self.assertRaises(ValueError):
            action_decomposition(np.zeros((1, 2, 2)), np.ones((1, 2, 2)), np.array([False]))
