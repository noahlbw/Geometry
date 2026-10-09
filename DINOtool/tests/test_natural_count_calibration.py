import math
import unittest

import torch

from dinotool.natural_count_calibration import (count_prior_delta, calibrate_coupled_counts,
                                               witness_error, choose_count_strength)


class NaturalCountCalibrationTest(unittest.TestCase):
    def test_zero_strength_is_bitwise_unchanged(self):
        scores = torch.randn(3, 2)
        result = calibrate_coupled_counts(scores, torch.eye(3), torch.tensor([2, 4]), strength=0.)
        self.assertIs(result, scores)

    def test_equal_counts_leave_class_probabilities_unchanged(self):
        scores = torch.tensor([[2., 3.], [1., -2.]], dtype=torch.float64)
        operator = torch.tensor([[.4, -.1], [-.1, .3]], dtype=torch.float64)
        result = calibrate_coupled_counts(scores, operator, torch.tensor([20, 20]))
        torch.testing.assert_close(result.softmax(-1), scores.softmax(-1), atol=1e-14, rtol=1e-14)

    def test_delta_matches_exact_reconstruction_replay(self):
        operator = torch.tensor([[.4, -.1], [-.1, .3]], dtype=torch.float64)
        local = torch.tensor([[2., 3.], [1., -2.]], dtype=torch.float64)
        broad = torch.tensor([[3., 5.], [1., 3.]], dtype=torch.float64)
        potential = torch.tensor([[.2, -.2], [.1, -.1]], dtype=torch.float64)
        counts = torch.tensor([2, 7])
        original = local + operator @ (broad - local + potential)
        changed = local + operator @ (broad - counts.double().log()[None]/3 - local + potential)
        replay = calibrate_coupled_counts(original, operator, counts, tau=3.)
        torch.testing.assert_close(replay, changed, atol=1e-14, rtol=1e-14)

    def test_duplicate_only_size_advantage_disappears(self):
        local = torch.tensor([[4., 4.]], dtype=torch.float64)
        # Each class has identical alias evidence; only their multiplicities differ.
        broad = local + torch.tensor([[math.log(2), math.log(8)]])
        operator = torch.tensor([[.5]], dtype=torch.float64)
        original = local + operator @ (broad-local)
        corrected = calibrate_coupled_counts(original, operator, torch.tensor([2, 8]))
        self.assertEqual(original.argmax(-1).item(), 1)
        torch.testing.assert_close(corrected[:, 0], corrected[:, 1], atol=1e-7, rtol=0)

    def test_invalid_row_receives_no_offset(self):
        operator = torch.tensor([[.5, 0.], [0., 0.]])
        delta = count_prior_delta(operator, torch.tensor([2, 8]))
        self.assertTrue((delta[1] == 0).all())

    def test_more_confident_wrong_scores_do_not_improve_objective(self):
        scores = torch.tensor([[1., 2.], [2., 1.]])
        labels, weights = torch.tensor([0, 1]), torch.ones(2)
        self.assertEqual(witness_error(scores, labels, weights), 1.)
        self.assertEqual(witness_error(scores*100, labels, weights), 1.)

    def test_repeated_improvement_can_enable_adjustment(self):
        row = dict(scores=torch.tensor([[1., 1.2], [0., 3.]], dtype=torch.float64),
                   operator=torch.eye(2)*.5, labels=torch.tensor([0, 1]), quality=torch.ones(2))
        result = choose_count_strength([row]*8, torch.tensor([1, 8]))
        self.assertGreater(result['strength'], 0.)
        self.assertFalse(result['target_masks_loaded'])

    def test_unsupported_or_inconclusive_adjustment_stays_off(self):
        self.assertEqual(choose_count_strength([], torch.tensor([1, 8]))['strength'], 0.)
        row = dict(scores=torch.tensor([[2., 1.], [0., 3.]], dtype=torch.float64),
                   operator=torch.eye(2)*.5, labels=torch.tensor([0, 1]), quality=torch.ones(2))
        self.assertEqual(choose_count_strength([row]*8, torch.tensor([1, 8]))['strength'], 0.)
        self.assertEqual(choose_count_strength([row], torch.tensor([1, 8]))['strength'], 0.)

    def test_invalid_counts_and_strength_are_rejected(self):
        for counts in (torch.tensor([0, 2]), torch.tensor([1., 2.5])):
            with self.assertRaises(ValueError):
                count_prior_delta(torch.eye(2), counts)
        for strength in (-1., 2., float('nan')):
            with self.assertRaises(ValueError):
                count_prior_delta(torch.eye(2), torch.tensor([1, 2]), strength=strength)


if __name__ == '__main__':
    unittest.main()
