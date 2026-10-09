import unittest
from types import SimpleNamespace

import numpy as np

from dinotool.natural_sense_calibration import (choose_profile, foreground_thresholds,
    reject_labels, weighted_quantile, witness_error, sample_accumulated_probabilities,
    finalize_thresholds)


class NaturalSenseCalibrationTest(unittest.TestCase):
    def test_inconclusive_profile_keeps_default(self):
        self.assertEqual(choose_profile([])['profile'], 'default')
        self.assertEqual(choose_profile([dict(default_error=.2, frozen_error=.1)])['profile'], 'default')
        self.assertEqual(choose_profile([dict(default_error=.2, frozen_error=.2)]*8)['profile'], 'default')

    def test_repeated_actual_reader_improvement_can_choose_source(self):
        self.assertEqual(choose_profile([dict(default_error=.2, frozen_error=.1)]*8)['profile'], 'frozen')

    def test_weighted_quantile_and_ties(self):
        self.assertEqual(weighted_quantile([.2, .2, .8], [1, 1, 1], .05), .2)
        with self.assertRaises(ValueError):
            weighted_quantile([.2], [0], .05)

    def test_no_background_protocol_keeps_rejection_off(self):
        self.assertEqual(foreground_thresholds([], 3, False, .4)['thresholds'], [0, 0, 0])

    def test_actual_probabilities_cap_the_class_threshold(self):
        row = dict(probabilities=np.array([[.3, .4, .3], [.1, .1, .8]]),
                   labels=np.array([1, 2]), quality=np.ones(2))
        result = foreground_thresholds([row]*8, 3, True, .6)
        self.assertEqual(result['thresholds'], [0, .4, .6])
        self.assertTrue(all(x <= .6 for x in result['thresholds']))

    def test_sparse_witnesses_do_not_fit_a_new_threshold(self):
        row = dict(probabilities=np.array([[.3, .4, .3]]), labels=np.array([1]), quality=np.ones(1))
        self.assertEqual(foreground_thresholds([row], 3, True, .6)['thresholds'], [0, .6, .6])

    def test_threshold_operation_preserves_reliable_foreground(self):
        labels, confidence = np.array([1, 1, 2, 0]), np.array([.4, .39, .6, .1])
        np.testing.assert_array_equal(reject_labels(labels, confidence, [0, .4, .6], True), [1, 0, 2, 0])
        np.testing.assert_array_equal(reject_labels(labels, confidence, [0, 0, 0], False), labels)

    def test_error_uses_class_balance_not_confidence_nll(self):
        self.assertEqual(witness_error(np.array([[.99, .01], [.51, .49]]),
            np.array([0, 1]), np.ones(2)), .5)
        self.assertIsNone(witness_error(np.ones((2, 2))*.5, np.array([0, 1]), np.zeros(2)))

    def test_samples_actual_accumulated_probabilities(self):
        acc = SimpleNamespace(height=1, width=2, classes=2,
            probabilities=np.array([[[.6, .2]], [[1.4, 1.8]]]), normalizer=np.array([[2., 2.]]))
        np.testing.assert_allclose(sample_accumulated_probabilities(acc, [[0, 0], [0, 1]]), [[.3, .7], [.1, .9]])
        with self.assertRaises(ValueError):
            sample_accumulated_probabilities(acc, [[1, 0]])

    def test_thresholds_use_float_confidence_not_quantized_confidence(self):
        acc = SimpleNamespace(height=1, width=2, classes=2,
            probabilities=np.array([[[.6, .2]], [[1.4, 1.8]]]), normalizer=np.array([[2., 2.]]))
        np.testing.assert_array_equal(finalize_thresholds(acc, [0, .7], True), [[1, 1]])
        np.testing.assert_array_equal(finalize_thresholds(acc, [0, .7001], True), [[0, 1]])


if __name__ == '__main__':
    unittest.main()
