import unittest

import numpy as np
import torch

from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator


class DeviceAccumulatorTests(unittest.TestCase):
    def check_stitching(self, device, classes, background):
        rng = np.random.default_rng(20261005)
        height, width, side = 35, 49, 24
        probs = np.zeros((classes, height, width), np.float32)
        normalizer = np.zeros((height, width), np.float32)
        with DeviceProbabilityAccumulator(classes, height, width, device) as accumulator:
            for top in (0, height - side):
                for left in (0, 12, width - side):
                    p = rng.random((classes, side, side), dtype=np.float32)
                    p /= p.sum(0)
                    weights = rng.random((side, side), dtype=np.float32) + np.float32(.05)
                    probs[:, top:top + side, left:left + side] += p * weights[None]
                    normalizer[top:top + side, left:left + side] += weights
                    accumulator.add(torch.from_numpy(p).to(device), torch.from_numpy(weights).to(device), left, top)
            self.assertTrue(np.array_equal(accumulator.probabilities.cpu().numpy(), probs))
            self.assertTrue(np.array_equal(accumulator.normalizer.cpu().numpy(), normalizer))
            normalized = probs / normalizer[None]
            self.assertTrue(np.array_equal((accumulator.probabilities / accumulator.normalizer[None]).cpu().numpy(), normalized))
            labels, confidence = normalized.argmax(0).astype(np.uint8), normalized.max(0)
            thresholds = rng.random(classes) / classes if background else np.zeros(classes)
            for threshold in (None, .4):
                expected = labels.copy()
                if threshold is not None:
                    expected[confidence < threshold] = 0
                expected_calibrated = labels.copy()
                if background:
                    expected_calibrated[confidence < thresholds[labels]] = 0
                actual, calibrated = accumulator.finalize_outputs(threshold, thresholds, background)
                self.assertTrue(np.array_equal(actual, expected))
                self.assertTrue(np.array_equal(calibrated, expected_calibrated))

    def test_cpu_stitching_and_frozen_thresholds_exact(self):
        for classes in (5, 150):
            for background in (False, True):
                self.check_stitching('cpu', classes, background)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for device arithmetic check')
    def test_cuda_stitching_and_frozen_thresholds_exact(self):
        for classes in (5, 150):
            for background in (False, True):
                self.check_stitching('cuda', classes, background)

    def test_ties_threshold_boundaries_and_coverage(self):
        with DeviceProbabilityAccumulator(3, 1, 2, 'cpu') as accumulator:
            with self.assertRaises(ValueError):
                accumulator.finalize_outputs(None, [0., 0., 0.], False)
            accumulator.add(torch.tensor([[[.5, .0]], [[.5, .5]], [[.0, .5]]]), torch.ones(1, 2), 0, 0)
            default, calibrated = accumulator.finalize_outputs(.5, [0., .5, .0], True)
            self.assertEqual(default.tolist(), [[0, 1]])
            self.assertEqual(calibrated.tolist(), [[0, 1]])


if __name__ == '__main__':
    unittest.main()
