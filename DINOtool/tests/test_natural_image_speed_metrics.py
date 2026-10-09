import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.benchmark_natural_sense_fast import (image_scores, without_fine_reader, without_fine_observer,
                                                 require_available_gpu)


class NaturalImageSpeedMetricTest(unittest.TestCase):
    def test_gpu_check_ignores_only_own_context(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), patch('os.getpid', return_value=123), patch(
                'subprocess.check_output', return_value='123\n') as queried:
            require_available_gpu(torch.device('cuda'))
            self.assertIn('0', queried.call_args.args[0])
            queried.return_value = '123\n456\n'
            with self.assertRaises(RuntimeError):
                require_available_gpu(torch.device('cuda'))

    def test_common_support_and_rejected_false_negatives(self):
        target = np.array([[0, 0], [1, -1]])
        scores = image_scores(target, {'a': np.array([[0, 0], [1, 0]]),
                                      'b': np.array([[0, 2], [255, 0]])}, ('a', 'b', 'c'))
        self.assertEqual(scores['common_class_indices'], [0, 1, 2])
        self.assertEqual(scores['ignored_target_pixels'], 1)
        self.assertEqual(scores['methods']['a']['common_support_miou_percent'], 200/3)
        self.assertEqual(scores['methods']['b']['common_support_miou_percent'], 50/3)
        self.assertEqual(scores['methods']['a']['own_union_miou_percent'], 100)
        self.assertEqual(scores['methods']['b']['confusion_matrix'][1][-1], 1)
        self.assertEqual(scores['methods']['b']['scored_target_pixels'], 3)

    def test_without_fine_preserves_unscreened_coupling_equation(self):
        local, broad = torch.tensor([[1., 2.]]), torch.tensor([[3., 4.]])
        operator = torch.tensor([[.25]])
        output, diagnostics = without_fine_reader(local, operator, broad)
        self.assertTrue(torch.equal(output['RivalFineHard_Exact'],
                                   local.double()+operator.double() @ (broad.double()-local.double())))
        self.assertEqual(diagnostics['mean_absolute_admission_potential'], 0)
        crops, count, stats = without_fine_observer(None, None, {'a': None}, None, None)
        self.assertEqual(crops, {'a': []})
        self.assertIsNone(count)
        self.assertEqual(stats['fine_forwards'], 0)


if __name__ == '__main__':
    unittest.main()
