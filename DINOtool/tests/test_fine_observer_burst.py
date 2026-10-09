import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dinotool.fine_observer_burst import FineObserverBurstGraph
from dinotool.fine_observer_execution import validate_features


class FineBurstTests(unittest.TestCase):
    def test_owner_shape_count_and_dtype_checks(self):
        observer = SimpleNamespace(device='cpu')
        graph = FineObserverBurstGraph(observer)
        with self.assertRaisesRegex(ValueError, 'fixed observer'):
            graph(SimpleNamespace(device='cpu'), [torch.ones(3, 512, 512)])
        for crops in ([], [torch.ones(3, 336, 336)], [torch.ones(3, 512, 512)]*5):
            with self.assertRaisesRegex(ValueError, 'CHW512'):
                graph(observer, crops)
        with self.assertRaisesRegex(ValueError, 'dtypes'):
            graph(observer, [torch.ones(3, 512, 512), torch.ones(3, 512, 512, dtype=torch.float64)])
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            graph(observer, [torch.ones(3, 512, 512)])

    def test_any_invalid_crop_is_rejected(self):
        features = torch.ones(4, 7, 8)
        features[2, 3, 1] = torch.nan
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            validate_features(features)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph required')
    def test_batch_one_order_clone_lifetime_and_replay(self):
        observer = SimpleNamespace(device='cuda')
        def feature_source(_, rgb, capture_safe=False):
            self.assertTrue(capture_safe)
            return rgb[:, :2, :2].flatten()[None, :, None]*2
        graph = FineObserverBurstGraph(observer)
        with patch('dinotool.fine_observer_burst.fine_patch_features', side_effect=feature_source):
            for count in (1, 2, 4):
                crops = [torch.full((3, 512, 512), float(index+1), device='cuda') for index in range(count)]
                first = graph(observer, crops)
                saved = [value.clone() for value in first]
                second = graph(observer, [rgb+3 for rgb in crops])
                for index in range(count):
                    self.assertTrue(torch.equal(first[index], saved[index]))
                    self.assertTrue(torch.equal(second[index], saved[index]+6))
        self.assertEqual(set(graph.graphs), {1, 2, 4})
        self.assertEqual(graph.replays, 6)


if __name__ == '__main__':
    unittest.main()
