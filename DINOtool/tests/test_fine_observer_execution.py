import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dinotool.fine_observer_execution import FineObserverGraph, sync_free_patch_features, validate_features
from dinotool.finite_vip_observer import finite_proxy_attention


class FineExecutionTests(unittest.TestCase):
    def test_unused_stats_do_not_change_features(self):
        torch.manual_seed(4)
        head = SimpleNamespace(patch_size=2)
        attention = SimpleNamespace(num_heads=2, qkv=torch.nn.Linear(4, 12),
                                    proj=torch.nn.Linear(4, 4), proj_drop=torch.nn.Identity())
        tokens = torch.randn(1, 9, 4)
        for raw in (torch.eye(4)[None], torch.ones(1, 4, 4), torch.randn(1, 4, 4)):
            original, _, rows = finite_proxy_attention(head, attention, tokens, raw)
            actual, count, new_rows = finite_proxy_attention(head, attention, tokens, raw, count_empty=False)
            self.assertTrue(torch.equal(original, actual))
            self.assertIsNone(count)
            self.assertEqual(rows, new_rows)

    def test_sync_free_preserves_finite_validation(self):
        tensor = torch.ones(1, 1024, 8)
        with patch('dinotool.fine_observer_execution.fine_patch_features', return_value=tensor) as mock:
            self.assertIs(sync_free_patch_features('observer', 'rgb'), tensor)
            mock.assert_called_once_with('observer', 'rgb', capture_safe=True)
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            validate_features(torch.tensor([float('nan')]))

    def test_graph_refuses_wrong_owner_shape_and_cpu(self):
        observer = SimpleNamespace(device='cpu')
        graph = FineObserverGraph(observer)
        with self.assertRaisesRegex(ValueError, 'fixed observer'):
            graph(SimpleNamespace(device='cpu'), torch.ones(3, 512, 512))
        with self.assertRaisesRegex(ValueError, 'CHW512'):
            graph(observer, torch.ones(3, 336, 336))
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            graph(observer, torch.ones(3, 512, 512))


if __name__ == '__main__':
    unittest.main()
