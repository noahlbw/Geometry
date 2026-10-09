import unittest
from unittest.mock import patch

import torch

from dinotool.bounded_fine_coverage import FineCoverageReader
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.rival_alias_fast import retained_scores_fast
from dinotool.rival_fine_full import retained_scores
from dinotool.stratified_soft_alias import WideCrop


class BoundedFineExecutionTests(unittest.TestCase):
    def test_backend_does_not_relax_encoding_cap(self):
        observer = object()
        backend = lambda o, crops: tuple(torch.ones(1) for _ in crops)
        reader = FineCoverageReader(observer, feature_reader=backend)
        crops = (torch.zeros(3, 512, 512),)*4
        for _ in range(4):
            self.assertEqual(len(reader(observer, crops)), 4)
        with self.assertRaises(RuntimeError):
            reader(observer, crops)
        self.assertEqual(reader.calls, 16)

    def test_backend_must_preserve_crop_count(self):
        observer = object()
        reader = FineCoverageReader(observer, feature_reader=lambda o, crops: ())
        with self.assertRaises(RuntimeError):
            reader(observer, (torch.zeros(3, 512, 512),))

    def test_observer_ownership_is_fixed(self):
        execution = FineCoverageExecution()
        execution.reader(object())
        with self.assertRaises(ValueError):
            execution.reader(object())

    def test_each_image_gets_an_independent_cap(self):
        observer = object()
        execution = FineCoverageExecution()
        a, b = execution.reader(observer), execution.reader(observer)
        a.calls = 16
        self.assertEqual(b.calls, 0)
        self.assertIsNot(a, b)

    def test_only_declared_backends_are_allowed(self):
        with self.assertRaises(ValueError):
            FineCoverageExecution(cached=False, burst=True)
        self.assertIs(FineCoverageExecution(cached=False).score_reader, retained_scores)
        self.assertIs(FineCoverageExecution().score_reader, retained_scores_fast)

    def test_burst_reuses_graph_but_not_image_counter(self):
        observer = object()
        with patch('dinotool.bounded_fine_execution.FineObserverBurstGraph') as graph:
            execution = FineCoverageExecution(burst=True)
            a, b = execution.reader(observer), execution.reader(observer)
            self.assertEqual(graph.call_count, 1)
            self.assertIs(a.feature_reader, b.feature_reader)
            self.assertIsNot(a, b)

    def test_cached_scores_and_risks_match_legacy_at_nonzero_offsets(self):
        torch.manual_seed(71)
        classes, k, n = 3, 4, 3
        members = torch.arange(classes*k).reshape(classes, k)
        parents, canonical = torch.arange(classes).repeat_interleave(k), members[:, 0]
        global_coordinates = torch.tensor([[40., 24.], [56., 40.], [72., 56.]])
        fine_coordinates = global_coordinates-torch.tensor([32., 16.])
        valid = torch.tensor([True, True, False])
        wide = [WideCrop(torch.randn(16, classes*k), torch.randn(classes*k), 32, 16, 64, 64, 4, 64)]
        fine = [WideCrop(torch.randn(16, classes*k), torch.randn(classes*k), 0, 0, 64, 64, 4, 64)]
        count = torch.ones(128, 128)
        fine_count = torch.ones(512, 512)
        local, broad = torch.randn(n, classes), torch.randn(n, classes)
        operator = torch.eye(n, dtype=torch.float64)*.5
        operator[-1] = 0
        args = (local, operator, broad, wide, count, fine, fine_count, global_coordinates,
                fine_coordinates, valid, members, canonical, parents, (128, 128))
        old, old_risk, _ = retained_scores(*args)
        new, new_risk, _ = retained_scores_fast(*args)
        self.assertTrue(torch.equal(old_risk, new_risk))
        for name in old:
            self.assertTrue(torch.equal(old[name], new[name]))


if __name__ == '__main__':
    unittest.main()
