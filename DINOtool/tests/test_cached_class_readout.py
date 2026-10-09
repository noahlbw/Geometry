"""Fixed text metadata caching must preserve the original arithmetic exactly."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_sparse_alias_natural_cached import FrozenTextCache, GroupedFrozenTextCache
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.natural_text_adaptation import class_logits


class CachedClassReadoutTests(unittest.TestCase):
    def test_exact_fixed_and_variable_groups(self):
        torch.manual_seed(20261005)
        for counts in ([20] * 5, [1, 3, 2, 7, 1], [1 + c % 5 for c in range(150)]):
            parents = torch.arange(len(counts)).repeat_interleave(torch.tensor(counts))
            scores = torch.randn(17, len(parents))
            salience = torch.randn(len(parents))
            cache = FrozenTextCache()
            for _ in range(2):
                self.assertTrue(torch.equal(cache.geometry(scores, parents, len(counts)),
                    alias_class_scores(scores, parents, len(counts))))
                for tau, tem in ((1., 1.), (5., .3)):
                    self.assertTrue(torch.equal(cache.wide(scores, salience, parents, len(counts), tau, tem),
                        class_logits(scores, salience, parents, len(counts), tau, tem)))

    def check_grouped(self, device):
        torch.manual_seed(20261005)
        for counts in ([20] * 5, [1, 3, 2, 7, 1], [1 + c % 5 for c in range(150)]):
            parents = torch.arange(len(counts), device=device).repeat_interleave(torch.tensor(counts, device=device))
            permutation = torch.randperm(len(parents), device=device)
            parents = parents[permutation]
            cache, reference = GroupedFrozenTextCache(), FrozenTextCache()
            for dtype in (torch.float32, torch.float64):
                scores = torch.randn(34, len(parents), device=device, dtype=dtype)[::2]
                salience = torch.randn(len(parents), device=device, dtype=dtype)
                for temperature in (.07, .4):
                    self.assertTrue(torch.equal(cache.geometry(scores, parents, len(counts), temperature),
                        reference.geometry(scores, parents, len(counts), temperature)))
                for tau, tem in ((1., 1.), (5., .3)):
                    self.assertTrue(torch.equal(cache.wide(scores, salience, parents, len(counts), tau, tem),
                        reference.wide(scores, salience, parents, len(counts), tau, tem)))
                shaped = scores.reshape(1, 17, -1)
                self.assertTrue(torch.equal(cache.geometry(shaped, parents, len(counts)),
                    reference.geometry(shaped, parents, len(counts))))
            self.assertEqual(len(cache.reduction_members(parents, len(counts))), len(set(counts)))

    def test_grouped_cpu_exact(self):
        self.check_grouped('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for exact GPU reductions.')
    def test_grouped_cuda_exact(self):
        self.check_grouped('cuda')

    def test_grouped_rejects_empty_classes_and_nonfinite_wide(self):
        cache = GroupedFrozenTextCache()
        parents = torch.tensor([0, 0, 1])
        with self.assertRaises(ValueError):
            cache.geometry(torch.zeros(3, 3), parents, 3)
        for tau, tem in ((0., 1.), (1., 0.)):
            with self.assertRaises(ValueError):
                cache.wide(torch.zeros(3, 3), torch.zeros(3), parents, 2, tau, tem)
        with self.assertRaises(ValueError):
            cache.wide(torch.full((3, 3), float('nan')), torch.zeros(3), parents, 2)


if __name__ == '__main__':
    unittest.main()
