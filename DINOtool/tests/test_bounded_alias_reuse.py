import unittest

import torch

from dinotool.bounded_alias_reuse import bounded_scores, contender_union
from dinotool.sparse_alias_reuse import SparseCrop, alias_layout, profile_aliases, sparse_scores


class BoundedAliasReuseTests(unittest.TestCase):
    def fixture(self, classes=5):
        torch.manual_seed(41)
        n, k = 9, 4
        parents = torch.arange(classes).repeat_interleave(k)
        layout = alias_layout(parents, torch.arange(classes) * k, classes)
        valid = torch.ones(n, dtype=torch.bool)
        valid[::3] = False
        crop = SparseCrop(torch.randn(n, classes, k), torch.arange(n)[:, None], torch.ones(n, 1))
        return (torch.randn(n, classes), torch.eye(n, dtype=torch.float64) * .4,
                torch.randn(n, classes), [crop], torch.randn(n, classes, k), valid, layout)

    def test_union_contains_all_branch_top2_and_is_bounded(self):
        torch.manual_seed(8)
        for classes in (2, 5, 150):
            scores = [torch.randn(19, classes) for _ in range(3)]
            pairs, active = contender_union(*scores)
            self.assertLessEqual(pairs.shape[1], 6)
            selected = torch.zeros(19, classes, dtype=torch.bool).scatter_(1, pairs, active)
            for branch in scores:
                self.assertTrue(bool(selected.gather(1, branch.argsort(dim=-1, descending=True, stable=True)[:, :2]).all()))

    def test_baseline_top2_matches_original_sparse_formula(self):
        for classes in (2, 5, 12):
            inputs = self.fixture(classes)
            old = sparse_scores(*inputs)[0]
            new = bounded_scores(*inputs, baseline_only=True)[0]
            self.assertTrue(torch.allclose(new, old, atol=1e-10, rtol=0), float((new-old).abs().max()))

    def test_blocking_is_exact_and_invalid_queries_identity(self):
        inputs = self.fixture()
        a, stats, fields = bounded_scores(*inputs, chunk=4)
        b = bounded_scores(*inputs, chunk=256)[0]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[~inputs[5]], fields['base'][~inputs[5]]))
        self.assertEqual(stats['fine_forwards'], 0.)
        self.assertLessEqual(stats['survivor_mass_max_error'], 1e-10)
        self.assertEqual(stats['canonical_weight_max_error'], 0.)

    def test_variable_alias_padding_and_natural_temperature(self):
        inputs = list(self.fixture(4))
        parents = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        layout = alias_layout(parents, torch.tensor([0, 1, 4, 6]), 4)
        evidence = profile_aliases(torch.randn(9, len(parents)), torch.randn(len(parents)), layout)
        crop = SparseCrop(evidence, torch.arange(9)[:, None], torch.ones(9, 1))
        inputs[3], inputs[4], inputs[6] = [crop], evidence, layout
        score, stats, _ = bounded_scores(*inputs, beta=5.)
        self.assertTrue(bool(torch.isfinite(score).all()))
        self.assertLessEqual(stats['survivor_mass_max_error'], 1e-10)
        self.assertEqual(stats['canonical_weight_max_error'], 0.)

    def test_150_classes_keep_constant_pair_budget(self):
        inputs = self.fixture(150)
        _, stats, fields = bounded_scores(*inputs, chunk=4)
        self.assertEqual(fields['pairs'].shape, (9, 6))
        self.assertLessEqual(stats['maximum_pair_alias_elements'], 4 * 6 * 6 * 4)


if __name__ == '__main__':
    unittest.main()
