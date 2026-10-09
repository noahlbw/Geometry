"""Sparse correction invariants and a separate dense two-class reference."""
import unittest

import torch

from dinotool.sparse_alias_reuse import (SparseCrop, alias_layout, profile_aliases,
                                        sparse_scores, survivor_allocation)


class SparseAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.n, self.c, self.k = 9, 5, 4
        self.parents = torch.arange(self.c).repeat_interleave(self.k)
        self.layout = alias_layout(self.parents, torch.arange(self.c) * self.k, self.c)
        self.valid = torch.ones(self.n, dtype=torch.bool)
        self.local = torch.randn(self.n, self.c)
        self.broad = torch.randn(self.n, self.c)
        self.operator = torch.eye(self.n, dtype=torch.float64) * .4
        self.witness = torch.randn(self.n, self.c, self.k)
        self.crop = SparseCrop(torch.randn(self.n, self.c, self.k),
                               torch.arange(self.n)[:, None], torch.ones(self.n, 1))

    def call(self, **kwargs):
        return sparse_scores(kwargs.pop('local', self.local), self.operator,
            kwargs.pop('broad', self.broad), kwargs.pop('observations', [self.crop]),
            kwargs.pop('witness', self.witness), kwargs.pop('valid', self.valid), self.layout, **kwargs)

    def test_protection_mass_and_sparse_budget(self):
        _, stats, fields = self.call()
        self.assertLess(stats['survivor_mass_max_error'], 1e-12)
        self.assertEqual(stats['canonical_weight_max_error'], 0)
        self.assertEqual(stats['fine_forwards'], 0)
        self.assertEqual(fields['weights'].shape, (self.n, 2, self.k))
        self.assertEqual(stats['maximum_sparse_pair_alias_elements'], self.n * 2 * self.k)
        self.assertTrue(torch.equal(fields['weights'][~fields['keep']], torch.zeros_like(fields['weights'][~fields['keep']])))

    def test_uniform_source_recovers_hard(self):
        keep = torch.rand(self.n, 2, self.k) > .5
        canonical = torch.zeros_like(keep)
        canonical[..., 0] = True
        keep |= canonical
        soft = survivor_allocation(torch.ones_like(keep, dtype=torch.float64), keep, canonical, True)
        self.assertTrue(torch.equal(soft, keep.double()))

    def test_invalid_rows_have_no_correction(self):
        valid = self.valid.clone()
        valid[::2] = False
        result, _, fields = self.call(valid=valid)
        self.assertTrue(torch.equal(result[~valid], fields['base'][~valid]))
        self.assertEqual(float(fields['action'][~valid].abs().max()), 0)

    def test_zero_operator_keeps_geometry(self):
        self.operator.zero_()
        result, _, _ = self.call()
        self.assertTrue(torch.equal(result, self.local.double()))

    def test_single_alias_classes_cannot_be_suppressed(self):
        layout = alias_layout(torch.arange(self.c), torch.arange(self.c), self.c)
        crop = SparseCrop(self.crop.evidence[..., :1], self.crop.indices, self.crop.coefficients)
        result, _, fields = sparse_scores(self.local, self.operator, self.broad, [crop],
                                         self.witness[..., :1], self.valid, layout)
        self.assertTrue(torch.equal(result, fields['base']))

    def test_variable_alias_counts_padding(self):
        parents = torch.tensor([0, 0, 1, 2, 2, 2])
        layout = alias_layout(parents, torch.tensor([0, 2, 3]), 3)
        raw = torch.randn(self.n, 6)
        evidence = profile_aliases(raw, torch.randn(6), layout)
        crop = SparseCrop(evidence, self.crop.indices, self.crop.coefficients)
        result, stats, _ = sparse_scores(self.local[:, :3], self.operator, self.broad[:, :3],
                                        [crop], evidence, self.valid, layout)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertLess(stats['survivor_mass_max_error'], 1e-12)
        self.assertTrue(bool(torch.isneginf(evidence[:, ~layout.valid]).all()))

    def test_frozen_natural_temperature_and_variable_groups(self):
        parents = torch.tensor([0, 0, 1, 2, 2, 2])
        layout = alias_layout(parents, torch.tensor([0, 2, 3]), 3)
        evidence = profile_aliases(torch.randn(self.n, 6), torch.randn(6), layout)
        crop = SparseCrop(evidence, self.crop.indices, self.crop.coefficients)
        result, stats, fields = sparse_scores(self.local[:, :3], self.operator,
            self.broad[:, :3], [crop], evidence, self.valid, layout, beta=5.)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertLess(stats['survivor_mass_max_error'], 1e-12)
        self.assertEqual(fields['weights'].shape, (self.n, 2, 3))

    def test_class_permutation_equivariance(self):
        perm = torch.tensor([3, 0, 4, 1, 2])
        before = self.call()[0]
        crop = SparseCrop(self.crop.evidence[:, perm], self.crop.indices, self.crop.coefficients)
        after = self.call(local=self.local[:, perm], broad=self.broad[:, perm],
                          observations=[crop], witness=self.witness[:, perm])[0]
        torch.testing.assert_close(after, before[:, perm], atol=1e-12, rtol=0)

    def test_query_permutation_equivariance(self):
        perm = torch.randperm(self.n)
        before = self.call()[0]
        crop = SparseCrop(self.crop.evidence[perm], self.crop.indices, self.crop.coefficients)
        after = self.call(local=self.local[perm], broad=self.broad[perm],
                          observations=[crop], witness=self.witness[perm])[0]
        torch.testing.assert_close(after, before[perm], atol=1e-12, rtol=0)

    def test_action_respects_direction_and_capacity(self):
        _, _, fields = self.call()
        ids, rows = fields['pairs'], torch.arange(self.n)
        selected = self.witness[rows[:, None], ids].double()
        difference = selected.logsumexp(-1) - self.broad.gather(1, ids).double()
        target = difference[:, 0] - difference[:, 1]
        self.assertTrue(bool((fields['action'].abs() <= target.abs() + 1e-12).all()))
        self.assertTrue(bool((fields['action'] * target >= 0).all()))
        self.assertLess(float(fields['potential'].sum(-1).abs().max()), 1e-12)

    def test_two_class_dense_reference(self):
        from dinotool.rival_competition_admission import posterior_potential, project_actions
        self.c = 2
        layout = alias_layout(torch.arange(2).repeat_interleave(self.k), torch.tensor([0, self.k]), 2)
        g, b, f = self.local[:, :2], self.broad[:, :2], self.witness[:, :2].double()
        w = self.crop.evidence[:, :2].double()
        crop = SparseCrop(w, self.crop.indices, self.crop.coefficients)
        result, _, _ = sparse_scores(g, self.operator, b, [crop], f, self.valid, layout)
        base = g.double() + self.operator @ (b.double() - g.double())
        means_w, means_f = w.logsumexp(-1) - torch.tensor(float(self.k)).double().log(), f.logsumexp(-1) - torch.tensor(float(self.k)).double().log()
        directed = torch.zeros(self.n, 2, 2, dtype=torch.float64)
        for c in range(2):
            for d in range(2):
                if c == d:
                    continue
                reject = (w[:, c] > means_w[:, d, None]) & (f[:, c] < means_f[:, d, None])
                reject[:, 0] = False
                keep = ~reject
                source = f[:, c].softmax(-1).clamp_min(1e-6)
                source[:, 0] = 0
                source.masked_fill_(~keep, 0)
                count = keep.sum(-1) - 1
                weights = source * count[:, None] / source.sum(-1, keepdim=True).clamp_min(1e-6)
                weights[:, 0] = 1
                directed[:, c, d] = (w[:, c] + weights.log()).logsumexp(-1) - w[:, c].logsumexp(-1)
                directed[:, c, d] += (self.k / keep.sum(-1).double()).log()
        innovation = f.logsumexp(-1) - b.double()
        target = innovation[:, :, None] - innovation[:, None]
        action = project_actions(directed - directed.transpose(1, 2), target)
        potential, _ = posterior_potential(action, base.softmax(-1), self.valid)
        torch.testing.assert_close(result, base + self.operator @ potential, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
