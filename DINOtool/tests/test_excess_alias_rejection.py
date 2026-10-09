import unittest

import torch

from dinotool.excess_alias_rejection import (ExcessRejectConfig, canonical_support,
    contextual_retention, excess_delta, hard_delete_delta, semantic_contradictions,
    vocabulary_variants)
from dinotool.prompts import ClassSpec


class ExcessAliasTest(unittest.TestCase):
    def test_exact_identity(self):
        x = torch.randn(7, 3, 20)
        self.assertTrue(torch.equal(excess_delta(x, torch.ones_like(x)), torch.zeros(7, 3)))

    def test_equal_evidence(self):
        x = torch.full((6, 20), 3.)
        self.assertTrue(torch.equal(excess_delta(x, torch.rand_like(x)), torch.zeros(6)))

    def test_monotone_and_finite(self):
        x = torch.randn(7, 20) * 1000
        a = torch.rand_like(x)
        b = a * torch.rand_like(x)
        da, db = excess_delta(x, a), excess_delta(x, b)
        self.assertTrue(bool(torch.isfinite(db).all()))
        self.assertTrue(bool((db <= da + 1e-6).all()))
        self.assertTrue(bool((da <= 0).all()))
        self.assertGreaterEqual(float(db.min()), -torch.tensor(20.).log().item() - 1e-6)

    def test_no_redistribution_and_permutation(self):
        x = torch.randn(7, 20)
        r = torch.rand_like(x)
        p = torch.randperm(20)
        torch.testing.assert_close(excess_delta(x[:, p], r[:, p]), excess_delta(x, r))
        q = x.softmax(-1)
        direct = torch.log(1 - ((1-r) * (q-1/20).clamp_min(0)).sum(-1))
        torch.testing.assert_close(excess_delta(x, r), direct)

    def test_low_responsibility_neutral(self):
        x = torch.tensor([[10., -10., -10.]])
        r = torch.tensor([[1., 0., 0.]])
        self.assertEqual(float(excess_delta(x, r)), 0.)

    def test_canonical_and_shared_unknown(self):
        features = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [0., 1.]])
        conflict = semantic_contradictions(features, torch.tensor([0, 1, 0, 0]), torch.tensor([0, 1]))
        self.assertTrue(torch.equal(conflict[:3], torch.zeros(3, 2)))
        self.assertEqual(float(conflict[3, 1]), 1.)

    def test_support_excludes_query_cells(self):
        logits = torch.tensor([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
        coords = torch.tensor([[0., 0.], [16., 16.], [64., 64.], [80., 80.]])
        cfg = ExcessRejectConfig(minimum_effective_support=2.)
        p, valid, _ = canonical_support(logits, torch.ones(4, 4), coords, torch.ones(4, dtype=torch.bool), cfg)
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool((p[:2, 1] > .99).all()))
        changed = logits.clone()
        changed[:2] *= -1
        again, _, _ = canonical_support(changed, torch.ones(4, 4), coords, torch.ones(4, dtype=torch.bool), cfg)
        torch.testing.assert_close(p[:2], again[:2])

    def test_identity_geometry_unknown(self):
        p, valid, _ = canonical_support(torch.randn(5, 2), torch.eye(5),
            torch.arange(10).reshape(5, 2).float() * 64, torch.ones(5, dtype=torch.bool))
        self.assertFalse(bool(valid.any()))
        r = contextual_retention(torch.zeros(5, 2), torch.ones(5, 2), p, valid,
                                  torch.ones(2, 2), torch.tensor([0, 1]))
        self.assertTrue(torch.equal(r, torch.ones_like(r)))

    def test_two_contradictions_and_gain(self):
        local = torch.zeros(1, 2)
        wide = torch.ones_like(local)
        support = torch.tensor([[.1, .9]])
        conflict = torch.tensor([[0., 1.], [0., 0.]])
        r = contextual_retention(local, wide, support, torch.tensor([True]), conflict, torch.tensor([0, 1]))
        torch.testing.assert_close(r, torch.tensor([[.2, 1.]]))
        neutral = contextual_retention(local, local, support, torch.tensor([True]), conflict, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(neutral, torch.ones_like(neutral)))

    def test_hard_deletion_count_normalization(self):
        x = torch.full((2, 20), 3.)
        r = torch.zeros_like(x)
        r[:, :4] = 1.
        torch.testing.assert_close(hard_delete_delta(x, r), torch.zeros(2), atol=1e-6, rtol=0.)
        self.assertTrue(torch.equal(hard_delete_delta(x, torch.ones_like(x)), torch.zeros(2)))

    def test_stress_preserves_count_and_canonical(self):
        specs = [ClassSpec(name, tuple([name] + [name + ' alias ' + str(i) for i in range(19)]))
                 for name in ('road', 'car')]
        arms, manifest = vocabulary_variants(specs)
        for name, classes in arms.items():
            for spec in classes:
                self.assertEqual(len(set(spec.synonyms)), 20)
                self.assertEqual(spec.synonyms[0], spec.name)
        self.assertEqual(manifest['wrong_parent'][0]['declared_rival'], 'car')


if __name__ == '__main__':
    unittest.main()
