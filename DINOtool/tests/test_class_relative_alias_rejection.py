import unittest

import torch

from dinotool.class_relative_alias_rejection import relative_retention
from dinotool.excess_alias_rejection import contextual_retention, excess_delta


class ClassRelativeAliasTest(unittest.TestCase):
    def setUp(self):
        self.support = torch.tensor([[.1, .9], [.9, .1]])
        self.conflict = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
        self.parents = torch.tensor([0, 1, 0])

    def test_rival_specific(self):
        r = relative_retention(self.support, torch.tensor([True, True]), self.conflict, self.parents)
        torch.testing.assert_close(r, torch.tensor([[.2, 1., 1.], [1., .2, 1.]]))

    def test_unknown_keeps_all(self):
        r = relative_retention(self.support, torch.tensor([False, False]), self.conflict, self.parents)
        self.assertTrue(torch.equal(r, torch.ones_like(r)))

    def test_no_class_contradiction_neutral(self):
        r = relative_retention(self.support, torch.tensor([True, True]), torch.zeros_like(self.conflict), self.parents)
        self.assertTrue(torch.equal(r, torch.ones_like(r)))

    def test_no_absolute_gain_dependency(self):
        r = relative_retention(self.support, torch.tensor([True, True]), self.conflict, self.parents)
        local = torch.full((2, 3), .5)
        old = contextual_retention(local, local-.1, self.support, torch.tensor([True, True]), self.conflict, self.parents)
        self.assertTrue(torch.equal(old, torch.ones_like(old)))
        self.assertLess(float(r[0, 0]), 1.)
        positive = contextual_retention(local, local+.07, self.support, torch.tensor([True, True]), self.conflict, self.parents)
        torch.testing.assert_close(positive, r)

    def test_class_permutation(self):
        r = relative_retention(self.support, torch.tensor([True, True]), self.conflict, self.parents)
        p = torch.tensor([1, 0])
        again = relative_retention(self.support[:, p], torch.tensor([True, True]), self.conflict[:, p], 1-self.parents)
        torch.testing.assert_close(again, r)

    def test_bounded_and_equal_evidence(self):
        r = relative_retention(self.support, torch.tensor([True, True]), self.conflict, self.parents)
        self.assertTrue(bool(((r >= 0) & (r <= 1)).all()))
        self.assertTrue(torch.equal(excess_delta(torch.full_like(r, 3.), r), torch.zeros(2)))


if __name__ == '__main__':
    unittest.main()
