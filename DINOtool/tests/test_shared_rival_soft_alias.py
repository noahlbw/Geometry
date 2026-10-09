import unittest

import torch

from dinotool.shared_rival_soft_alias import (alias_permutation, local_contradiction,
    normalized_delta, positive_margin, rival_reference, weights_from_risk)
from dinotool.sparse_alias_reuse import alias_layout


def layout(counts):
    parents = torch.arange(len(counts)).repeat_interleave(torch.tensor(counts))
    canonical = torch.tensor([sum(counts[:c]) for c in range(len(counts))])
    return alias_layout(parents, canonical, len(counts))


class SharedRivalTests(unittest.TestCase):
    def test_stable_all_class_summary_matches_exhaustive_reference(self):
        torch.manual_seed(7)
        for counts in ([2, 3, 1], [20] * 6, [3] * 150):
            groups = layout(counts)
            evidence = (torch.randn(5, *groups.members.shape) * 1000).double()
            evidence.masked_fill_(~groups.valid[None], -torch.inf)
            scores = evidence.logsumexp(-1) - groups.counts.double().log()
            expected = torch.stack([scores[:, torch.arange(len(counts)) != c].logsumexp(-1)
                - torch.tensor(len(counts) - 1).double().log() for c in range(len(counts))], -1)
            self.assertTrue(torch.allclose(rival_reference(evidence, groups), expected, atol=1e-10, rtol=0))

    def test_weight_identity_and_constant_evidence_neutrality(self):
        groups = layout([3, 2])
        evidence = torch.randn(4, 16, 2, 3, dtype=torch.float64).masked_fill(~groups.valid[None, None], -torch.inf)
        weights = groups.valid.double()[None].expand(4, -1, -1).clone()
        self.assertTrue(torch.equal(normalized_delta(evidence.log_softmax(-1), weights, groups), torch.zeros(4, 16, 2)))
        weights[:, 0] = torch.tensor([1., .2, .7])
        weights[:, 1, :2] = torch.tensor([1., .3])
        constant = torch.ones_like(evidence).masked_fill(~groups.valid[None, None], -torch.inf)
        self.assertLess(float(normalized_delta(constant.log_softmax(-1), weights, groups).abs().max()), 1e-12)

    def test_weights_match_independent_normalized_weighted_lme(self):
        groups = layout([4, 3])
        evidence = torch.randn(2, 7, 2, 4, dtype=torch.float64).masked_fill(~groups.valid[None, None], -torch.inf)
        weights = torch.rand(2, 2, 4, dtype=torch.float64).masked_fill(~groups.valid[None], 0.)
        actual = normalized_delta(evidence.log_softmax(-1), weights, groups)
        expected = torch.empty_like(actual)
        for n in range(2):
            for t in range(7):
                for c, k in enumerate((4, 3)):
                    z, w = evidence[n, t, c, :k], weights[n, c, :k]
                    expected[n, t, c] = (z + w.log()).logsumexp(0) - w.sum().log() - z.logsumexp(0) + torch.tensor(k).double().log()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))

    def test_canonical_invalid_and_permutation_spectrum(self):
        groups = layout([20, 20, 20])
        valid = torch.tensor([True, False])
        weights = weights_from_risk(torch.rand(2, 3, 20), groups, valid)
        self.assertTrue(torch.equal(weights[:, groups.canonical], torch.ones(2, 3)))
        self.assertTrue(torch.equal(weights[1], torch.ones(3, 20)))
        shuffled = weights.gather(-1, alias_permutation(groups)[None].expand_as(weights))
        self.assertTrue(torch.equal(shuffled.sort(-1).values, weights.sort(-1).values))
        self.assertTrue(torch.equal(shuffled[:, groups.canonical], torch.ones(2, 3)))

    def test_contradiction_requires_both_local_readouts(self):
        groups = layout([3, 3])
        wrong = torch.tensor([[[0., 0., 0.], [5., 5., 5.]]])
        right = wrong.flip(1)
        self.assertTrue(torch.equal(local_contradiction(wrong, right, groups), torch.zeros_like(wrong)))
        self.assertGreater(float(local_contradiction(wrong, wrong, groups)[0, 0, 1]), .99)
        equal = torch.ones_like(wrong)
        self.assertLess(float(positive_margin(equal, groups).abs().max()), 1e-12)


if __name__ == '__main__':
    unittest.main()
