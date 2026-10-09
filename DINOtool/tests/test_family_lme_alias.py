"""Meaningful algebra checks for fixed-evidence expression mass correction."""
import math
import unittest
from types import SimpleNamespace

import torch

from dinotool.family_lme_alias import expression_log_weights, reduce_fixed_evidence, prepare_plan


class FamilyLMETest(unittest.TestCase):
    def test_equal_evidence_has_common_twenty_offset(self):
        for groups in ([0]*19+[1], list(range(20)), [i % 3 for i in range(20)]):
            weight = torch.tensor(expression_log_weights(groups), dtype=torch.float64)
            e = torch.full((20, 7), 2.5, dtype=torch.float64)
            self.assertTrue(torch.allclose(reduce_fixed_evidence(e, weight),
                                           torch.full((7,), 2.5+math.log(20), dtype=torch.float64)))

    def test_matches_nested_exponential_family_means(self):
        labels = [0]*13+[1]*5+[2]*2
        e = torch.arange(80, dtype=torch.float64).reshape(20, 4)/20
        weight = torch.tensor(expression_log_weights(labels), dtype=torch.float64)
        expected = torch.stack([e[torch.tensor(labels)==g].exp().mean(0) for g in range(3)]).mean(0).log()+math.log(20)
        self.assertTrue(torch.allclose(reduce_fixed_evidence(e, weight), expected))
        mean_before_exp = torch.stack([e[torch.tensor(labels)==g].mean(0) for g in range(3)]).logsumexp(0)-math.log(3)+math.log(20)
        self.assertFalse(torch.allclose(expected, mean_before_exp))

    def test_balanced_families_exactly_preserve_inherited_scores(self):
        e = torch.randn(20, 9)
        for labels in ([0]*20, list(range(20)), [i//5 for i in range(20)]):
            lw = torch.tensor(expression_log_weights(labels))
            self.assertTrue(torch.equal(reduce_fixed_evidence(e, lw), e.logsumexp(0)))

    def test_shuffle_preserves_canonical_weight_and_weight_distribution(self):
        bank = SimpleNamespace(class_count=1, parent_indices=torch.zeros(20, dtype=torch.long),
                               canonical_mask=torch.tensor([True]+[False]*19))
        plan = prepare_plan(bank, [[0]*13+[1]*5+[2]*2])
        self.assertEqual(plan.log_weights[0][0], plan.shuffled_log_weights[0][0])
        self.assertTrue(torch.equal(plan.log_weights[0].sort().values, plan.shuffled_log_weights[0].sort().values))
        self.assertFalse(torch.equal(plan.log_weights[0], plan.shuffled_log_weights[0]))

    def test_incomplete_or_noncontiguous_partitions_rejected(self):
        for labels in ([0]*19, [0]*19+[2]):
            with self.assertRaises(ValueError): expression_log_weights(labels)


if __name__ == '__main__': unittest.main()
