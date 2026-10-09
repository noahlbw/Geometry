import math
from types import SimpleNamespace
import unittest

import torch

from dinotool.generation_family_alias import prepare_plan, quotient


class FamilyQuotientTests(unittest.TestCase):
    def test_equal_response_has_no_family_count_prior(self):
        e = torch.full((20,7),2.5,dtype=torch.float64)
        for labels in ([0]*20,list(range(20)),[0]*7+[1]*7+[2]*6):
            torch.testing.assert_close(quotient(e,torch.tensor(labels)),
                torch.full((7,),2.5+math.log(20),dtype=e.dtype),rtol=0,atol=1e-14)

    def test_singleton_families_replay_lse(self):
        e = torch.randn(20,9,dtype=torch.float64,generator=torch.Generator().manual_seed(41))
        torch.testing.assert_close(quotient(e,torch.arange(20)),e.logsumexp(0),rtol=0,atol=1e-14)

    def test_one_family_is_mean_not_lme(self):
        e = torch.arange(20,dtype=torch.float64)[:,None]
        q = quotient(e,torch.zeros(20,dtype=torch.long))
        torch.testing.assert_close(q,e.mean(0)+math.log(20),rtol=0,atol=1e-14)
        self.assertGreater(float(e.logsumexp(0)-q),5.)

    def test_equal_size_group_means_remove_jensen_inflation(self):
        e = torch.randn(20,6,dtype=torch.float64,generator=torch.Generator().manual_seed(4))
        labels = torch.arange(20)//5
        self.assertTrue(bool((quotient(e,labels)<=e.logsumexp(0)+1e-14).all()))
        grouped_constant = e[::5].repeat_interleave(5,0)
        torch.testing.assert_close(quotient(grouped_constant,labels),grouped_constant.logsumexp(0),
                                   rtol=0,atol=1e-14)

    def test_shuffle_preserves_sizes_and_canonical_assignment(self):
        bank = SimpleNamespace(parent_indices=torch.arange(2).repeat_interleave(20),
            canonical_mask=torch.arange(40)%20==0,class_count=2)
        groups = [[i%3 for i in range(20)],[i%4 for i in range(20)]]
        plan = prepare_plan(bank,groups)
        for original,control in zip(plan.family,plan.shuffled):
            self.assertTrue(torch.equal(torch.bincount(original),torch.bincount(control)))
            self.assertEqual(int(original[0]),int(control[0]))
            self.assertFalse(torch.equal(original,control))


if __name__ == '__main__':
    unittest.main()
