"""Check the exact distinction between class calibration and alias identity."""
import math
from types import SimpleNamespace
import unittest

import torch

from dinotool.family_mass_alias import prepare_plan, reduce_routes


def plan(labels):
    bank = SimpleNamespace(class_count=1, parent_indices=torch.zeros(20, dtype=torch.long),
                           canonical_mask=torch.tensor([True]+[False]*19))
    return prepare_plan(bank, [labels])


class FamilyMassTest(unittest.TestCase):
    def test_sum_matches_family_exponential_means(self):
        labels = [0]*13+[1]*5+[2]*2
        p = plan(labels); e = torch.arange(80, dtype=torch.float64).reshape(20,4)/20
        for tau in (.5, 1., 2.):
            expected = torch.stack([(tau*e[torch.tensor(labels)==g]).exp().mean(0)
                                    for g in range(3)]).sum(0).log()/tau
            got = reduce_routes(e, p, 0, ('sum',), tau)['sum']
            self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_factorial_interaction_preserves_existing_relative_alias_signal(self):
        p=plan([0]*13+[1]*5+[2]*2); e=torch.randn(20,7,dtype=torch.float64)
        f=reduce_routes(e,p,0,('base','uniform','mean','sum'),.7)
        self.assertTrue(torch.allclose(f['sum']-f['uniform'],f['mean']-f['base']))
        self.assertTrue(torch.allclose(f['sum']-f['mean'],
                        torch.full((7,),math.log(3/20)/.7,dtype=torch.float64)))

    def test_equal_evidence_has_log_family_count(self):
        for labels in ([0]*20, list(range(20)), [i%3 for i in range(20)]):
            p=plan(labels); e=torch.full((20,5),2.5,dtype=torch.float64)
            value=reduce_routes(e,p,0,('sum',),1.)['sum']
            self.assertTrue(torch.allclose(value,torch.full((5,),2.5+math.log(len(set(labels))),dtype=torch.float64),atol=1e-6))

    def test_uniform_families_do_not_create_word_identity_advantage(self):
        e=torch.randn(20,9)
        for labels in ([0]*20,[i//5 for i in range(20)],list(range(20))):
            f=reduce_routes(e,plan(labels),0,('base','uniform','mean','sum','shuffle'),1.)
            self.assertTrue(torch.equal(f['sum'],f['uniform']))
            self.assertTrue(torch.equal(f['sum'],f['shuffle']))
            self.assertTrue(torch.equal(f['base'],f['mean']))

    def test_shuffle_preserves_class_mass_and_canonical_weight(self):
        p=plan([0]*13+[1]*5+[2]*2)
        self.assertEqual(float(p.log_weights[0][0]),float(p.shuffled_log_weights[0][0]))
        self.assertTrue(torch.equal(p.log_weights[0].sort().values,p.shuffled_log_weights[0].sort().values))
        self.assertFalse(torch.equal(p.log_weights[0],p.shuffled_log_weights[0]))


if __name__=='__main__': unittest.main()
