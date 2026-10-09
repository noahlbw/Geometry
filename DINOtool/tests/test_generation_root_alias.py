import math
from types import SimpleNamespace
import unittest

import torch

from dinotool.generation_root_alias import (BASE,PRIMARY,QUOTIENT,SHUFFLE,CANONICAL,
    METHODS,RootPlan,class_fields,prepare_plan)


class RootRepresentativeTests(unittest.TestCase):
    def plan(self):
        labels=torch.arange(20)%2
        return RootPlan((torch.arange(20),),(0,),(labels,),
            (torch.tensor([0,1]),),(torch.tensor([4,5]),),(2,))

    def test_equal_responses_remove_group_count_prior(self):
        e=torch.full((20,4),3.,dtype=torch.float64)
        fields=class_fields(e,self.plan(),0,METHODS,1.)
        for value in fields.values():
            torch.testing.assert_close(value,torch.full((4,),3.+math.log(20),dtype=e.dtype),rtol=0,atol=1e-14)

    def test_roles_differ_from_family_mean(self):
        e=torch.zeros(20,3,dtype=torch.float64);e[0]=2.;e[1]=4.
        fields=class_fields(e,self.plan(),0,METHODS,1.)
        self.assertGreater(float((fields[PRIMARY]-fields[QUOTIENT]).min()),2.)
        self.assertGreater(float((fields[PRIMARY]-fields[SHUFFLE]).min()),2.)
        self.assertGreater(float((fields[PRIMARY]-fields[CANONICAL]).min()),1.)

    def test_no_provenance_fallback_is_bitwise_original(self):
        bank=SimpleNamespace(parent_indices=torch.zeros(20,dtype=torch.long),class_count=1,
            canonical_mask=torch.arange(20)==0)
        plan=prepare_plan(bank)
        e=torch.randn(20,9,generator=torch.Generator().manual_seed(11))
        fields=class_fields(e,plan,0,METHODS,1.)
        for name in (PRIMARY,QUOTIENT,SHUFFLE):self.assertTrue(torch.equal(fields[BASE],fields[name]))

    def test_exact_existing_roots_and_in_family_control(self):
        words=['thing','object']+[('visible thing '+str(i)) if i%2==0 else ('visible object '+str(i)) for i in range(18)]
        bank=SimpleNamespace(parent_indices=torch.zeros(20,dtype=torch.long),class_count=1,
            canonical_mask=torch.arange(20)==0,class_names=('thing',),alias_names=tuple(words))
        provenance=dict(class_names=['thing'],aliases=words,roots=[['thing','object']],family_ids=[[i%2 for i in range(20)]])
        plan=prepare_plan(bank,provenance)
        self.assertEqual(plan.roots[0].tolist(),[0,1])
        for g,slot in enumerate(plan.shuffled[0]):self.assertEqual(int(plan.labels[0][slot]),g)
        broken={**provenance,'roots':[['missing','object']]}
        with self.assertRaises(ValueError):prepare_plan(bank,broken)


if __name__=='__main__':unittest.main()
