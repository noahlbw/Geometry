"""Uniform transfer engine must preserve the inherited local reduction."""
from types import SimpleNamespace
import unittest

import torch

from dinotool import local_role_alias as role
from dinotool.local_role_transfer import METHODS, uniform_plan
from dinotool.geometry_readout_trace import alias_class_scores


class TransferTest(unittest.TestCase):
    def test_uniform_comparator_is_exact_at_each_frozen_scale(self):
        bank = SimpleNamespace(class_count=2,
            alias_names=tuple('a'+str(i) for i in range(20))+tuple('b'+str(i) for i in range(20)),
            parent_indices=torch.arange(2).repeat_interleave(20),canonical_mask=torch.arange(40)%20==0)
        weighted = role.prepare_plan(bank, [[0]*13+[1]*7]*2, [['a0','a13'],['b0','b13']])
        uniform = uniform_plan(weighted)
        self.assertTrue(torch.equal(weighted.log_prior[:,0], torch.full((2,),-.6931471805599453)))
        torch.manual_seed(20261009)
        x = torch.randn(1024,40)
        for scale in (.07,.1):
            expected = alias_class_scores(x,bank.parent_indices,2)/scale
            self.assertTrue(torch.equal(role.local_scores(x,uniform,role.PRIMARY,scale),expected))
        self.assertEqual(len(METHODS),3)
        self.assertTrue(all('Shuffle' not in name for name in METHODS))


if __name__=='__main__':
    unittest.main()
