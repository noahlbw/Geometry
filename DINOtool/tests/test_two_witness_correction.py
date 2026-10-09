import unittest
import torch
from dinotool.development_readout import coupled
from dinotool.two_witness_correction import two_witness_coupled


class TwoWitnessTests(unittest.TestCase):
    def setUp(self):
        self.a=torch.tensor([[2.,0.],[2.,0.]])
        self.b=torch.tensor([[1.,0.],[10.,0.]])
        self.h=torch.tensor([[0.,-1.],[0.,0.]])

    def test_two_opposed_witnesses_revert_foreground(self):
        self.assertEqual(coupled(self.a,self.b,self.h,1.)[0].argmax().item(),1)
        result=two_witness_coupled(self.a,self.b,self.h,1.)
        torch.testing.assert_close(result[0],self.a[0].double())

    def test_background_transition_exempt(self):
        for background in (0,1):
            self.assertTrue(torch.equal(two_witness_coupled(self.a,self.b,self.h,1.,background),coupled(self.a,self.b,self.h,1.)))

    def test_one_supporting_witness_retains_proposal(self):
        b=self.b.clone();b[0]=torch.tensor([0.,10.])
        self.assertTrue(torch.equal(two_witness_coupled(self.a,b,self.h,1.),coupled(self.a,b,self.h,1.)))

    def test_isolated_rows_and_equal_endpoints_replay(self):
        for h,b in ((torch.eye(2),self.b),(self.h,self.a)):
            self.assertTrue(torch.equal(two_witness_coupled(self.a,b,h,1.),coupled(self.a,b,h,1.)))
