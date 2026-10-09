import unittest
import torch
from dinotool.geometry_support_audit import relation_support


class GeometrySupportTests(unittest.TestCase):
    def test_offdiagonal_only_and_sign_invariance(self):
        a=torch.tensor([[8.,0.],[0.,8.]])
        h=torch.tensor([[100.,-2.],[3.,200.]])
        result=relation_support(a,a,h)
        torch.testing.assert_close(result,a.double().softmax(-1).flip(0))
        torch.testing.assert_close(result,relation_support(a,a,-h))

    def test_isolated_patch_is_class_neutral(self):
        a=torch.tensor([[8.,0.],[0.,8.]])
        self.assertTrue(torch.equal(relation_support(a,a,torch.eye(2)),torch.zeros(2,2,dtype=torch.float64)))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):relation_support(torch.tensor([[float('nan'),0.]]),torch.zeros(1,2),torch.eye(1))
