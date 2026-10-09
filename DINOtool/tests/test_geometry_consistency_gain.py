import unittest
import torch
from dinotool.geometry_consistency_gain import consistency_coupled


class ConsistencyGainTests(unittest.TestCase):
    def test_identical_views_preserve_scores(self):
        a=torch.randn(5,4,dtype=torch.double)
        torch.testing.assert_close(consistency_coupled(a,a,torch.randn(5,5),2.),a)

    def test_diagonal_geometry_retains_frozen_gain(self):
        a,b=torch.randn(5,4),torch.randn(5,4)
        h=torch.eye(5)*.4
        torch.testing.assert_close(consistency_coupled(a,b,h,.5),a.double()+.5*h.double()@(b.double()-a.double()))

    def test_consistent_broad_gets_twice_gain(self):
        a=torch.tensor([[4.,0.],[0.,4.]])
        b=torch.zeros_like(a)
        h=torch.tensor([[.4,-.1],[-.1,.4]])
        expected=a.double()+2*h.double()@(b.double()-a.double())
        torch.testing.assert_close(consistency_coupled(a,b,h,1.),expected)

    def test_consistent_local_rejects_inconsistent_broad(self):
        a=torch.zeros(2,2)
        b=torch.tensor([[4.,0.],[0.,4.]])
        h=torch.tensor([[.4,-.1],[-.1,.4]])
        torch.testing.assert_close(consistency_coupled(a,b,h,1.),a.double())

    def test_class_and_patch_permutation(self):
        torch.manual_seed(13)
        a,b,h=torch.randn(5,4),torch.randn(5,4),torch.randn(5,5)
        z=consistency_coupled(a,b,h,2.)
        cp=torch.tensor([2,0,3,1]);pp=torch.tensor([2,4,1,0,3])
        torch.testing.assert_close(consistency_coupled(a[:,cp],b[:,cp],h,2.),z[:,cp])
        torch.testing.assert_close(consistency_coupled(a[pp],b[pp],h[pp][:,pp],2.),z[pp])


if __name__=='__main__':unittest.main()
