import unittest
from types import SimpleNamespace
import torch
from dinotool.bounded_wide_rival import BoundedWideRival


class WideRivalTests(unittest.TestCase):
    def reader(self,background=None):
        q=SimpleNamespace(features=torch.tensor([[[1.,0.]],[[1.,1.]],[[0.,1.]],[[0.,1.]],[[1.,-1.]]]),
            parents=torch.tensor([0,0,1,1,2]),class_names=('a','b','residual'))
        return BoundedWideRival(q,background)

    def test_bound_and_residual_identity(self):
        r=self.reader(2);torch.manual_seed(4)
        groups=[torch.randn(2,3,3),torch.randn(2,3,3),torch.randn(1,3,3)]
        base=torch.stack([torch.logsumexp(x*2,0)/2 for x in groups])
        z=r.reduce(groups,2.)
        self.assertTrue(bool((z<=base+1e-6).all()))
        self.assertTrue(bool((z>=base-torch.log(torch.tensor(2.))/2-1e-6).all()))
        self.assertTrue(torch.equal(z[2],base[2]))

    def test_single_foreground_has_no_rival(self):
        q=SimpleNamespace(features=torch.eye(2)[:,None],parents=torch.arange(2),class_names=('a','residual'))
        r=BoundedWideRival(q,1);groups=[torch.randn(1,2,2),torch.randn(1,2,2)]
        torch.testing.assert_close(r.reduce(groups,1.),torch.stack([torch.logsumexp(x,0) for x in groups]))

    def test_ambiguous_alias_has_more_attenuation(self):
        r=self.reader(2)
        r.semantic=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[0.,1.,0.],[0.,0.,1.]])
        groups=[torch.zeros(2,1,1),torch.ones(2,1,1)*3,torch.zeros(1,1,1)]
        z=r.reduce(groups,1.)
        self.assertLess(float(z[0]-torch.log(torch.tensor(2.))),float(z[1]-3-torch.log(torch.tensor(2.))))

    def test_alias_order_invariance(self):
        r=self.reader(2);groups=[torch.randn(2,2,2),torch.randn(2,2,2),torch.randn(1,2,2)]
        z=r.reduce(groups,1.)
        r.semantic=r.semantic[[1,0,2,3,4]].clone()
        groups[0]=groups[0].flip(0)
        torch.testing.assert_close(r.reduce(groups,1.),z)


if __name__=='__main__':unittest.main()
