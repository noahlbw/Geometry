import unittest
import torch
from dinotool.evidence_envelope import envelope_coupled


class EnvelopeTests(unittest.TestCase):
    def test_class_bounds_and_finite_signed_transport(self):
        a=torch.tensor([[6.,0.,1.],[0.,5.,-1.]])
        b=torch.tensor([[0.,4.,1.],[2.,0.,1.]])
        h=torch.tensor([[.4,-.1],[-.1,.4]])
        z=envelope_coupled(a,b,h,2.)
        lower=torch.minimum(a.double().log_softmax(-1),b.double().log_softmax(-1))
        upper=torch.maximum(a.double().log_softmax(-1),b.double().log_softmax(-1))
        self.assertTrue(bool(torch.isfinite(z).all()))
        self.assertTrue(bool(((z>=lower)&(z<=upper)).all()))

    def test_identical_observers_keep_probability(self):
        a=torch.randn(7,5,dtype=torch.float64)
        z=envelope_coupled(a,a,torch.eye(7),2.)
        torch.testing.assert_close(z.softmax(-1),a.softmax(-1))

    def test_independent_row_gauge_and_class_permutation(self):
        torch.manual_seed(7)
        a,b,h=torch.randn(7,5),torch.randn(7,5),torch.randn(7,7)*.2
        z=envelope_coupled(a,b,h,1.)
        shifted=envelope_coupled(a.double()+torch.arange(7)[:,None],b.double()-9,h,1.)
        torch.testing.assert_close(z,shifted)
        perm=torch.tensor([3,1,4,0,2])
        torch.testing.assert_close(envelope_coupled(a[:,perm],b[:,perm],h,1.),z[:,perm])

    def test_not_a_new_class_prior(self):
        a=torch.zeros(3,6)
        z=envelope_coupled(a,a,torch.randn(3,3),.5)
        torch.testing.assert_close(z.softmax(-1),torch.full((3,6),1/6,dtype=torch.float64))


if __name__=='__main__':
    unittest.main()
