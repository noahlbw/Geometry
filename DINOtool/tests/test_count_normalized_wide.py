import unittest
import torch
from dinotool.count_normalized_wide import CountNormalizedWide


def inherited_group(raw,salience,tau):
    weights=salience.softmax(-1)
    return (tau*raw*(len(salience)*weights)).logsumexp(-1)/tau


class CountWideTests(unittest.TestCase):
    def test_whole_group_duplicate_invariance(self):
        raw=torch.tensor([[.1,.7,-.2],[.4,.3,.1]],dtype=torch.float64)*40
        salience=torch.tensor([.1,.7,-.2],dtype=torch.float64)
        for tau in (1.,2.2,4.):
            base=inherited_group(raw,salience,tau)
            doubled=inherited_group(raw.repeat(1,2),salience.repeat(2),tau)
            torch.testing.assert_close(doubled-base,torch.full_like(base,torch.log(torch.tensor(2.)).double()/tau),rtol=1e-6,atol=1e-6)
            a=CountNormalizedWide(torch.zeros(3,dtype=torch.long),1).apply(base[None],tau)
            b=CountNormalizedWide(torch.zeros(6,dtype=torch.long),1).apply(doubled[None],tau)
            torch.testing.assert_close(a,b)

    def test_equal_counts_only_common_probability_offset(self):
        parents=torch.tensor([0,0,1,1,2,2]);raw=torch.randn(3,4,5,dtype=torch.float64)
        ordinary=CountNormalizedWide(parents,3).apply(raw,1.)
        torch.testing.assert_close(ordinary.softmax(0),raw.softmax(0))
        exact=CountNormalizedWide(parents,3,keep_equal_count_gauge=True).apply(raw,1.)
        self.assertIs(exact,raw)

    def test_external_residual_requires_common_offset_removal(self):
        parents=torch.tensor([0,0,1,1]);raw=torch.tensor([[3.,2.],[1.,5.]])
        changed=CountNormalizedWide(parents,2,keep_equal_count_gauge=False).apply(raw,1.)
        torch.testing.assert_close(changed,raw-torch.tensor(2.).log())

    def test_unequal_counts_and_class_order(self):
        parents=torch.tensor([0,1,1,1,2,2]);raw=torch.zeros(3,2,2,dtype=torch.float64)
        out=CountNormalizedWide(parents,3).apply(raw,2.)
        wanted=-torch.tensor([1.,3.,2.],dtype=torch.float64).log()/2
        torch.testing.assert_close(out,wanted[:,None,None].expand_as(raw))
        with self.assertRaises(ValueError):CountNormalizedWide(torch.tensor([0,2]),3)


if __name__=='__main__':unittest.main()
