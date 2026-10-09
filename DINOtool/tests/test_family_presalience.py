"""Check the diagnostic factor identities and true membership shuffle."""
import unittest
import torch
from dinotool.family_presalience import amplification,shuffled_families


class PreSalienceTest(unittest.TestCase):
    def test_uniform_salience_makes_pre_factor_exactly_neutral(self):
        for labels in ([0]*20,[0]*7+[1]*5+[2]*8,list(range(20))):
            slot,family=amplification(torch.zeros(20),labels,1.)
            self.assertTrue(torch.allclose(slot,torch.ones(20),atol=1e-7))
            self.assertTrue(torch.equal(slot,family))

    def test_group_normalization_counts_each_family_once(self):
        labels=[0]*12+[1]*8;salience=torch.tensor([2.]*12+[0.]*8)
        _,h=amplification(salience,labels,1.)
        self.assertTrue(torch.allclose(h[[0,12]],2*torch.tensor([2.,0.]).softmax(0)))

    def test_shuffle_changes_membership_but_preserves_every_family_size(self):
        labels=[0]*7+[1]*5+[2]*8;shuffled=shuffled_families([labels])[0]
        self.assertEqual(sorted(labels),sorted(shuffled));self.assertNotEqual(labels,shuffled)
        self.assertEqual(shuffled,shuffled_families([labels])[0])


if __name__=='__main__':unittest.main()
