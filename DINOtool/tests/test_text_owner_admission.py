import unittest
import torch
from dinotool.text_owner_admission import admit


class OwnerAdmissionTests(unittest.TestCase):
    def test_competitor_alias_rejected_canonical_retained(self):
        x=torch.tensor([[1.,0.],[0.,1.],[0.,1.]])
        keep,_=admit(x,x,torch.tensor([0,0,1]),torch.tensor([True,False,True]),2)
        self.assertEqual(keep.tolist(),[True,False,True])

    def test_both_representations_required_and_background_preserved(self):
        x=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        y=x.clone();y[1]=torch.tensor([0.,1.])
        parents=torch.tensor([0,0,1]);canonical=torch.tensor([True,False,True])
        self.assertEqual(admit(x,y,parents,canonical,2)[0].tolist(),[True,False,True])
        self.assertEqual(admit(x,y,parents,canonical,2,background=0)[0].tolist(),[True,True,True])

    def test_tie_rejected_no_tuned_margin(self):
        x=torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
        self.assertFalse(admit(x,x,torch.tensor([0,0,1]),torch.tensor([True,False,True]),2)[0][1])
