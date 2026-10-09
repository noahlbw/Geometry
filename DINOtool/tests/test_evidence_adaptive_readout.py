import unittest
import torch
from types import SimpleNamespace
from dinotool.evidence_adaptive_readout import competition_scores,adaptive_coupled,residual_prediction


class EvidenceReadoutTests(unittest.TestCase):
    def test_duplicate_alias_does_not_increase_class_evidence(self):
        features=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        bank=SimpleNamespace(features=features,parent_indices=torch.tensor([0,0,1]),
            canonical_mask=torch.tensor([True,False,True]),class_count=2)
        alias=torch.tensor([[.8,.8,.2],[.1,.1,.9]])
        torch.testing.assert_close(competition_scores(alias,bank),alias[:,[0,2]])

    def test_equal_branches_preserve_scores(self):
        local=torch.randn(8,3)
        result,gain=adaptive_coupled(local,local,torch.eye(8)*.5,torch.eye(8))
        torch.testing.assert_close(result,local.double())
        torch.testing.assert_close(gain,torch.ones_like(gain))

    def test_geometry_inconsistency_reduces_trust(self):
        local=torch.tensor([[5.,0.],[0.,5.]])
        broad=torch.tensor([[5.,0.],[5.,0.]])
        _,gain=adaptive_coupled(local,broad,torch.eye(2),torch.ones(2,2)/2)
        self.assertTrue(bool((gain>1).all()))
        self.assertTrue(bool((gain<=2).all()))

    def test_no_background_no_rejection(self):
        p=torch.rand(3,8,8).softmax(0)
        prediction,record=residual_prediction(p,None)
        torch.testing.assert_close(prediction,p.argmax(0))
        self.assertFalse(record['rejection'])

    def test_residual_two_confidence_modes(self):
        confidence=torch.cat([torch.full((64,32),.1),torch.full((64,32),.9)],1)
        p=torch.stack((1-confidence,confidence))
        prediction,record=residual_prediction(p,0)
        self.assertTrue(record['rejection'])
        self.assertTrue(bool((prediction[:,:32]==0).all()))
        self.assertTrue(bool((prediction[:,32:]==1).all()))


if __name__=='__main__':
    unittest.main()
