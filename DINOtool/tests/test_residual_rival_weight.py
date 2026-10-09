import unittest
import torch
from dinotool.residual_rival_weight import ResidualRivalWeight


class ResidualRivalTests(unittest.TestCase):
    def test_bounded_suppression_no_boost(self):
        torch.manual_seed(17)
        reader=ResidualRivalWeight(torch.randn(9,12),torch.randn(4,12))
        score=torch.randn(19,9)
        result=reader.score(score,torch.arange(19)%4)
        maximum=score.amax(-1)
        self.assertTrue(bool((result<=maximum+1e-6).all()))
        self.assertTrue(bool((result>=maximum-reader.maximum_raw_score_loss-1e-6).all()))

    def test_rival_changes_weight_without_removing_queries(self):
        text=torch.tensor([[1.,0.],[0.,1.]])
        anchors=torch.tensor([[1.,0.],[-1.,0.]])
        reader=ResidualRivalWeight(text,anchors)
        score=torch.tensor([[.3,-.1],[.3,-.1]])
        result=reader.score(score,torch.tensor([0,1]))
        self.assertLess(float(result[0]),float(result[1]))
        self.assertEqual(reader.cross.shape,(2,2))


if __name__=='__main__':
    unittest.main()
