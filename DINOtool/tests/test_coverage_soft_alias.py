import math
import unittest
from types import SimpleNamespace
import torch
from dinotool.coverage_soft_alias import coverage_scores
from dinotool.geometry_readout_trace import alias_class_scores


class CoverageAliasTests(unittest.TestCase):
    def test_lower_bound_with_wrong_word_high_response(self):
        # Class0 contains a rival-looking alias with high visual response.
        bank=SimpleNamespace(features=torch.tensor([[1.,0.],[0.,1.],[0.,1.]]),
            parent_indices=torch.tensor([0,0,1]),class_count=2)
        alias=torch.tensor([[.1,.9,.8],[.9,.1,.8]])
        base=alias_class_scores(alias,bank.parent_indices,2)
        new=coverage_scores(alias,bank)
        self.assertTrue(bool((new>=base-.07*math.log(2)-1e-6).all()))
        self.assertLess(float(new[0,0]),float(base[0,0]))

    def test_identical_aliases_preserve_scores(self):
        bank=SimpleNamespace(features=torch.tensor([[1.,0.],[1.,0.],[0.,1.]]),
            parent_indices=torch.tensor([0,0,1]),class_count=2)
        alias=torch.tensor([[.8,.8,.2],[.1,.1,.9]])
        torch.testing.assert_close(coverage_scores(alias,bank),alias[:,[0,2]])

    def test_unequal_counts_and_class_order(self):
        torch.manual_seed(3)
        features=torch.randn(8,12)
        parents=torch.tensor([0,0,0,1,2,2,2,2])
        alias=torch.randn(20,8)
        bank=SimpleNamespace(features=features,parent_indices=parents,class_count=3)
        values=coverage_scores(alias,bank)
        renamed=SimpleNamespace(features=features,parent_indices=2-parents,class_count=3)
        torch.testing.assert_close(coverage_scores(alias,renamed),values.flip(-1))
        base=alias_class_scores(alias,parents,3)
        self.assertTrue(bool((values>=base-.07*math.log(2)-1e-5).all()))


if __name__=='__main__':
    unittest.main()
