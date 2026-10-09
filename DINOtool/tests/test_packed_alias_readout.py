import unittest
from types import SimpleNamespace
import torch
from dinotool.packed_alias_readout import PackedAliasReadout
from dinotool.coverage_soft_alias import coverage_scores
from dinotool.geometry_readout_trace import alias_class_scores


class PackedAliasTests(unittest.TestCase):
    def test_unequal_groups_equal_scalar_reference(self):
        torch.manual_seed(12)
        parents=torch.tensor([0,0,1,2,2,2,2,3,3,3])
        bank=SimpleNamespace(features=torch.randn(10,16),parent_indices=parents,class_count=4)
        alias=torch.randn(31,10)
        packed=PackedAliasReadout(bank)
        torch.testing.assert_close(packed.uniform(alias),alias_class_scores(alias,parents,4),rtol=1e-6,atol=1e-6)
        self.assertTrue(torch.equal(packed.uniform_reference(alias),alias_class_scores(alias,parents,4)))
        torch.testing.assert_close(packed.coverage(alias),coverage_scores(alias,bank),rtol=1e-6,atol=1e-6)

    def test_noncontiguous_alias_order_and_singletons(self):
        torch.manual_seed(18)
        parents=torch.tensor([2,0,2,1,0])
        bank=SimpleNamespace(features=torch.randn(5,8),parent_indices=parents,class_count=3)
        alias=torch.rand(16,5)
        packed=PackedAliasReadout(bank)
        torch.testing.assert_close(packed.coverage(alias),coverage_scores(alias,bank),rtol=1e-6,atol=1e-6)

    def test_single_class_and_positive_temperature(self):
        bank=SimpleNamespace(features=torch.randn(3,8),parent_indices=torch.zeros(3,dtype=torch.long),class_count=1)
        alias=torch.rand(5,3)
        packed=PackedAliasReadout(bank)
        torch.testing.assert_close(packed.coverage(alias),packed.uniform(alias))
        with self.assertRaises(ValueError):
            PackedAliasReadout(bank,temperature=0.)


if __name__=='__main__':
    unittest.main()
