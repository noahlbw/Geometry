"""Check the signal, shared mass, and exact correspondence controls."""
import unittest
import torch
from dinotool.region_residual_alias import (residual_correlation,log_weights,
    region_support,alias_derangements,aligned_raw,ANCHORS)


class RegionResidualTest(unittest.TestCase):
    def test_same_alias_positive_and_negative_regions_are_distinguished(self):
        torch.manual_seed(1); local=torch.randn(12,3,20)
        self.assertTrue(torch.allclose(residual_correlation(local,local),torch.ones(3,20),atol=1e-5))
        self.assertTrue(torch.allclose(residual_correlation(local,-local),-torch.ones(3,20),atol=1e-5))

    def test_class_common_response_and_per_alias_global_offsets_cancel(self):
        torch.manual_seed(2); l=torch.randn(9,2,20); w=torch.randn(9,2,20)
        q=residual_correlation(l,w)
        altered=residual_correlation(l+torch.randn(9,2,1)+torch.randn(1,2,20),
                                     w+torch.randn(9,2,1)+torch.randn(1,2,20))
        self.assertTrue(torch.allclose(q,altered,atol=1e-5))

    def test_no_variation_or_one_region_is_neutral(self):
        self.assertTrue(torch.equal(residual_correlation(torch.zeros(3,2,20),torch.ones(3,2,20)),torch.zeros(2,20)))
        self.assertTrue(torch.equal(residual_correlation(torch.randn(1,2,20),torch.randn(1,2,20)),torch.zeros(2,20)))

    def test_all_aliases_positive_and_same_declared_class_mass(self):
        q=torch.linspace(-1,1,60).reshape(3,20); masses=(1,7,20)
        weights=log_weights(q,masses).exp()
        self.assertTrue(bool((weights>0).all()))
        self.assertTrue(torch.allclose(weights.sum(-1),torch.tensor(masses).float(),atol=2e-6))
        neutral=log_weights(torch.zeros_like(q),masses).exp()
        self.assertTrue(torch.allclose(neutral,torch.tensor(masses).float()[:,None].expand(3,20)/20))

    def test_alias_null_has_no_fixed_points_and_preserves_exact_weight_multisets(self):
        p=alias_derangements(5,'cpu'); identity=torch.arange(20)[None]
        self.assertTrue(bool((p!=identity).all()))
        self.assertTrue(torch.equal(p,alias_derangements(5,'cpu')))
        w=log_weights(torch.randn(5,20),(1,2,3,4,5)); shuffled=w.gather(1,p)
        self.assertTrue(torch.equal(w.sort(-1).values,shuffled.sort(-1).values))

    def test_support_excludes_self_padding_and_has_fixed_budget(self):
        relation=torch.ones(1024,1024); valid=torch.ones(1024,dtype=torch.bool); valid[900:]=False
        a=region_support(relation,valid)
        anchors=[i for i in ANCHORS if valid[i]]
        self.assertEqual(a.shape,(len(anchors),1024))
        self.assertTrue(bool((a[:,900:]==0).all()))
        self.assertTrue(bool((a[torch.arange(len(anchors)),torch.tensor(anchors)]==0).all()))
        self.assertTrue(bool(((a>0).sum(-1)<=32).all()))
        self.assertTrue(torch.allclose(a.sum(-1),torch.ones(len(anchors))))
        self.assertEqual(len(region_support(torch.zeros_like(relation),valid)),0)

    def test_physical_sampling_uses_crop_offsets_and_coverage_average(self):
        source=dict(size=(512,512),wide_size=(336,336))
        tile=dict(top=0,left=0)
        a=dict(raw=torch.full((2,21,21),2.),top=0,left=0,ah=336,aw=336)
        b=dict(raw=torch.full((2,21,21),6.),top=0,left=168,ah=336,aw=168)
        sampled=aligned_raw((a,b),source,tile).reshape(32,32,2)
        self.assertTrue(torch.equal(sampled[:,:16],torch.full((32,16,2),2.)))
        self.assertTrue(torch.equal(sampled[:,16:],torch.full((32,16,2),4.)))


if __name__=='__main__': unittest.main()
