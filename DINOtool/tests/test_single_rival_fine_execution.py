import unittest

import torch

from dinotool.single_rival_fine_execution import vector_stencil,tensor_fields
from dinotool import single_rival_fine_alias as rule
from types import SimpleNamespace
from dinotool.stratified_soft_alias import WideCrop,crop_stencil


class StencilTests(unittest.TestCase):
    def test_bitwise_all16_neighbors_on_both_grids(self):
        torch.manual_seed(8)
        for grid,side,shape in ((21,336,(336,448)),(21,336,(336,672)),(32,256,(512,512))):
            count = torch.randint(1,5,shape).float()
            coords = torch.randn(1024,2)*500+250
            for top,left,ah,aw in ((0,0,side,side),(112,0,123,side),(0,112,side,89)):
                crop = WideCrop(None,None,top,left,ah,aw,grid,side)
                a,b = crop_stencil(crop,count,coords,(896,640))
                x,y = vector_stencil(crop,count,coords,(896,640))
                self.assertTrue(torch.equal(a,x))
                self.assertTrue(torch.equal(b,y))

    def test_tensor_core_exact_frozen_patch_logits(self):
        torch.manual_seed(8)
        plan = rule.prepare_plan(SimpleNamespace(class_count=3,
            parent_indices=torch.arange(3).repeat_interleave(20),
            canonical_mask=torch.arange(60)%20==0))
        local,broad = torch.randn(5,3),torch.randn(5,3)
        h = torch.rand(5,5,dtype=torch.float64)
        h /= h.sum(-1,keepdim=True)
        valid = torch.tensor([True,True,True,True,False])
        ids = torch.randint(0,7,(5,16))
        coefficients = torch.rand(5,16)
        coefficients /= coefficients.sum(-1,keepdim=True)
        def crop():
            e = torch.randn(7,3,20)*2
            return rule.SparseCrop(e,e.logsumexp(-1),ids,coefficients)
        wc,fc = [crop(),crop()],[crop(),crop()]
        for methods in (rule.METHODS,(rule.PRIMARY,),(rule.NOALIAS,)):
            expected,_ = rule.tile_fields(local,h,broad,wc,fc,valid,plan,methods)
            actual = tensor_fields(local,h,broad,wc,fc,valid,plan,methods)
            for n,name in enumerate(methods):
                self.assertTrue(torch.equal(actual[n],expected[name]))


if __name__=='__main__':unittest.main()
