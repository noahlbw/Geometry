import unittest

import torch

from dinotool.single_rival_fine_alias import (BASE,NOALIAS,PRIMARY,POOLED,SHUFFLED,
    SparseCrop,reversal_risk,risk_control,sample_cache,attenuated_delta,tile_fields,prepare_plan,rival_indices)
from types import SimpleNamespace


class SparseFineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        c,k = 3,20
        self.plan = prepare_plan(SimpleNamespace(class_count=c,parent_indices=torch.arange(c).repeat_interleave(k),
            canonical_mask=torch.arange(c*k)%k==0))

    def crop(self,evidence,ids,coefficients):
        return SparseCrop(evidence,evidence.logsumexp(-1),ids,coefficients)

    def test_interpolate_then_gather_equals_dense_reference(self):
        e = torch.randn(7,3,20)
        ids = torch.tensor([[0,1],[3,6],[2,4]])
        coeff = torch.tensor([[.2,.8],[.6,.4],[.5,.5]])
        a,c,_ = sample_cache([self.crop(e,ids,coeff)])
        r = rival_indices(torch.zeros(3,3))
        sparse = a-(c.gather(1,r)-torch.log(torch.tensor(20.)))[...,None]
        dense = e.flatten(1)[...,None]-(e.logsumexp(-1)-torch.log(torch.tensor(20.)))[:,None]
        dense = (dense[ids]*coeff[...,None,None]).sum(1).reshape(3,3,20,3)
        dense = dense.gather(-1,r[:,:,None,None].expand(-1,-1,20,1)).squeeze(-1)
        torch.testing.assert_close(sparse,dense,atol=3e-7,rtol=1e-6)
        self.assertEqual(r.tolist(),[[1,0,0]]*3)

    def test_protection_controls_and_invalid(self):
        a = torch.full((4,3,20),2.)
        f = -a
        classes = torch.full((4,3),torch.log(torch.tensor(20.)).item())
        valid = torch.tensor([True,True,True,False])
        r = rival_indices(classes)
        risk = reversal_risk(a,classes,f,classes,r,self.plan.protected,valid)
        self.assertEqual(float(risk[:,self.plan.protected].max()),0.)
        self.assertEqual(float(risk[-1].max()),0.)
        self.assertTrue(torch.equal(risk[0,:,1:],torch.full((3,19),.5)))
        for name in (POOLED,SHUFFLED):
            control = risk_control(risk,self.plan,name)
            torch.testing.assert_close(control.sum(-1),risk.sum(-1))
            self.assertEqual(float(control[:,self.plan.protected].max()),0.)

    def test_crop_lse_before_stencil_and_no_compensation(self):
        e = torch.randn(5,3,20)
        ids = torch.tensor([[0,2],[3,4]])
        coeff = torch.tensor([[.25,.75],[.4,.6]])
        crop = self.crop(e,ids,coeff)
        risk = torch.rand(2,3,20)*.8
        risk[:,0] = 0
        risk[:,:,0] = 0
        valid = torch.ones(2,dtype=torch.bool)
        delta,_ = attenuated_delta([crop],risk,valid)
        direct = ((e[ids].double()+(1-risk.double()).log()[:,None]).logsumexp(-1)
                  -e[ids].double().logsumexp(-1))
        direct = (direct*coeff.double()[...,None]).sum(1)
        torch.testing.assert_close(delta,direct,atol=1e-12,rtol=0)
        self.assertTrue(bool((delta<=0).all()))
        self.assertTrue(torch.equal(delta[:,0],torch.zeros(2,dtype=torch.float64)))

    def test_zero_risk_exact_noalias_and_writer_identity(self):
        local = torch.randn(3,3)
        wide = torch.randn(3,3)
        h = torch.rand(3,3,dtype=torch.float64)
        h /= h.sum(-1,keepdim=True)
        # Wide zero alias margins guarantee no active risk.
        e = torch.zeros(3,3,20)
        crop = self.crop(e,torch.arange(3)[:,None],torch.ones(3,1))
        valid = torch.tensor([True,True,False])
        values,_ = tile_fields(local,h,wide,[crop],[crop],valid,self.plan,
            (BASE,NOALIAS,PRIMARY,POOLED,SHUFFLED))
        for name in (PRIMARY,POOLED,SHUFFLED):
            self.assertTrue(torch.equal(values[name],values[NOALIAS]))
        expected = values[BASE]+.25*(h@((e.logsumexp(-1).double()-wide.double()).masked_fill(~valid[:,None],0.)))
        self.assertTrue(torch.equal(values[NOALIAS],expected))


if __name__=='__main__':
    unittest.main()
