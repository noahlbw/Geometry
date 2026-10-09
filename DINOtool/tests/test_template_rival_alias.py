import unittest

import torch

from dinotool.template_rival_alias import (BASE, PRIMARY, POOLED, SHUFFLED, UNPAIRED,
    METHODS, TemplatePlan, margin_correction, patch_fields, reduce_alias, rival_indices)


class TemplateMarginTests(unittest.TestCase):
    def setUp(self):
        members = torch.arange(40).reshape(2,20)
        protected = torch.zeros(2,20,dtype=torch.bool);protected[:,0] = True
        permutation = torch.cat((torch.zeros(2,1,dtype=torch.long),
                                 torch.arange(19,0,-1).expand(2,-1)),1)
        self.plan = TemplatePlan(members,torch.tensor([0,20]),protected,permutation)

    def test_rivals_ties_and_exclusion(self):
        scores = torch.tensor([[2.,2.,1.],[0.,1.,3.]])
        self.assertEqual(rival_indices(scores).tolist(),[[1,0,0],[2,2,1]])

    def test_same_template_response_replays_base(self):
        values = torch.linspace(-.2,.4,120).reshape(3,40,1).expand(-1,-1,6)
        salience = torch.linspace(0.,1.,40)
        result,stats = patch_fields(values,salience,self.plan,METHODS)
        expected = reduce_alias(values.mean(-1)*40.,salience,self.plan,1.,1.)
        for name in METHODS:
            torch.testing.assert_close(result[name],expected,rtol=0,atol=0)
        self.assertEqual(stats['mean_abs_alias_correction'],0.)

    def test_bound_anchor_and_covariance(self):
        generator = torch.Generator().manual_seed(17)
        responses = torch.randn(4,2,20,6,generator=generator)
        own = responses[:,:,0].clone()
        rival = torch.randn(4,2,6,generator=generator)
        delta = margin_correction(responses,own,rival,self.plan.protected)
        bound = (responses-own[:,:,None]).std(-1,correction=0)
        self.assertTrue(bool((delta.abs()<=bound+1e-6).all()))
        self.assertTrue(bool((delta[:,:,0]==0).all()))
        changed = margin_correction(responses,own,rival.flip(-1),self.plan.protected)
        self.assertGreater(float((delta-changed).abs().max()),.01)
        # A rival common to every template cancels, while covariance does not.
        common = torch.randn(4,2,1,generator=generator)
        torch.testing.assert_close(margin_correction(responses+common[:,:,None],own+common,
            rival+common,self.plan.protected),delta,rtol=1e-5,atol=1e-6)

    def test_controls_preserve_anchor_and_alias_spectrum(self):
        values = torch.randn(3,40,6,generator=torch.Generator().manual_seed(9))*.03
        result,stats = patch_fields(values,torch.zeros(40),self.plan,METHODS)
        self.assertGreater(stats['mean_abs_unpair_difference'],0.)
        self.assertEqual(stats['canonical_correction_max'],0.)
        self.assertLess(stats['bound_violation_max'],1e-5)
        self.assertTrue(all(bool(torch.isfinite(result[name]).all()) for name in METHODS))


if __name__ == '__main__':
    unittest.main()
