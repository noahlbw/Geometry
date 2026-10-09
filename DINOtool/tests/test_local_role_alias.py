"""Local mass, provenance, identity-null and coupled action invariants."""
import math
from types import SimpleNamespace
import unittest

import torch

from dinotool.local_role_alias import prepare_plan, role_priors, local_scores, PRIMARY, SHUFFLE
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.development_readout import coupled


def bank():
    return SimpleNamespace(class_count=2, class_names=('a', 'b'),
        alias_names=tuple('a'+str(i) for i in range(20))+tuple('b'+str(i) for i in range(20)),
        parent_indices=torch.arange(2).repeat_interleave(20), canonical_mask=torch.arange(40)%20 == 0)


class LocalRoleTest(unittest.TestCase):
    def test_recorded_roots_have_unit_mass_and_wrappers_zero(self):
        labels = [0]*13+[1]*7
        words = ['a'+str(i) for i in range(20)]
        prior = role_priors(words, labels, 0, ['a0','a13'])
        self.assertEqual(prior[0], .5)
        self.assertEqual(prior[13], .5)
        self.assertEqual(sum(prior), 1.)
        self.assertEqual(sum(x == 0 for x in prior), 18)

    def test_missing_provenance_retains_every_member_and_equal_family_mass(self):
        p = role_priors(['a'+str(i) for i in range(20)], [0]*13+[1]*7, 0)
        self.assertTrue(all(x > 0 for x in p))
        self.assertAlmostEqual(sum(p[:13]), .5)
        self.assertAlmostEqual(sum(p[13:]), .5)

    def test_bad_root_and_unprotected_anchor_are_rejected(self):
        words = ['a'+str(i) for i in range(20)]
        with self.assertRaises(ValueError): role_priors(words, [0]*20, 0, ['missing'])
        with self.assertRaises(ValueError): role_priors(words, [0]*20, 0, ['a1'])

    def test_shuffle_preserves_canonical_mass_support_count_and_spectrum(self):
        b = bank(); p = prepare_plan(b, [[0]*13+[1]*7]*2, [['a0','a13'],['b0','b13']])
        self.assertTrue(torch.equal(p.log_prior[:,0], p.shuffled_log_prior[:,0]))
        self.assertTrue(torch.equal(p.log_prior.sort(-1).values, p.shuffled_log_prior.sort(-1).values))
        self.assertFalse(torch.equal(p.log_prior, p.shuffled_log_prior))

    def test_uniform_exactly_replays_original_local_lme(self):
        b = bank(); p = prepare_plan(b, [list(range(20))]*2)
        x = torch.randn(31,40)
        expected = alias_class_scores(x, b.parent_indices, 2)/.07
        self.assertTrue(torch.equal(local_scores(x,p,PRIMARY),expected))
        self.assertTrue(torch.equal(local_scores(x,p,SHUFFLE),expected))

    def test_weighted_local_changes_only_through_i_minus_gain_h(self):
        b = bank(); p = prepare_plan(b, [[0]*13+[1]*7]*2, [['a0','a13'],['b0','b13']])
        x = torch.randn(11,40,dtype=torch.float64)
        local = alias_class_scores(x,b.parent_indices,2)/.07
        modified = local_scores(x,p,PRIMARY)
        wide = torch.randn(11,2,dtype=torch.float64)
        h = torch.rand(11,11,dtype=torch.float64)
        h = h/h.sum(-1,keepdim=True)
        delta = modified-local
        expected = delta-.5*(h@delta)
        self.assertTrue(torch.allclose(coupled(modified,wide,h,.5)-coupled(local,wide,h,.5),expected,atol=1e-12))
        # Every local class is a normalized mixture: identical alias scores stay identical.
        equal = local_scores(torch.full((4,40),.2),p,PRIMARY)
        self.assertTrue(torch.allclose(equal,torch.full_like(equal,.2/.07),atol=1e-6))


if __name__ == '__main__': unittest.main()
