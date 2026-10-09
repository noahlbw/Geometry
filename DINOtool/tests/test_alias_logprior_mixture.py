import math
import unittest

import torch

from dinotool.alias_logprior_mixture import MixturePlan, PRIMARY, UNIFORM, SHUFFLED, mixture_logits


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.plan = MixturePlan(torch.arange(40).reshape(2,20),
                                torch.arange(19,-1,-1).expand(2,-1))

    def test_uniform_identity_and_joint_class_shift(self):
        raw = torch.linspace(-5,7,120).reshape(3,40)
        salience = torch.zeros(40)
        expected = raw[:,self.plan.members].logsumexp(-1)
        for method in (PRIMARY,UNIFORM,SHUFFLED):
            torch.testing.assert_close(mixture_logits(raw,salience,self.plan,method),expected,rtol=0,atol=0)
        shifted = raw+torch.tensor([3.,-2.]).repeat_interleave(20)
        torch.testing.assert_close(mixture_logits(shifted,salience,self.plan,PRIMARY),
                                   expected+torch.tensor([3.,-2.]),rtol=1e-6,atol=1e-6)

    def test_weight_alignment_and_prior_bound(self):
        raw = torch.zeros(1,40);raw[:,self.plan.members[:,0]] = 5
        salience = torch.zeros(40);salience[self.plan.members[:,0]] = 1
        actual = mixture_logits(raw,salience,self.plan,PRIMARY)
        uniform = mixture_logits(raw,salience,self.plan,UNIFORM)
        shuffled = mixture_logits(raw,salience,self.plan,SHUFFLED)
        self.assertTrue(bool((actual>uniform).all() and (uniform>shuffled).all()))
        prior = salience[self.plan.members].log_softmax(-1)+math.log(20)
        delta = actual-uniform
        self.assertTrue(bool((delta>=prior.min(-1).values-1e-6).all()
                             and (delta<=prior.max(-1).values+1e-6).all()))


if __name__ == '__main__':
    unittest.main()
