import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.canonical_rival_penalty import (BASE, PRIMARY, POOLED, SHUFFLED,
    prepare_plan, penalties, class_logits, tile_scores)


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261008)
        text = F.normalize(torch.randn(60, 16), dim=-1)
        self.bank = SimpleNamespace(features=text, parent_indices=torch.arange(3).repeat_interleave(20),
                                   class_count=3, canonical_mask=torch.arange(60) % 20 == 0)
        self.plan = prepare_plan(self.bank)
        self.feature = F.normalize(torch.randn(9, 16), dim=-1)
        self.cosine = self.feature@text.T

    def test_sparse_formula_and_protection(self):
        p, t = self.plan, self.bank.features
        own, foreign = t[p.canonical][:, None], t[p.canonical][p.rival]
        unit = F.normalize(foreign-p.own_rival_cosine[..., None]*own, dim=-1)
        explicit = (torch.einsum('nd,ckd->nck', self.feature, unit).clamp_min(0.)
                    *(t[p.members]*unit).sum(-1).clamp_min(0.))
        explicit.masked_fill_(p.protected[None], 0.)
        torch.testing.assert_close(penalties(self.cosine, p), explicit, atol=2e-7, rtol=2e-5)
        self.assertEqual(float(penalties(self.cosine, p)[:, p.protected].abs().max()), 0.)
        self.assertTrue(bool(((own*unit).sum(-1).abs() < 2e-6).all()))

    def test_fixed_slot_and_writer(self):
        q = penalties(self.cosine, self.plan)
        local = class_logits(self.cosine, self.plan)
        changed = class_logits(self.cosine, self.plan, q)
        self.assertTrue(bool((changed <= local+1e-6).all()))
        slots = self.cosine[:, self.plan.members]/.07
        gate = (-q/.07).exp()
        manual = ((slots.exp()*gate).sum(-1)/20).log()
        torch.testing.assert_close(changed, manual)
        torch.testing.assert_close(class_logits(self.cosine, self.plan, torch.zeros_like(q)), local)
        operator = torch.eye(9, dtype=torch.float64)*.6
        broad = torch.randn(9, 3)
        values, _ = tile_scores(self.cosine, broad, operator, self.plan, .5)
        direct = changed.double()+.5*(operator@(broad.double()-changed.double()))
        torch.testing.assert_close(values[PRIMARY], direct)
        for method in (BASE, PRIMARY, POOLED, SHUFFLED):
            single, _ = tile_scores(self.cosine, broad, operator, self.plan, .5, (method,))
            torch.testing.assert_close(single[method], values[method])

    def test_controls_and_degenerate_canonicals(self):
        q = penalties(self.cosine, self.plan)
        shuffled = q.gather(-1, self.plan.permutation[None].expand_as(q))
        torch.testing.assert_close(q.sort(-1).values, shuffled.sort(-1).values)
        self.assertEqual(float(shuffled[:, self.plan.protected].abs().max()), 0.)
        pooled = (q.sum(-1, keepdim=True).expand_as(q)/19).masked_fill(self.plan.protected[None], 0.)
        torch.testing.assert_close(pooled.sum(-1), q.sum(-1))
        self.bank.features[20] = self.bank.features[0]
        self.bank.features[40] = self.bank.features[0]
        degenerate = prepare_plan(self.bank)
        self.assertTrue(bool((degenerate.alpha == 0).all()))
        self.assertTrue(bool(torch.isfinite(penalties(self.cosine, degenerate)).all()))


if __name__ == '__main__':
    unittest.main()
