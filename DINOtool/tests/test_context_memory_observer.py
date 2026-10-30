import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.context_memory_observer import (
    ContextMemoryConfig, CompactContextScorer, context_assignments, memory_read,
    regression_factors, regression_read, read_response_before_head,
)


class ContextMemoryTests(unittest.TestCase):
    def test_response_before_identity_matches_native_head(self):
        torch.manual_seed(19)
        nn = torch.nn
        blocks=[]
        for _ in range(2):
            blocks.append(SimpleNamespace(norm1=nn.LayerNorm(6),norm2=nn.LayerNorm(6),
                attn=SimpleNamespace(num_heads=2,scale=6**-0.5,qkv=nn.Linear(6,18),proj=nn.Linear(6,6),proj_drop=nn.Identity()),
                ls1=nn.Identity(),ls2=nn.Identity(),mlp=nn.Sequential(nn.Linear(6,12),nn.GELU(),nn.Linear(12,6))))
        head=SimpleNamespace(blocks=blocks,ln_final=nn.LayerNorm(6),linear_projection=nn.Linear(6,4))
        tokens=torch.randn(1,10,6)
        assignment=torch.eye(5)[None]
        expected=tokens
        for block in blocks:
            encoded=block.attn.qkv(block.norm1(expected)).reshape(1,10,3,2,3)
            query,key,value=(part.transpose(1,2) for part in encoded.unbind(2))
            weights=(query.float()@key.float().transpose(-1,-2)*block.attn.scale).softmax(-1).to(value.dtype)
            merged=(weights@value).transpose(1,2).reshape(1,10,6)
            expected=expected+block.ls1(block.attn.proj(merged))
            expected=expected+block.ls2(block.mlp(block.norm2(expected)))
        expected=F.normalize(head.linear_projection(head.ln_final(expected)).float(),dim=-1)
        actual=read_response_before_head(head,tokens,assignment)
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
    def test_regression_mean_and_spectrum(self):
        torch.manual_seed(17)
        p = torch.randn(1, 23, 5).softmax(-1)
        factors = regression_factors(p)
        relation = regression_read(factors, torch.eye(23)[None])
        torch.testing.assert_close(relation, relation.transpose(1,2), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(relation.sum(-1), torch.ones(1,23), atol=2e-6, rtol=2e-6)
        eig = torch.linalg.eigvalsh(relation)
        self.assertGreaterEqual(float(eig.min()), -2e-6)
        self.assertLessEqual(float(eig.max()), 1.000002)
        v = torch.randn(1,23,7)
        torch.testing.assert_close(regression_read(factors,v), relation @ v, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(regression_read(factors,v).mean(1), v.mean(1), atol=2e-6, rtol=2e-6)
        constant = torch.ones(1,23,3)
        torch.testing.assert_close(regression_read(factors,constant), constant, atol=2e-6, rtol=2e-6)

    def test_compact_scores_equal_template_level_reference(self):
        torch.manual_seed(15)
        features = torch.randn(1, 6, 5)
        text = torch.randn(4, 3, 5)
        query = SimpleNamespace(features=text, parents=torch.tensor([0, 1, 0, 2]),
                                class_names=('a', 'b', 'c'))
        source = dict(wide_size=(2, 3), wide=[dict(top=0, left=0, ah=2, aw=3,
            grid_hw=(2, 3), output_hw=(2, 3), features=features)])
        aliases = torch.einsum('bnd,mtd->bnmt', features, text).mean(-1)[0]*40.
        reference = torch.stack([torch.logsumexp(.7*aliases[:, query.parents == c], -1)/.7
                                 for c in range(3)]).reshape(3, 2, 3)
        torch.testing.assert_close(CompactContextScorer()(source, query, .7), reference, atol=1e-5, rtol=1e-5)
        salience = (torch.nn.functional.normalize(features.mean(1), dim=-1)
                    @ torch.nn.functional.normalize(text.mean(1), dim=-1).T)[0]/.3
        weighted = []
        for c in range(3):
            selected = query.parents == c
            weights = salience[selected].softmax(0)*int(selected.sum())
            weighted.append(torch.logsumexp(.7*aliases[:,selected]*weights, -1)/.7)
        reference = torch.stack(weighted).reshape(3,2,3)
        torch.testing.assert_close(CompactContextScorer(salience=True)(source,query,.7,.3),
                                   reference,atol=1e-4,rtol=1e-5)

    def test_logmeanexp_is_alias_duplicate_invariant(self):
        torch.manual_seed(151)
        features = torch.randn(1, 4, 5)
        text = torch.randn(3, 2, 5)
        query = SimpleNamespace(features=text, parents=torch.tensor([0, 0, 1]),
                                class_names=('a', 'b'))
        source = dict(wide_size=(2, 2), wide=[dict(top=0, left=0, ah=2, aw=2,
            grid_hw=(2, 2), output_hw=(2, 2), features=features)])
        scorer = CompactContextScorer(aggregation='logmeanexp')
        base = scorer(source, query, .7)

        # Duplicate the complete alias bank.  Log-sum-exp changes every class
        # by log(2)/tau, whereas log-mean-exp is invariant.
        duplicated = SimpleNamespace(
            features=text.repeat_interleave(2, dim=0),
            parents=torch.tensor([0, 0, 0, 0, 1, 1]),
            class_names=('a', 'b'),
        )
        duplicated_result = CompactContextScorer(aggregation='logmeanexp')(
            source, duplicated, .7)
        torch.testing.assert_close(base, duplicated_result, atol=1e-5, rtol=1e-5)

    def test_invalid_compact_aggregation_is_rejected(self):
        with self.assertRaises(ValueError):
            CompactContextScorer(aggregation='sum')

    def test_factorized_read_matches_explicit_relation_and_conserves_mass(self):
        torch.manual_seed(106)
        p = torch.randn(1, 11, 4).softmax(-1)
        v = torch.randn(1, 11, 7)
        relation = (p/p.sum(1)[:, None]) @ p.transpose(1, 2)
        actual = memory_read(p, v)
        torch.testing.assert_close(actual, relation @ v)
        torch.testing.assert_close(relation, relation.transpose(1, 2))
        torch.testing.assert_close(relation.sum(-1), torch.ones(1, 11))
        torch.testing.assert_close(relation.sum(-2), torch.ones(1, 11))
        torch.testing.assert_close(actual.sum(1), v.sum(1), atol=2e-6, rtol=2e-6)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(relation)[0].min()), -1e-6)

    def test_empty_slots_constant_values_and_identity(self):
        p = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        v = torch.tensor([[[3., 2.], [3., 2.], [3., 2.]]])
        torch.testing.assert_close(memory_read(p, v), v)
        x = torch.randn(1, 7, 3)
        torch.testing.assert_close(memory_read(torch.eye(7)[None], x), x)

    def test_deterministic_content_assignment_and_degenerate_inputs(self):
        torch.manual_seed(104)
        raw = torch.randn(1, 15, 8)
        config = ContextMemoryConfig(capacity=4)
        p = context_assignments(raw, config)
        self.assertEqual(p.shape, (1, 15, 4))
        self.assertTrue(torch.equal(p, context_assignments(raw, config)))
        torch.testing.assert_close(p.sum(-1), torch.ones(1, 15))
        self.assertTrue(torch.isfinite(context_assignments(torch.ones_like(raw), config)).all())
        self.assertTrue(torch.isfinite(context_assignments(torch.zeros_like(raw), config)).all())
        h = context_assignments(raw, ContextMemoryConfig(capacity=4, hard=True))
        self.assertTrue(((h == 0) | (h == 1)).all())
        torch.testing.assert_close(h.sum(-1), torch.ones(1, 15))


if __name__ == '__main__':
    unittest.main()
