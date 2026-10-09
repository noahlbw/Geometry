import types
import unittest
import torch

from dinotool.shared_local_alias import (local_evidence, shared_scores, PRIMARY, MEAN,
                                         POOLED, SHUFFLED)
from dinotool.rival_alias_fast import CachedCrop
from dinotool.matched_contribution_alias import contribution_margins


class SharedLocalTests(unittest.TestCase):
    def test_template_mean_matches_explicit_linear_average(self):
        torch.manual_seed(1008)
        features = torch.nn.functional.normalize(torch.randn(1, 9, 16), dim=-1)
        query = types.SimpleNamespace(features=torch.randn(12, 5, 16))
        members = torch.arange(12).reshape(3, 4)
        actual = local_evidence(features, query, members)
        dot = torch.einsum('bnd,mtd->bnmt', features, query.features).mean(-1)[0]*40
        text = query.features.mean(1)
        salience = (torch.nn.functional.normalize(features.mean(1), dim=-1) @
                    torch.nn.functional.normalize(text, dim=-1).T)[0]
        expected = dot[:, members]*(4*salience[members].softmax(-1))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

    def test_no_conflict_reduces_to_same_information_mean(self):
        torch.manual_seed(81)
        e = torch.randn(7, 3, 4)
        members = torch.arange(12).reshape(3, 4)
        parents = torch.arange(3).repeat_interleave(4)
        canonical = members[:, 0]
        valid = torch.tensor([True]*6+[False])
        coord = torch.randn(7, 2)
        crop = CachedCrop(e, contribution_margins(e), torch.arange(7)[:, None], torch.ones(7, 1))
        local = torch.randn(7, 3)
        broad = e.logsumexp(-1)
        operator = torch.diag(valid.double()*.5)
        values, stats = shared_scores(local, operator, broad, e, [crop], coord, valid,
            members, canonical, parents, (PRIMARY, MEAN, POOLED, SHUFFLED))
        for method in (PRIMARY, POOLED, SHUFFLED):
            torch.testing.assert_close(values[method], values[MEAN], atol=1e-12, rtol=0)
        self.assertEqual(stats['canonical_risk_max'], 0)


if __name__ == '__main__':
    unittest.main()
