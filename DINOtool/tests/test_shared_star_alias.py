import unittest
import torch

from dinotool.shared_star_alias import (StarCrop, edge_fields, edge_risk,
    edge_action, graph, project_star, shuffle_risk)
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.native_query_alias import native_risk
from dinotool.one_sided_alias_stream import fixed_slot_stream
from dinotool.rival_alias_fast import CachedCrop, sampled_cached


class StarAliasTests(unittest.TestCase):
    def test_matches_dense_selected_edges_and_action(self):
        gen = torch.Generator().manual_seed(1008)
        n,c,k,t = 13,7,20,31
        local = torch.randn(n,c,k,generator=gen)*7
        members = torch.arange(c*k).reshape(c,k)
        canonical = members[:,0]
        parents = torch.arange(c).repeat_interleave(k)
        valid = torch.ones(n,dtype=torch.bool);valid[-2:] = False
        coordinates = torch.zeros(n,2)
        sparse,dense = [],[]
        for _ in range(2):
            evidence = torch.randn(t,c,k,generator=gen)*9
            ids = torch.randint(t,(n,4),generator=gen)
            coef = torch.rand(n,4,generator=gen);coef/=coef.sum(-1,keepdim=True)
            sparse.append(StarCrop(evidence,ids,coef))
            dense.append(CachedCrop(evidence,contribution_margins(evidence),ids,coef))
        nodes,src,dst = graph(torch.randn(n,c,generator=gen),2)
        wm,lm,_,_,cached = edge_fields(sparse,local,src,dst)
        protect = (members == canonical[:,None])[src]
        actual_risk = edge_risk(wm,lm,protect,valid)
        full_w = sampled_cached(dense,coordinates)
        full_l = contribution_margins(local)
        dense_risk = native_risk(full_w,full_l,full_l,parents,canonical,valid)
        rows = torch.arange(n)[:,None,None]
        alias = members[src]
        expected_risk = dense_risk[rows,alias,dst[...,None]]
        self.assertTrue(torch.equal(expected_risk,actual_risk))
        full_action,_ = fixed_slot_stream(dense,members,dense_risk,valid)
        expected = full_action[torch.arange(n)[:,None],src,dst]
        actual = edge_action(cached,actual_risk,valid)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        self.assertTrue((actual <= 1e-12).all())

    def test_star_minimum_norm_and_untouched_classes(self):
        nodes = torch.tensor([[2,4,1],[0,2,3]])
        directed = torch.tensor([[-.4,-.3,-.2,-.7],[-.2,-.6,-.1,-.3]],dtype=torch.float64)
        delta,margin = project_star(nodes,directed,6)
        for row in range(2):
            incidence = torch.zeros(2,6,dtype=torch.float64)
            incidence[:,nodes[row,0]] = 1
            incidence[torch.arange(2),nodes[row,1:]] = -1
            reference = torch.linalg.lstsq(incidence,margin[row,:,None],driver='gelsd').solution[:,0]
            torch.testing.assert_close(delta[row],reference,atol=1e-12,rtol=1e-12)
        self.assertTrue((delta[0,torch.tensor([0,3,5])] == 0).all())
        torch.testing.assert_close(delta.sum(-1),torch.zeros(2,dtype=torch.float64),atol=1e-12,rtol=0)

    def test_zero_risk_extremes_and_shuffle_protection(self):
        n,c,k = 3,5,20
        nodes,src,dst = graph(torch.zeros(n,c),4)
        self.assertEqual(nodes[0].tolist(),list(range(c)))
        protected = torch.zeros(c,k,dtype=torch.bool);protected[:,7] = True
        risk = torch.rand(n,8,k);risk.masked_fill_(protected[src],0.)
        shuffled = shuffle_risk(risk,src,protected)
        self.assertTrue((shuffled[protected[src]] == 0).all())
        torch.testing.assert_close(shuffled.sort(-1).values,risk.sort(-1).values)
        values = torch.zeros(n,4,8,k,dtype=torch.float64);values[...,7] = -10000.
        coeff = torch.full((n,4),.25,dtype=torch.float64)
        valid = torch.tensor([True,True,False])
        actual = edge_action([(values,coeff)],torch.zeros_like(risk),valid)
        self.assertTrue((actual == 0).all())
        # All probability-bearing aliases attenuated fully; canonical has
        # subnormal mass, requiring stable logsumexp rather than log(0).
        extreme = torch.ones_like(risk);extreme.masked_fill_(protected[src],0.)
        actual = edge_action([(values,coeff)],extreme,valid)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual[-1] == 0).all())
        self.assertTrue((actual[:2] < -9999).all())


if __name__ == '__main__':unittest.main()
