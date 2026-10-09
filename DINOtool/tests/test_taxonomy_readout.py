import unittest
from types import SimpleNamespace
import torch
from dinotool.taxonomy_readout import rank_fraction,natural_profile


def bank(features):
    return SimpleNamespace(features=features,parent_indices=torch.arange(len(features)),
        canonical_mask=torch.ones(len(features),dtype=torch.bool),class_count=len(features))


class TaxonomyTests(unittest.TestCase):
    def test_rank_rotation_and_order_invariance(self):
        torch.manual_seed(7)
        x=torch.randn(30,12)
        rotation=torch.linalg.qr(torch.randn(12,12)).Q
        self.assertAlmostEqual(rank_fraction(bank(x)),rank_fraction(bank(x[torch.randperm(30)]@rotation)),places=6)

    def test_collapsed_taxonomy_uses_fidelity_route(self):
        p,record=natural_profile(bank(torch.ones(20,5)))
        self.assertEqual(p.strength,'original')
        self.assertEqual(p.coupling,.5)
        self.assertEqual(record['canonical_rank_fraction'],0.)

    def test_orthogonal_taxonomy_uses_stronger_wide_prior(self):
        p,_=natural_profile(bank(torch.eye(20)))
        self.assertEqual(p.strength,2.)
        self.assertEqual(p.coupling,2.)


if __name__=='__main__':
    unittest.main()
