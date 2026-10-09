import unittest

import torch

from dinotool.rival_alias_count import count_predictions
from dinotool.rival_fine_full import METHODS, retained_scores, tile_reference_coordinates
from dinotool.stratified_soft_alias import WideCrop


class RivalFineFullTest(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(9)
        members = torch.arange(40).reshape(2, 20)
        parents, canonical = torch.arange(2).repeat_interleave(20), members[:, 0]
        crop = WideCrop(torch.randn(1024, 40), torch.randn(40), 0, 0, 512, 512, 32, 512)
        fine = WideCrop(torch.randn(1024, 40), torch.randn(40), 0, 0, 512, 512, 32, 512)
        coords = torch.tensor([[8., 8.], [504., 504.], [520., 520.]])
        valid = torch.tensor([True, True, False])
        local, broad = torch.randn(3, 2), torch.randn(3, 2)
        return local, torch.eye(3), broad, [crop], torch.ones(512, 512), [fine], torch.ones(512, 512), coords, valid, members, canonical, parents

    def test_exact_historical_primary_replay(self):
        local, op, broad, wide, wc, fine, fc, coords, valid, members, canonical, parents = self.fixture()
        old, old_pair, _ = count_predictions(local, op, broad, wide, wc, fine, fc, coords,
                                             valid, members, canonical, parents, (512, 512))
        actual, pair, _ = retained_scores(local, op, broad, wide, wc, fine, fc, coords, coords,
                                          valid, members, canonical, parents, (512, 512))
        self.assertTrue(torch.equal(pair, old_pair))
        self.assertEqual(float(pair[:, canonical].abs().max()), 0.)
        self.assertEqual(float(pair[~valid].abs().max()), 0.)
        for method in METHODS:
            self.assertTrue(torch.equal(actual[method], old[method]))

    def test_global_to_fine_tile_coordinates(self):
        local = torch.tensor([[8., 8.], [504., 504.]])
        global_coords = local + torch.tensor([384., 768.])
        self.assertTrue(torch.equal(tile_reference_coordinates(global_coords, 384, 768), local))

    def test_no_risk_identity(self):
        local, op, broad, wide, wc, fine, fc, coords, valid, members, canonical, parents = self.fixture()
        crop = WideCrop(torch.zeros(1024, 40), torch.zeros(40), 0, 0, 512, 512, 32, 512)
        scores, pair, _ = retained_scores(local, op, broad, [crop], wc, [crop], fc, coords, coords,
                                          valid, members, canonical, parents, (512, 512))
        self.assertEqual(float(pair.abs().max()), 0.)
        self.assertTrue(torch.equal(scores['NoAdmission_Exact'], scores['RivalFineHard_Exact']))


if __name__ == '__main__':
    unittest.main()
