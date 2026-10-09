import unittest

import torch

from dinotool.stratified_soft_alias import WideCrop
from dinotool.target_context_alias import (alias_permutations, crop_masks, geometry_masks,
    intervention_retention, observation_delta, shuffled_crop_masks)


class TargetContextAliasTest(unittest.TestCase):
    def test_uniform_geometry_is_unknown(self):
        coords = self.coordinates()
        masks, assignment, known = geometry_masks(torch.ones(1024, 1024)/1024, coords, torch.ones(1024, dtype=torch.bool))
        self.assertFalse(bool(known.any()))
        self.assertEqual(float(masks.abs().max()), 0.)
        self.assertEqual(assignment.unique().numel(), 16)

    @staticmethod
    def coordinates():
        y, x = torch.meshgrid((torch.arange(32)+.5)*16, (torch.arange(32)+.5)*16, indexing='ij')
        return torch.stack((y, x), -1).reshape(-1, 2)

    def test_cell_geometry_support_and_padding(self):
        coords = self.coordinates()
        cells = (coords/128).long()
        relation = (cells[:, None] == cells[None]).all(-1).float()
        valid = torch.ones(1024, dtype=torch.bool)
        valid[-32:] = False
        masks, _, known = geometry_masks(relation, coords, valid)
        self.assertTrue(bool(known.all()))
        self.assertTrue(torch.equal(masks[:, -1], torch.zeros(16, 32)))
        self.assertEqual(float(masks[0].sum()), 64.)

    def test_crop_identity_and_outside_tile(self):
        crop = WideCrop(torch.zeros(441, 4), torch.zeros(4), 0, 0, 336, 336)
        masks = torch.ones(2, 32, 32)
        torch.testing.assert_close(crop_masks(masks, crop, (512, 512), (512, 512)), torch.ones(2, 336, 336))
        crop = WideCrop(crop.alias_logits, crop.salience, 512, 512, 336, 336)
        self.assertEqual(float(crop_masks(masks, crop, (1024, 1024), (1024, 1024)).sum()), 0.)

    def test_shuffled_mask_has_identical_area_spectrum(self):
        masks = torch.rand(3, 336, 336)
        shuffled = shuffled_crop_masks(masks, 200, 180, 9)
        torch.testing.assert_close(masks[:, :200, :180].flatten(1).sort(-1).values,
                                   shuffled[:, :200, :180].flatten(1).sort(-1).values)
        self.assertEqual(float(shuffled[:, 200:].sum()), 0.)
        self.assertEqual(float(shuffled[:, :, 180:].sum()), 0.)

    def test_valid_attached_alias_can_be_rejected(self):
        full = torch.tensor([[.8, .9, .3, .4]])
        kept = torch.tensor([[[.1, .2, .7, .5]], [[.1, .2, .7, .5]]])
        removed = full[None].expand_as(kept)
        rho, _ = intervention_retention(full, kept, removed, torch.tensor([0, 0, 1, 1]),
                                        torch.tensor([0, 2]), torch.tensor([True]))
        self.assertLess(float(rho[0, 1]), .01)
        self.assertEqual(float(rho[0, 0]), 1.)
        self.assertEqual(float(rho[0, 2]), 1.)

    def test_target_supported_alias_is_preserved(self):
        full = torch.tensor([[.5, .8, .1, .2]])
        kept = full[None].expand(2, -1, -1)
        rho, context = intervention_retention(full, kept, torch.zeros_like(kept), torch.tensor([0, 0, 1, 1]),
                                              torch.tensor([0, 2]), torch.tensor([True]))
        self.assertTrue(torch.equal(rho, torch.ones_like(full)))
        self.assertTrue(torch.equal(context, torch.ones_like(full)))

    def test_disagreeing_fills_and_unknown_do_not_reject(self):
        full = torch.tensor([[.8, .9, .3, .4]])
        kept = torch.stack((torch.tensor([[.1, .2, .7, .5]]), full))
        removed = full[None].expand_as(kept)
        for known in (True, False):
            rho, _ = intervention_retention(full, kept, removed, torch.tensor([0, 0, 1, 1]),
                                            torch.tensor([0, 2]), torch.tensor([known]))
            self.assertTrue(torch.equal(rho, torch.ones_like(full)))

    def test_three_alias_shuffles_protect_canonical_and_spectrum(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            members = torch.arange(8, device=device).reshape(2, 4)
            canonical = torch.tensor([1, 6], device=device)
            values = torch.rand(3, 2, 4, device=device)
            for permutation in alias_permutations(members, canonical):
                self.assertEqual(permutation.device, members.device)
                self.assertEqual(int(permutation[0, 1]), 1)
                self.assertEqual(int(permutation[1, 2]), 2)
                torch.testing.assert_close(values.gather(-1, permutation[None].expand_as(values)).sort(-1).values,
                                           values.sort(-1).values)

    def test_writer_identity_and_monotonicity(self):
        crop = WideCrop(torch.randn(441, 4), torch.randn(4), 0, 0, 336, 336)
        coords = torch.tensor([[8., 8.], [100., 100.]])
        members = torch.arange(4).reshape(2, 2)
        count = torch.ones(336, 336)
        valid = torch.tensor([True, False])
        identity = observation_delta([crop], count, coords, (336, 336), members, torch.ones(2, 4), valid)
        self.assertEqual(float(identity.abs().max()), 0.)
        changed = observation_delta([crop], count, coords, (336, 336), members, torch.zeros(2, 4), valid)
        self.assertTrue(bool((changed <= 0).all()))
        self.assertEqual(float(changed[1].abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
