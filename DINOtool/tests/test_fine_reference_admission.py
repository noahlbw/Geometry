from dataclasses import replace
import unittest

import torch
import torch.nn.functional as F

from dinotool.coherent_native_admission import coherent_observation
from dinotool.fine_reference_admission import CONFIG, PRIMARY, fine_crop_positions, fine_source_controls
from dinotool.semantic_reference_admission import weighted_family_field
from dinotool.stratified_soft_alias import WideCrop, crop_stencil


class FineReferenceAdmissionTest(unittest.TestCase):
    def fixture(self):
        positions = fine_crop_positions(512, 512)
        crops = [WideCrop(torch.tensor([[2., 8., 1., 1., 8., 3.]]).expand(1024, -1).clone(),
            torch.tensor([0., 2., 0., 0., 2., 0.]), *pos, grid_side=32, crop_side=256) for pos in positions]
        coords = torch.tensor([[8., 8.], [248., 248.], [264., 264.], [504., 504.]])
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        excluded = torch.tensor([[False, True, False, False, True, False]])
        valid = torch.tensor([True, True, True, False])
        return crops, torch.ones(512, 512), coords, members, excluded, valid

    def test_positions_cover_real_pixels_only_once(self):
        for h, w in ((512, 512), (401, 315), (1, 1), (255, 257)):
            coverage = torch.zeros(h, w)
            positions = fine_crop_positions(h, w)
            self.assertLessEqual(len(positions), 4)
            for t, l, ch, cw in positions:
                coverage[t:t+ch, l:l+cw] += 1
            torch.testing.assert_close(coverage, torch.ones_like(coverage), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            fine_crop_positions(513, 512)

    def test_physical_stencil_matches_dense_interpolation(self):
        crops, count, coords, *_ = self.fixture()
        torch.manual_seed(12)
        coords = torch.cat((coords, torch.tensor([[255.8, 255.9], [256.1, 256.3]])))
        dense = torch.zeros(1, 2, 512, 512)
        sampled = torch.zeros(len(coords), 2)
        coverage = torch.zeros(len(coords))
        for crop in crops:
            values = torch.randn(32*32, 2)
            ids, coeff = crop_stencil(crop, count, coords, (512, 512))
            sampled += (values[ids]*coeff[..., None]).sum(1)
            coverage += coeff.sum(1)
            up = F.interpolate(values.T.reshape(1, 2, 32, 32), (256, 256), mode='bilinear', align_corners=False)
            dense[:, :, crop.top:crop.top+256, crop.left:crop.left+256] = up
        grid = (coords.flip(-1)/512*2-1).reshape(1, -1, 1, 2)
        expected = F.grid_sample(dense, grid, align_corners=False, padding_mode='border')[0, :, :, 0].T
        torch.testing.assert_close(coverage, torch.ones_like(coverage), atol=1e-6, rtol=0)
        torch.testing.assert_close(sampled, expected, atol=2e-5, rtol=0)

    def test_held_family_cannot_confirm_fine_reference(self):
        args = list(self.fixture())
        weight = torch.tensor([1., .2, .5, 1., .2, .5])
        before = weighted_family_field(*args, weight)[0]
        args[0] = [replace(c, alias_logits=c.alias_logits.clone(), salience=c.salience.clone()) for c in args[0]]
        for crop in args[0]:
            crop.alias_logits[:, [1, 4]], crop.salience[[1, 4]] = 10000, 10000
        torch.testing.assert_close(before, weighted_family_field(*args, weight)[0], atol=0, rtol=0)

    def test_zero_semantic_identity_and_source_spectra(self):
        args = self.fixture()
        parents, canonical = torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3])
        conflict = torch.tensor([[0., 0.], [0., .8], [0., .3], [0., 0.], [.7, 0.], [.2, 0.]], dtype=torch.float64)
        arguments = (*args, torch.zeros(6, dtype=torch.long), parents, canonical)
        sources, fields = fine_source_controls(*arguments, conflict, conflict, torch.zeros(4, 6, 2))
        self.assertEqual(fields['spatial_risk_spectrum_error'], 0.)
        self.assertEqual(max(fields['semantic_spectrum_errors'].values()), 0.)
        for risk in sources.values():
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))
            self.assertEqual(float(risk[~args[5]].abs().max()), 0.)
        neutral, fields = fine_source_controls(*arguments, torch.zeros_like(conflict), conflict, torch.zeros(4, 6, 2))
        torch.testing.assert_close(neutral[PRIMARY], neutral['FineReferenceOnly_Exact'], atol=0, rtol=0)
        torch.testing.assert_close(neutral[PRIMARY], fields['old_base'], atol=0, rtol=0)

    def test_fine_reference_writer_keeps_zero_and_capacity(self):
        crops, count, coords, members, _, valid = self.fixture()
        zero = coherent_observation(crops, count, coords, (512, 512), members, torch.zeros(4, 6), valid, CONFIG)[0]
        self.assertEqual(float(zero.abs().max()), 0.)
        action, cap, stats = coherent_observation(crops, count, coords, (512, 512), members, torch.ones(4, 6), valid, CONFIG)
        self.assertLessEqual(stats['capacity_excess_max'], 1e-10)
        torch.testing.assert_close(-action, cap, atol=1e-12, rtol=0)

    def test_config_is_fixed(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, encoder_side=336).validate()


if __name__ == '__main__':
    unittest.main()
