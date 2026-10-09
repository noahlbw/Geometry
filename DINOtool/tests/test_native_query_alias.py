import unittest

import torch

from dinotool.native_query_alias import native_crop_positions, native_risk, query_relation, support_controls, support_margins
from dinotool.matched_contribution_alias import stencil_margins
from dinotool.stratified_soft_alias import WideCrop, crop_stencil


class NativeQueryTest(unittest.TestCase):
    def test_actual_sampler_matches_declared_native_stride(self):
        self.assertEqual(native_crop_positions(512, 512),
            [(0, 0, 336, 336), (0, 176, 336, 336), (176, 0, 336, 336), (176, 176, 336, 336)])
        self.assertEqual(native_crop_positions(160, 512), [(0, 0, 160, 336), (0, 176, 160, 336)])
        self.assertEqual(native_crop_positions(160, 160), [(0, 0, 160, 160)])

    def source(self):
        broad = torch.ones(4, 6, 2)*2
        native = -torch.ones_like(broad)
        support = -torch.ones_like(broad)*3
        return broad, native, support, torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3]), torch.tensor([True]*3+[False])

    def test_minimum_two_witnesses_and_protection(self):
        args = self.source()
        risk = native_risk(*args)
        self.assertAlmostEqual(float(risk[0, 1, 1]), 1/3, places=7)
        self.assertEqual(float(risk[:, args[4]].abs().max()), 0.)
        self.assertEqual(float(risk[-1].abs().max()), 0.)
        self.assertEqual(float(risk[0, 1, 0]), 0.)

    def test_each_observation_sign_is_required(self):
        for field in (0, 1, 2):
            args = list(self.source())
            args[field].zero_()
            self.assertEqual(float(native_risk(*args).abs().max()), 0.)

    def test_scale_invariance(self):
        args = list(self.source())
        expected = native_risk(*args)
        args[:3] = [x*7 for x in args[:3]]
        torch.testing.assert_close(native_risk(*args), expected, atol=0, rtol=0)

    def test_unknown_retention(self):
        args = list(self.source())
        args[-1].zero_()
        self.assertEqual(float(native_risk(*args).abs().max()), 0.)

    def test_query_support_excludes_invalid_donors(self):
        weights, known = query_relation(torch.ones(4, 4), torch.tensor([True, False, True, False]))
        torch.testing.assert_close(weights[0], torch.tensor([.5, 0., .5, 0.]))
        self.assertEqual(float(weights[1].abs().max()), 0.)
        self.assertEqual(known.tolist(), [True, False, True, False])

    def test_zero_relation_is_unknown(self):
        weights, known = query_relation(torch.zeros(4, 4), torch.ones(4, dtype=torch.bool))
        self.assertFalse(bool(known.any()))
        self.assertEqual(float(weights.abs().max()), 0.)

    def test_support_is_query_specific(self):
        margin = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        identity = torch.eye(4)
        torch.testing.assert_close(support_margins(identity, margin), margin, atol=0, rtol=0)
        permutation = identity[[3, 1, 0, 2]]
        torch.testing.assert_close(support_margins(permutation, margin), margin[[3, 1, 0, 2]], atol=0, rtol=0)

    def test_control_row_mass_and_shuffle_spectrum(self):
        relation = torch.rand(6, 6, generator=torch.Generator().manual_seed(19))
        valid = torch.tensor([True, True, False, True, True, False])
        weights, _ = query_relation(relation, valid)
        coordinates = torch.tensor([[8., 8.], [8., 24.], [8., 40.], [8., 200.], [8., 216.], [8., 240.]])
        controls = support_controls(weights, coordinates, valid)
        torch.testing.assert_close(controls['shared'].sum(-1), weights.sum(-1))
        torch.testing.assert_close(controls['shuffle'].sort(-1).values, weights.sort(-1).values, atol=0, rtol=0)
        torch.testing.assert_close(controls['shared'][0], controls['shared'][1], atol=0, rtol=0)
        self.assertGreater(float((controls['shared']-weights).abs().max()), 0.)

    def test_class_and_alias_permutation(self):
        args = self.source()
        original = native_risk(*args)
        slots = torch.tensor([3, 5, 4, 0, 2, 1])
        changed = [x[:, slots][:, :, [1, 0]] for x in args[:3]]
        changed += [1-args[3][slots], torch.tensor([0, 3]), args[5]]
        torch.testing.assert_close(native_risk(*changed), original[:, slots][:, :, [1, 0]], atol=0, rtol=0)

    def test_native_overlap_covers_same_pixel_coordinates(self):
        coords = torch.stack(torch.meshgrid((torch.arange(32)+.5)*16, (torch.arange(32)+.5)*16, indexing='ij'), -1).reshape(-1, 2)
        count = torch.zeros(512, 512)
        for top in (0, 176):
            for left in (0, 176):
                count[top:top+336, left:left+336] += 1
        coefficients, measured = torch.zeros(1024), torch.zeros(1024, 6, 2)
        evidence = torch.tensor([[[2., 1., 0.], [-1., 0., 1.]]]).expand(441, -1, -1)
        for top in (0, 176):
            for left in (0, 176):
                crop = WideCrop(torch.zeros(441, 6), torch.zeros(6), top, left, 336, 336)
                ids, coeff = crop_stencil(crop, count, coords, (512, 512))
                coefficients += coeff.sum(-1)
                measured += stencil_margins(evidence, ids, coeff)
        torch.testing.assert_close(coefficients, torch.ones_like(coefficients), atol=1e-6, rtol=0)
        torch.testing.assert_close(measured, measured[:1].expand_as(measured), atol=1e-6, rtol=0)

    def test_bad_inputs_fail(self):
        args = list(self.source())
        args[1][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            native_risk(*args)
        with self.assertRaises(ValueError):
            query_relation(-torch.ones(3, 3), torch.ones(3, dtype=torch.bool))
