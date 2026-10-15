import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dinotool.anchored_slot_memory import (
    READ_MODES, anchored_slot_features, anchored_slot_wide_observations,
)
from dinotool.context_memory_observer import ContextMemoryConfig
from dinotool.query_slot_memory import _sample_global, query_slot_features, query_slot_wide_observations


class AnchoredSlotAlgebraTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20261015)
        self.query = torch.randn(1, 7, 4, generator=generator)
        self.reference = torch.randn(1, 7, 4, generator=generator)
        self.centers = F.normalize(torch.randn(1, 3, 4, generator=generator), dim=-1)
        self.values = F.normalize(torch.randn(1, 3, 5, generator=generator), dim=-1)
        self.baseline = torch.randn(1, 7, 5, generator=generator)
        self.prior = torch.randn(1, 7, 3, generator=generator).softmax(-1)

    def read(self, **kwargs):
        args = dict(query_raw=self.query, reference_raw=self.reference,
                    centers=self.centers, values=self.values, baseline=self.baseline,
                    spatial_prior=self.prior)
        args.update(kwargs)
        return anchored_slot_features(**args)

    def test_equal_descriptors_restore_coordinate_baseline(self):
        actual = self.read(reference_raw=self.query)
        torch.testing.assert_close(actual, F.normalize(self.baseline, dim=-1), atol=1e-6, rtol=1e-6)

    def test_coordinate_and_beta_zero_skip_all_routing_inputs(self):
        expected = F.normalize(self.baseline, dim=-1)
        coordinate = anchored_slot_features(None, None, None, None, self.baseline, None,
                                             mode='coordinate')
        beta_zero = anchored_slot_features(None, None, None, None, self.baseline, None, beta=0.)
        self.assertTrue(torch.equal(coordinate, expected))
        self.assertTrue(torch.equal(beta_zero, expected))

    def test_constant_values_and_common_value_shift_cancel(self):
        constant = torch.tensor([[[1., 2., 3., 4., 5.]]]).expand_as(self.values)
        torch.testing.assert_close(self.read(values=constant), F.normalize(self.baseline, dim=-1),
                                   atol=2e-6, rtol=2e-6)
        offset = torch.tensor([[[.3, -.5, 2., 1., -2.]]])
        torch.testing.assert_close(self.read(values=self.values+offset), self.read(),
                                   atol=2e-6, rtol=2e-6)

    def test_no_prior_replaces_posterior_and_residual_anchor(self):
        uniform = torch.full_like(self.prior, 1./self.prior.shape[-1])
        expected = self.read(spatial_prior=uniform)
        actual = self.read(mode='no_prior')
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(self.read(mode='no_prior', spatial_prior=self.prior.flip(-1)), actual)
        self.assertGreater(float((self.read()-actual).abs().sum()), 1e-3)

    def test_control_formulas_and_legacy_exact_match(self):
        q = F.normalize(self.query, dim=-1)
        qbar = F.normalize(self.reference, dim=-1)
        posterior = (self.prior.log()+(q-qbar) @ self.centers.transpose(1, 2)/.1).softmax(-1)
        expected = F.normalize(.5*self.baseline+.5*(posterior @ self.values), dim=-1)
        torch.testing.assert_close(self.read(mode='absolute_mix'), expected)
        q_posterior = (self.prior.log()+q @ self.centers.transpose(1, 2)/.1).softmax(-1)
        expected = F.normalize(self.baseline+.5*((q_posterior-self.prior) @ self.values), dim=-1)
        torch.testing.assert_close(self.read(mode='no_reference'), expected, atol=1e-6, rtol=1e-6)
        legacy = query_slot_features(self.query, self.centers, self.values, self.baseline)
        self.assertTrue(torch.equal(self.read(mode='legacy'), legacy))
        torch.testing.assert_close(self.read(mode='shuffled'),
                                   self.read(query_raw=self.query.roll(self.query.shape[1]//2, dims=1)))

    def test_degenerate_prior_and_zero_tensors_are_finite(self):
        prior = self.prior.clone()
        prior[:, 0] = 0
        prior[:, 1] = torch.tensor([1., 0., 0.])
        for mode in READ_MODES:
            actual = self.read(mode=mode, spatial_prior=prior)
            self.assertTrue(torch.isfinite(actual).all(), mode)
        zero = self.read(query_raw=torch.zeros_like(self.query),
                         reference_raw=torch.zeros_like(self.reference),
                         centers=torch.zeros_like(self.centers),
                         values=torch.zeros_like(self.values),
                         baseline=torch.zeros_like(self.baseline), spatial_prior=torch.zeros_like(prior))
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_coordinate_sampling_matches_patch_centers_and_border(self):
        # The local source and full grid share a 16-pixel patch lattice.
        grid = torch.arange(20*56, dtype=torch.float32).reshape(1, 20*56, 1)
        actual = _sample_global(grid, (20, 56), (320, 896), 0, 384)[0, :, 0].reshape(32, 32)
        rows = torch.arange(32).clamp_max(19)[:, None]
        cols = torch.arange(24, 56)[None]
        expected = (rows*56+cols).float()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-6)


class AnchoredSlotIntegrationTests(unittest.TestCase):
    def test_multimode_shares_frozen_local_path_and_matches_incumbent_controls(self):
        generator = torch.Generator().manual_seed(19)
        image = torch.zeros(3, 100, 280)
        resized = torch.zeros(3, 320, 896)
        raw = torch.randn(1, 6, 4, generator=generator)
        semantic = F.normalize(torch.randn(1, 6, 5, generator=generator), dim=-1)
        assignment = torch.randn(1, 6, 3, generator=generator).softmax(-1)
        raw_query = torch.randn(1, 1024, 4, generator=generator)
        local_features = torch.randn(1024, 5, generator=generator)
        operator = object()
        geometry, reconstruction = object(), object()
        config = ContextMemoryConfig(capacity=24, temperature=.1, iterations=5, center=False, hard=False)
        observer = SimpleNamespace(config=config, raw_patch_features=lambda _: raw,
            backbone=SimpleNamespace(_autocast=nullcontext,
                model=SimpleNamespace(visual_model=SimpleNamespace(head=object()))))
        calls = []

        def local_iterator(actual_resized, size, actual_geometry, strengths, **kwargs):
            self.assertIs(actual_resized, resized)
            self.assertEqual(size, (320, 896))
            self.assertIs(actual_geometry, geometry)
            self.assertEqual(strengths, [1.])
            self.assertIs(kwargs['reconstruction'], reconstruction)
            calls.append(kwargs['include_raw'])
            for left in (0, 384):
                tile = dict(top=0, left=left, features={1.: local_features}, operator=operator)
                if kwargs['include_raw']:
                    tile['raw'] = raw_query
                yield tile

        compact = ModuleType('dinotool.compact_deployment')
        compact._iter_local_observations = local_iterator
        bounded = ModuleType('dinotool.bounded_physical_coupling')
        bounded.resize_geometry = lambda _: resized
        context = ModuleType('dinotool.context_memory_deployment')
        # The 32x48 padded RGB retains an unpadded 25x43 extent. All readers
        # intentionally use the same incumbent full-padded-lattice map.
        context.memory_rgb = lambda _, **kwargs: (torch.zeros(3, 32, 48), (25, 43))
        stubs = {module.__name__: module for module in (compact, bounded, context)}
        with patch.dict(sys.modules, stubs), \
                patch('dinotool.anchored_slot_memory.context_assignments', return_value=assignment) as assign, \
                patch('dinotool.anchored_slot_memory.read_context_head', return_value=semantic) as head, \
                patch('dinotool.query_slot_memory.context_assignments', return_value=assignment), \
                patch('dinotool.query_slot_memory.read_context_head', return_value=semantic):
            outputs = anchored_slot_wide_observations(image, geometry, observer, [1.],
                reconstruction=reconstruction, modes=READ_MODES, audit=True)
            self.assertEqual(calls, [True])
            assign.assert_called_once_with(raw, config)
            head.assert_called_once()
            incumbent = query_slot_wide_observations(image, geometry, observer, [1.],
                                                     reconstruction=reconstruction)
            coordinate = anchored_slot_wide_observations(image, geometry, observer, [1.],
                reconstruction=reconstruction, mode='coordinate')
            beta0 = anchored_slot_wide_observations(image, geometry, observer, [1.],
                reconstruction=reconstruction, beta=0.)
            self.assertEqual(calls, [True, True, False, False])
        shared_local = outputs['anchored']['local']
        for name in READ_MODES:
            source = outputs[name]
            self.assertIs(source['local'], shared_local)
            self.assertTrue(source['anchored_audit']['legacy_match'])
            self.assertTrue(source['anchored_audit']['coordinate_beta0_match'])
            self.assertTrue(source['anchored_audit']['shared_local_identity'])
            self.assertEqual(source['anchored_audit']['checked_tiles'], 2)
            for i, tile in enumerate(source['local']):
                self.assertIs(tile['features'][1.], local_features)
                self.assertIs(tile['operator'], operator)
                self.assertNotIn('raw', tile)
                self.assertTrue(torch.equal(tile['features'][1.], incumbent['local'][i]['features'][1.]))
        for i in range(2):
            self.assertTrue(torch.equal(outputs['legacy']['wide'][i]['features'],
                                        incumbent['wide'][i]['features']))
            self.assertTrue(torch.equal(outputs['coordinate']['wide'][i]['features'],
                                        coordinate['wide'][i]['features']))
            self.assertTrue(torch.equal(coordinate['wide'][i]['features'], beta0['wide'][i]['features']))


if __name__ == '__main__':
    unittest.main()
