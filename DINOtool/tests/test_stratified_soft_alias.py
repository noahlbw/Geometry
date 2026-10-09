import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.soft_competitive_alias import leave_one_out_scores
from dinotool.stratified_soft_alias import (StratifiedAliasConfig, StratifiedSoftAliases,
    WideCrop, build_image_references, weighted_wide_pairs)
from test_pair_conditional_alias import example_bank


class StratifiedAliasTest(unittest.TestCase):
    def setUp(self):
        self.bank = example_bank()
        self.config = StratifiedAliasConfig(reference_pool_size=8, relation_neighbors=8,
                                             minimum_effective_reference=4, query_chunk=2)
        self.reader = StratifiedSoftAliases(self.bank, self.config)
        a = [.8, .65, .3, .2, .1, .1, .1, .1, -1., -1., -1., -1.]
        b = [.1, .65, .1, .1, .8, .7, .3, .2, -1., -1., -1., -1.]
        c = [.1] * 8 + [.8, .7, .3, .2]
        self.aliases = torch.tensor([a] * 8 + [b] * 8 + [c] * 8)
        self.features = torch.eye(3).repeat_interleave(8, 0)
        self.coordinates = torch.stack((torch.full((24,), 8.), 8 + 64 * torch.arange(24).float()), -1)
        self.image_size = (64, 2048)
        self.broad = alias_class_scores(self.aliases, self.bank.parent_indices, 3) / .07
        self.without = leave_one_out_scores(self.aliases, self.bank.parent_indices, 3) / .07
        self.references = self.build()
        self.query_features = torch.tensor([[1., 0., 0.]])
        self.query_coordinates = torch.tensor([[8., 2000.]])
        self.local = torch.tensor([[3., 2., -10.]])
        self.valid = torch.ones(1, dtype=torch.bool)

    def build(self, **changes):
        values = dict(features=self.features, coordinates=self.coordinates, local_aliases=self.aliases,
                      broad=self.broad, broad_without_alias=self.without, parents=self.bank.parent_indices,
                      classes=3, image_size=self.image_size, config=self.config)
        values.update(changes)
        return build_image_references(**values)

    def allocate(self, **changes):
        values = dict(features=self.query_features, coordinates=self.query_coordinates,
                      local=self.local, references=self.references, valid=self.valid)
        values.update(changes)
        return self.reader.allocate(**values)

    def test_stratified_retrieval_keeps_both_sides_despite_pure_query_feature(self):
        _, _, diagnostics = self.allocate()
        self.assertGreater(diagnostics["reference_supported_alias_fraction"], .5)
        self.assertGreater(diagnostics["mean_reference_trust"], 0)

    def test_discriminative_alias_outweighs_visual_confounder(self):
        _, weights, _ = self.allocate()
        self.assertGreater(float(weights["soft"][0, 0, 0]), float(weights["soft"][0, 0, 1]))

    def test_allocation_changes_with_rival(self):
        _, ab, _ = self.allocate()
        _, ac, _ = self.allocate(local=torch.tensor([[3., -10., 2.]]))
        self.assertGreater(float(ac["soft"][0, 0, 1]), float(ab["soft"][0, 0, 1]))

    def test_positive_normalized_weights_and_exact_shuffled_spectrum(self):
        _, weights, diagnostics = self.allocate()
        self.assertTrue(torch.allclose(weights["soft"].sum(-1), torch.ones(1, 2)))
        self.assertGreaterEqual(float(weights["soft"].min()), .5 / 4)
        self.assertEqual(diagnostics["shuffled_weight_spectrum_error"], 0.)

    def test_spatial_cells_are_not_counted_twice_per_pool(self):
        coordinates = self.coordinates.clone()
        coordinates[:8] = torch.tensor([8., 8.])
        references = self.build(coordinates=coordinates)
        for indices, valid in zip(references.pool_indices.flatten(0, 1), references.pool_valid.flatten(0, 1)):
            cells = references.cells[indices[valid]]
            self.assertEqual(len(cells), len(cells.unique()))

    def test_query_region_cannot_witness_itself(self):
        references = replace(self.references, cells=torch.full_like(self.references.cells, 31))
        _, weights, diagnostics = self.allocate(references=references)
        self.assertEqual(diagnostics["reference_supported_alias_fraction"], 0.)
        self.assertTrue(torch.equal(weights["soft"], torch.full((1, 2, 4), .25)))

    def test_absent_rival_reference_abstains(self):
        available = self.references.pool_valid.clone()
        available[:, 1] = False
        _, weights, diagnostics = self.allocate(references=replace(self.references, pool_valid=available))
        self.assertEqual(diagnostics["reference_supported_alias_fraction"], 0.)
        self.assertTrue(torch.equal(weights["soft"], torch.full((1, 2, 4), .25)))

    def test_invalid_query_abstains(self):
        _, weights, _ = self.allocate(valid=torch.zeros(1, dtype=torch.bool))
        self.assertTrue(torch.equal(weights["soft"], torch.full((1, 2, 4), .25)))

    def test_empty_reference_bank_abstains(self):
        references = self.build(broad=torch.zeros_like(self.broad), broad_without_alias=torch.zeros_like(self.without))
        _, weights, _ = self.allocate(references=references)
        self.assertFalse(bool(references.pool_valid.any()))
        self.assertTrue(torch.equal(weights["soft"], torch.full((1, 2, 4), .25)))

    def crops(self):
        torch.manual_seed(42)
        crops = [WideCrop(torch.randn(16, 12), torch.randn(12), 0, left, 16, 16, 4, 16) for left in (0, 8)]
        count = torch.zeros(16, 24)
        maps = torch.zeros(3, 16, 24)
        for crop in crops:
            count[:, crop.left:crop.left + 16] += 1
            scores = []
            for ids in self.reader.members:
                salience = crop.salience[ids].softmax(0)
                scaled = crop.alias_logits[:, ids] * (4 * salience)
                scores.append(scaled.logsumexp(-1).reshape(4, 4))
            dense = F.interpolate(torch.stack(scores)[None], (16, 16), mode="bilinear", align_corners=False)[0]
            maps[:, :, crop.left:crop.left + 16] += dense
        return crops, count, maps / count

    def test_stencil_matches_fresh_crop_aggregation_overlap_and_borders(self):
        crops, count, maps = self.crops()
        coordinates = torch.tensor([[.1, .1], [8., 8.], [80., 120.], [159.9, 239.9], [200., 300.]])
        pairs = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 0], [1, 0]])
        weights = torch.full((5, 2, 4), .25)
        actual = weighted_wide_pairs(crops, count, coordinates, (160, 240), pairs, weights, self.reader.members)
        grid = (coordinates / torch.tensor([160., 240.]) * 2 - 1).flip(-1)[None, None]
        expected = F.grid_sample(maps[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T.gather(-1, pairs)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6))

    def test_nonuniform_weighting_matches_fresh_dense_branch_readout(self):
        crops, count, _ = self.crops()
        weights = torch.tensor([[[.55, .15, .15, .15], [.1, .2, .3, .4]]])
        pairs = torch.tensor([[0, 1]])
        coordinates = torch.tensor([[80., 120.]])
        actual = weighted_wide_pairs(crops, count, coordinates, (160, 240), pairs, weights, self.reader.members)
        dense = torch.zeros(2, 16, 24)
        for crop in crops:
            scores = []
            for side, c in enumerate(pairs[0]):
                ids = self.reader.members[c]
                salience = crop.salience[ids].softmax(0) * weights[0, side]
                salience = salience / salience.sum()
                scores.append((crop.alias_logits[:, ids] * (4 * salience)).logsumexp(-1).reshape(4, 4))
            dense[:, :, crop.left:crop.left + 16] += F.interpolate(torch.stack(scores)[None], (16, 16), mode="bilinear", align_corners=False)[0]
        dense = dense / count
        grid = (coordinates / torch.tensor([160., 240.]) * 2 - 1).flip(-1)[None, None]
        expected = F.grid_sample(dense[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6))

    def test_uniform_read_is_exact_and_bounded_pair_update_preserves_partition(self):
        crops, count, _ = self.crops()
        reader = StratifiedSoftAliases(self.bank, replace(self.config, uniform_prior=1))
        variants, _ = reader.read(self.query_features, self.query_coordinates, self.local, self.local,
                                 self.references, crops, count, self.valid)
        self.assertTrue(all(torch.equal(value, self.local) for value in variants.values()))
        variants, diagnostics = self.reader.read(self.query_features, self.query_coordinates, self.local, self.local,
                                               self.references, crops, count, self.valid)
        self.assertLessEqual(diagnostics["maximum_absolute_margin_correction"], .5)
        self.assertLess(diagnostics["pair_partition_error"], 1e-6)
        self.assertEqual(float(variants["soft"][0, 2]), float(self.local[0, 2]))


if __name__ == "__main__":
    unittest.main()
