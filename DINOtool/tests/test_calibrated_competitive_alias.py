import unittest
from dataclasses import replace

import torch

from dinotool.calibrated_competitive_alias import (CalibratedCompetitiveAliases, PRIMARY, SHUFFLED,
    action_and_group_scores, calibrated_wide_pairs, group_references, profiled_logits, text_holdouts)
from dinotool.stratified_soft_alias import weighted_wide_pairs
from test_stratified_soft_alias import StratifiedAliasTest


class CalibratedAliasTest(unittest.TestCase):
    def setUp(self):
        self.fixture = StratifiedAliasTest()
        self.fixture.setUp()
        self.reader = CalibratedCompetitiveAliases(self.fixture.bank, self.fixture.config)
        self.crops, self.count, _ = self.fixture.crops()
        self.coordinates = torch.tensor([[80., 120.]])
        self.image_size = (160, 240)
        self.pairs = torch.tensor([[0, 1]])

    def test_uniform_writer_replays_original_stencil(self):
        weights = torch.full((1, 2, 4), .25)
        before = weighted_wide_pairs(self.crops, self.count, self.coordinates, self.image_size,
                                     self.pairs, weights, self.reader.members)
        after = calibrated_wide_pairs(self.crops, self.count, self.coordinates, self.image_size,
                                      self.pairs, weights, self.reader.members)
        self.assertTrue(torch.allclose(before, after, atol=2e-6))

    def test_constant_profiled_evidence_is_invariant_to_weight_dispersion(self):
        crops = [replace(crop, alias_logits=torch.full_like(crop.alias_logits, 3.),
                         salience=torch.zeros_like(crop.salience)) for crop in self.crops]
        uniform = torch.full((1, 2, 4), .25)
        varied = torch.tensor([[[.55, .15, .15, .15], [.1, .2, .3, .4]]])
        outputs = [calibrated_wide_pairs(crops, self.count, self.coordinates, self.image_size,
                    self.pairs, weights, self.reader.members) for weights in (uniform, varied)]
        self.assertTrue(torch.allclose(*outputs, atol=1e-6))

    def test_single_attenuation_marginal_matches_fresh_weighted_writer(self):
        marginals, _ = action_and_group_scores(self.crops, self.count, self.coordinates,
                self.image_size, self.reader.members, self.reader.holdouts)
        uniform = torch.full((1, 2, 4), .25)
        original = calibrated_wide_pairs(self.crops, self.count, self.coordinates, self.image_size,
                    self.pairs, uniform, self.reader.members)
        for side, c in enumerate(self.pairs[0]):
            for index, alias in enumerate(self.reader.members[c]):
                weights = uniform.clone()
                weights[0, side, index] *= .5
                weights[0, side] /= weights[0, side].sum()
                changed = calibrated_wide_pairs(self.crops, self.count, self.coordinates, self.image_size,
                            self.pairs, weights, self.reader.members)
                self.assertAlmostEqual(float(original[0, side] - changed[0, side]), float(marginals[0, alias]), places=5)

    def test_holdout_finds_identical_alias_in_another_class(self):
        features = torch.eye(5)
        features[4] = features[0]
        mask = text_holdouts(features, 2)
        self.assertTrue(mask[0, 0] and mask[0, 4])
        self.assertTrue(torch.equal(mask.sum(-1), torch.full((5,), 2)))

    def test_group_witness_is_independent_of_removed_responses(self):
        _, before = action_and_group_scores(self.crops, self.count, self.coordinates,
                self.image_size, self.reader.members, self.reader.holdouts)
        removed = self.reader.holdouts[0]
        crops = []
        for crop in self.crops:
            logits = crop.alias_logits.clone()
            logits[:, removed] += 100
            salience = crop.salience.clone()
            salience[removed] += 100
            crops.append(replace(crop, alias_logits=logits, salience=salience))
        _, after = action_and_group_scores(crops, self.count, self.coordinates,
                self.image_size, self.reader.members, self.reader.holdouts)
        self.assertTrue(torch.equal(before[:, 0], after[:, 0]))

    def test_group_reference_winner_ignores_removed_local_responses(self):
        fixture = self.fixture
        holdouts = self.reader.holdouts
        group = torch.zeros(24, 12, 3)
        group[:8, :, 0], group[8:16, :, 1], group[16:, :, 2] = 10, 10, 10
        marginal = torch.ones(24, 12)
        first = group_references(fixture.features, fixture.coordinates, fixture.aliases, group, marginal,
                                 self.reader.members, holdouts, fixture.image_size, fixture.config)
        changed = fixture.aliases.clone()
        changed[:, holdouts[0]] += 100
        second = group_references(fixture.features, fixture.coordinates, changed, group, marginal,
                                  self.reader.members, holdouts, fixture.image_size, fixture.config)
        self.assertTrue(torch.equal(first.pool_indices[0], second.pool_indices[0]))
        self.assertTrue(torch.equal(first.pool_valid[0], second.pool_valid[0]))

    def test_missing_reference_abstains_with_exact_output(self):
        fixture = self.fixture
        empty = replace(fixture.references, pool_valid=torch.zeros_like(fixture.references.pool_valid))
        outputs, _ = self.reader.read(fixture.query_features, fixture.query_coordinates, fixture.local, fixture.local,
            empty, empty, self.crops, self.count, fixture.valid)
        for name in (PRIMARY, *SHUFFLED):
            self.assertTrue(torch.equal(outputs[name], fixture.local))

    def test_corrected_pairs_preserve_partition_and_shuffle_spectra(self):
        fixture = self.fixture
        outputs, diagnostics = self.reader.read(fixture.query_features, fixture.query_coordinates, fixture.local,
            fixture.local, fixture.references, fixture.references, self.crops, self.count, fixture.valid)
        for name, values in outputs.items():
            self.assertLess(diagnostics[name + "_partition_error"], 1e-6)
            self.assertEqual(float(values[0, 2]), float(fixture.local[0, 2]))
        for name in SHUFFLED:
            self.assertEqual(diagnostics[name + "_spectrum_error"], 0.)


if __name__ == "__main__":
    unittest.main()
