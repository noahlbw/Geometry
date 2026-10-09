import unittest
from unittest.mock import patch

import torch

from dinotool.region_semantic_readout import (
    RegionReadoutConfig, joint_energy, joint_readout, region_crops,
    restricted_pool, semantic_probability, support_weights)


class Pool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.randn(1, 1, 8))
        self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.layernorm = torch.nn.LayerNorm(8)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))

    def forward(self, x):
        value = self.attention(self.probe.expand(len(x), -1, -1), x, x)[0]
        return (value+self.mlp(self.layernorm(value)))[:, 0]


class RegionalReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_full_support_replays_trained_pool(self):
        head, hidden = Pool(), torch.randn(3, 7, 8)
        torch.testing.assert_close(restricted_pool(head, hidden, torch.ones(3, 7)), head(hidden))

    def test_masked_pool_ignores_outside_values(self):
        head, hidden = Pool(), torch.randn(2, 7, 8)
        support = torch.zeros(2, 7); support[:, :2] = 1
        changed = hidden.clone(); changed[:, 2:] += 100
        torch.testing.assert_close(restricted_pool(head, hidden, support), restricted_pool(head, changed, support))

    def test_empty_support_is_rejected(self):
        with self.assertRaises(ValueError):
            restricted_pool(Pool(), torch.randn(1, 3, 8), torch.zeros(1, 3))

    def test_uniform_semantics_have_zero_influence(self):
        scores = torch.zeros(3, 5)
        q, influence, _ = semantic_probability(scores, scores, torch.ones(3, dtype=torch.bool), .07)
        torch.testing.assert_close(q, torch.full((3, 5), .2))
        torch.testing.assert_close(influence, torch.zeros(3), atol=1e-6, rtol=0)

    def test_no_evidence_is_exact_geometry(self):
        local = torch.randn(8, 3)
        output, _, _ = joint_readout(local, torch.full((2, 3), 1/3), torch.zeros(2, 8), RegionReadoutConfig())
        self.assertTrue(torch.equal(local, output))

    def test_solver_energy_decreases_and_probabilities_are_valid(self):
        local = torch.randn(9, 4)*.03
        q, weights = torch.randn(3, 4).softmax(-1), torch.rand(3, 9)/3
        config = RegionReadoutConfig(solver_iterations=100)
        output, z, info = joint_readout(local, q, weights, config)
        p = (output/config.alias_temperature).softmax(-1)
        self.assertLessEqual(info["energy_after"], info["energy_before"]+1e-5)
        torch.testing.assert_close(z.sum(-1), torch.ones(3))
        energy = joint_energy(p, z, (local/.07).softmax(-1), q, weights)
        self.assertAlmostEqual(float(energy), info["energy_after"], places=4)

    def test_permutation_preserves_competition(self):
        local, q = torch.randn(7, 3)*.05, torch.randn(2, 3).softmax(-1)
        weights, permutation = torch.rand(2, 7), torch.tensor([2, 0, 1])
        first, _, _ = joint_readout(local, q, weights, RegionReadoutConfig())
        second, _, _ = joint_readout(local[:, permutation], q[:, permutation], weights, RegionReadoutConfig())
        torch.testing.assert_close(first[:, permutation], second)

    def test_single_token_support_and_padding_are_preserved(self):
        valid = torch.ones(4, 4, dtype=torch.bool); valid[-1, -1] = False
        fine = torch.zeros(4, 4, dtype=torch.long); fine[0, 0] = 1; fine[-1, -1] = -1
        coarse = torch.zeros_like(fine); coarse[-1, -1] = -1
        with patch("dinotool.region_semantic_readout.partition_supports", side_effect=[fine, coarse]):
            operator, weights, masks, _ = support_weights(torch.randn(16, 8),
                torch.ones(16, 16)/16, valid, RegionReadoutConfig())
        self.assertEqual(int((masks.flatten(1).sum(-1) == 1).sum()), 1)
        self.assertTrue(bool((operator.masked_select(~masks.flatten(1)) == 0).all()))
        torch.testing.assert_close(weights.sum(0), valid.flatten().float())

    def test_small_border_support_remains_in_both_crop_coordinates(self):
        masks = torch.zeros(3, 32, 32, dtype=torch.bool)
        for index, position in enumerate((0, 15, 31)):
            masks[index, position, position] = True
        operator = masks.flatten(1).float()
        rgb = torch.rand(1, 3, 512, 512)
        crops, weights = region_crops(rgb, operator, masks, RegionReadoutConfig())
        self.assertEqual(crops.shape, (3, 3, 256, 256))
        self.assertTrue(bool((weights.sum(-1) > 0).all()))
        self.assertTrue(bool(torch.isfinite(crops).all()))


if __name__ == "__main__":
    unittest.main()
