import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

import torch

from dinotool.geometry_unit_readout import UnitConfig, detail_lift, read_unit_geometry, reencode_units, unit_labels, unit_operators
from dinotool.tcpr import _intervened_head

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
if UPSTREAM.is_dir():
    sys.path.insert(0, str(UPSTREAM))
from dinov3.layers.block import SelfAttentionBlock


class UnitReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.geometry = torch.randn(16, 16).softmax(-1)
        self.labels = torch.arange(16) // 4

    def test_pool_lift_identity_and_unit_rows(self):
        pool, lift = unit_operators(self.geometry, self.labels)
        torch.testing.assert_close(pool @ lift, torch.eye(4), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pool.sum(-1), torch.ones(4))
        self.assertTrue(bool((pool >= 0).all()))

    def test_lift_reconstructs_unit_means(self):
        pool, lift = unit_operators(self.geometry, self.labels)
        fine, coarse = torch.randn(16, 7), torch.randn(4, 7)
        output = detail_lift(fine, coarse, pool, lift)
        torch.testing.assert_close(pool @ output, coarse, atol=1e-6, rtol=1e-6)

    def test_fine_within_unit_differences_retained(self):
        pool, lift = unit_operators(self.geometry, self.labels)
        fine = torch.randn(16, 7)
        output = detail_lift(fine, torch.randn(4, 7), pool, lift)
        for index in range(0, 16, 4):
            torch.testing.assert_close(output[index:index+4]-output[index], fine[index:index+4]-fine[index], atol=1e-6, rtol=1e-6)

    def test_no_semantic_innovation_leaves_fine_field(self):
        pool, lift = unit_operators(self.geometry, self.labels)
        fine = torch.randn(16, 7)
        self.assertTrue(torch.equal(detail_lift(fine, pool @ fine, pool, lift), fine))

    def test_invalid_positions_are_never_written(self):
        self.labels[-3:] = -1
        pool, lift = unit_operators(self.geometry, self.labels)
        fine = torch.randn(16, 7)
        output = detail_lift(fine, torch.randn(4, 7), pool, lift)
        self.assertTrue(torch.equal(output[-3:], fine[-3:]))
        self.assertTrue(bool((pool[:, -3:] == 0).all()))

    def test_coarse_relation_is_stochastic(self):
        pool, lift = unit_operators(self.geometry, self.labels)
        relation = pool @ self.geometry @ lift
        torch.testing.assert_close(relation.sum(-1), torch.ones(4), atol=1e-6, rtol=1e-6)

    def test_exact_unit_budget_and_connected_groups(self):
        labels = unit_labels(self.geometry, torch.ones(16, dtype=torch.bool), (4, 4), UnitConfig(2))
        self.assertEqual(len(torch.unique(labels)), 4)
        for group in torch.unique(labels):
            indices = set((labels == group).nonzero().flatten().tolist())
            visited, frontier = set(), [next(iter(indices))]
            while frontier:
                current = frontier.pop()
                if current in visited:
                    continue
                visited.add(current)
                y, x = divmod(current, 4)
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    neighbor = yy*4+xx
                    if 0 <= yy < 4 and 0 <= xx < 4 and neighbor in indices and neighbor not in visited:
                        frontier.append(neighbor)
            self.assertEqual(visited, indices)

    def test_small_support_is_singletons_not_forced_merging(self):
        valid = torch.zeros(16, dtype=torch.bool)
        valid[:3] = True
        labels = unit_labels(self.geometry, valid, (4, 4), UnitConfig(2))
        self.assertEqual(labels[:3].tolist(), [0, 1, 2])
        self.assertTrue(bool((labels[3:] == -1).all()))

    def test_disconnected_supports_are_not_merged_across_padding(self):
        valid = torch.zeros(16, dtype=torch.bool)
        valid[[0, 3, 12, 15]] = True
        labels = unit_labels(self.geometry, valid, (4, 4), UnitConfig(1))
        self.assertEqual(len(torch.unique(labels[valid])), 4)

    def test_spatial_control_has_same_budget(self):
        labels = unit_labels(self.geometry, torch.ones(16, dtype=torch.bool), (4, 4), UnitConfig(2), spatial=True)
        self.assertEqual(len(torch.unique(labels)), 4)

    def test_all_invalid_is_empty_neutral_support(self):
        labels = unit_labels(self.geometry, torch.zeros(16, dtype=torch.bool), (4, 4))
        pool, lift = unit_operators(self.geometry, labels)
        self.assertEqual(pool.shape, (0, 16))
        self.assertEqual(lift.shape, (16, 0))

    def test_malformed_geometry_and_noncontiguous_labels_rejected(self):
        with self.assertRaises(ValueError):
            unit_labels(self.geometry*float("nan"), torch.ones(16, dtype=torch.bool), (4, 4))
        with self.assertRaises(ValueError):
            unit_operators(self.geometry, self.labels+1)

    def test_actual_frozen_blocks_unit_reencoding_and_cache(self):
        blocks = [SelfAttentionBlock(16, 4, ffn_ratio=2., init_values=.2).eval().requires_grad_(False)
                  for _ in range(2)]
        for block in blocks:
            block.ls1.reset_parameters()
            block.ls2.reset_parameters()
        head = SimpleNamespace(blocks=blocks, ln_final=torch.nn.LayerNorm(16).requires_grad_(False),
                               linear_projection=torch.nn.Identity())
        tokens = torch.randn(1, 18, 16)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, block_index=1,
                                   geometry_patch_conditional=self.geometry[None])
        fine = _intervened_head(head, tokens, self.geometry[None], 2, 1, 2, "preserve")[:, 2:]
        cache = tokens.clone()
        output, only, stats = reencode_units(head, prepared, self.labels, fine)
        self.assertEqual(output.shape, fine.shape)
        self.assertEqual(only.shape, fine.shape)
        self.assertLess(stats["unit_mean_max_error"], 1e-5)
        self.assertTrue(torch.equal(tokens, cache))
        identity, _, _ = reencode_units(head, prepared, torch.arange(16), fine)
        self.assertTrue(torch.equal(identity, fine))

    def test_control_operators_receive_exact_unnormalized_backbone_patches(self):
        blocks = [SelfAttentionBlock(16, 4, ffn_ratio=2., init_values=.2).eval().requires_grad_(False)
                  for _ in range(2)]
        for block in blocks:
            block.ls1.reset_parameters()
            block.ls2.reset_parameters()
        head = SimpleNamespace(blocks=blocks, ln_final=torch.nn.LayerNorm(16).requires_grad_(False),
                               linear_projection=torch.nn.Identity())
        tokens = 3 * torch.randn(1, 18, 16)
        fine = _intervened_head(head, tokens, self.geometry[None], 2, 1, 2, "preserve")[:, 2:].float()
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, block_index=1,
            geometry_patch_conditional=self.geometry[None], grid_height=4, grid_width=4,
            raw_patch_tokens=torch.nn.functional.normalize(tokens[:, 2:], dim=-1),
            geometry_projected=torch.nn.functional.normalize(fine, dim=-1))
        backbone = SimpleNamespace(_autocast=nullcontext, model=SimpleNamespace(visual_model=SimpleNamespace(head=head)))
        def control(head, original, raw, *args):
            self.assertTrue(torch.equal(raw, tokens[:, 2:]))
            return prepared.geometry_projected, {}
        with patch("dinotool.geometry_unit_readout.run_head", side_effect=control) as mocked:
            read_unit_geometry(SimpleNamespace(backbone=backbone), prepared, torch.ones(1, 16, dtype=torch.bool))
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
