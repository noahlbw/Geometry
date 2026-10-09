"""Correctness checks for the single complete cross-scale readout."""
from types import SimpleNamespace

import torch
from torch import nn

from dinotool.cross_scale_geometry import (
    CrossScaleConfig, aggregate_fine_reference, four_to_grid, grid_to_four,
    lift_relative_relations, relation_logits, reread_fine_head,
)
from dinotool.tcpr import _intervened_head


def fixture():
    torch.manual_seed(31)
    raw = nn.functional.normalize(torch.randn(12, 8), dim=-1)
    xy = torch.rand(12, 2)
    footprint = torch.arange(12) // 3
    valid = torch.ones(12, dtype=torch.bool)
    config = CrossScaleConfig(query_chunk=5)
    return raw, xy, footprint, valid, config


def test_grid_round_trip():
    packed = torch.arange(4 * 9 * 7).reshape(4, 9, 7)
    torch.testing.assert_close(grid_to_four(four_to_grid(packed)), packed)


def test_reference_matches_direct_aggregation():
    raw, xy, footprint, valid, config = fixture()
    valid[2] = False
    base = relation_logits(raw, raw, xy, xy, valid, config).softmax(-1)
    expected = torch.stack([torch.stack([
        base[(footprint == a) & valid][:, footprint == b].sum(-1).mean()
        for b in range(4)]) for a in range(4)])
    reference = aggregate_fine_reference(raw, xy, footprint, valid, 4, config)
    torch.testing.assert_close(reference, expected)
    torch.testing.assert_close(reference.sum(-1), torch.ones(4))


def test_zero_increment_and_zero_availability_are_identity():
    raw, xy, footprint, valid, config = fixture()
    base = relation_logits(raw, raw, xy, xy, valid, config).softmax(-1)
    for increment, available in ((torch.zeros(4, 4), torch.rand(4)),
                                 (torch.randn(4, 4), torch.zeros(4))):
        changed, diagnostics = lift_relative_relations(
            raw, xy, footprint, footprint, valid, increment, available, config)
        torch.testing.assert_close(changed, base, atol=1e-7, rtol=1e-5)
        assert abs(diagnostics["relation_kl"]) < 1e-6


def test_invalid_donors_and_within_footprint_conditionals():
    raw, xy, footprint, valid, config = fixture()
    valid[1] = False
    base = relation_logits(raw, raw, xy, xy, valid, config).softmax(-1)
    changed, _ = lift_relative_relations(raw, xy, footprint, footprint, valid,
                                         torch.randn(4, 4), torch.rand(4), config)
    assert not bool(changed[:, ~valid].any())
    torch.testing.assert_close(changed.sum(-1), torch.ones(12))
    for group in range(4):
        selected = (footprint == group) & valid
        torch.testing.assert_close(changed[:, selected] / changed[:, selected].sum(-1, keepdim=True),
                                   base[:, selected] / base[:, selected].sum(-1, keepdim=True),
                                   atol=2e-6, rtol=2e-5)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(12), nn.LayerNorm(12)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(12, 36)
        self.attn.num_heads, self.attn.scale = 3, .5
        self.attn.proj, self.attn.proj_drop = nn.Linear(12, 12), nn.Identity()
        self.ls1, self.ls2 = nn.Identity(), nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12))


def test_crop_prefix_and_two_block_values_match_geometry():
    torch.manual_seed(7)
    head = SimpleNamespace(blocks=nn.ModuleList([Block(), Block()]),
                           ln_final=nn.LayerNorm(12), linear_projection=nn.Linear(12, 9))
    tokens = torch.randn(4, 6, 12)  # two prefix + 2x2 patches per origin crop
    geometry = torch.randn(4, 4, 4).softmax(-1)
    owner = four_to_grid(torch.arange(4)[:, None, None].expand(-1, 4, 1)).flatten()
    relation = torch.zeros(16, 16)
    for crop in range(4):
        indices = (owner == crop).nonzero().flatten()
        relation[indices[:, None], indices[None]] = geometry[crop]
    with torch.no_grad():
        expected = _intervened_head(head, tokens, geometry, 2, 1, 2, "preserve")[:, 2:]
        expected = nn.functional.normalize(four_to_grid(expected), dim=-1)
        actual = reread_fine_head(head, tokens, relation, 2, 3)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    checks = [value for name, value in list(globals().items())
              if name.startswith("test_") and callable(value)]
    for check in checks:
        check()
    print(f"{len(checks)} complete-model correctness checks passed.")
