import pytest
import torch

from dinotool.tcpr import TCPRConfig, _geometry_attended


def test_prefix_block_preserves_only_patch_geometry_for_patch_queries():
    values = torch.tensor([[[[10.0], [20.0], [1.0], [3.0]]]])
    attention = torch.tensor([[[
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.50, 0.00, 0.25, 0.25],
    ]]])
    geometry = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
    preserved = _geometry_attended(attention, geometry, values, prefix=2)
    blocked = _geometry_attended(attention, geometry, values, prefix=2, prefix_policy="block")
    torch.testing.assert_close(blocked[:, :, :2], preserved[:, :, :2])
    torch.testing.assert_close(blocked[0, 0, 2:, 0], torch.tensor([1.5, 2.5]))
    assert torch.all(preserved[0, 0, 2:, 0] > blocked[0, 0, 2:, 0])


@pytest.mark.parametrize("kwargs", [
    {"geometry_depth": 3},
    {"prefix_policy": "drop_all"},
])
def test_invalid_visual_head_ablation_rejected(kwargs):
    with pytest.raises(ValueError):
        TCPRConfig(**kwargs).validate()
