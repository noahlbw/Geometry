import torch

from dinotool.parallel_readout import ParallelReadoutConfig, patch_relation_weights, structural_logits


def _native_attention(batch: int = 2, heads: int = 3, tokens: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn((batch, heads, tokens, tokens), generator=torch.Generator().manual_seed(7))
    return torch.softmax(logits, dim=-1), logits


def test_native_and_zero_joint_are_exact_noops() -> None:
    native, logits = _native_attention()
    geometry = torch.randn((2, 5, 5), generator=torch.Generator().manual_seed(8))

    assert patch_relation_weights(native, logits, geometry, 2, ParallelReadoutConfig(mode="native")) is native
    assert patch_relation_weights(
        native, logits, geometry, 2, ParallelReadoutConfig(mode="joint", beta=0.0)
    ) is native


def test_intervention_preserves_special_entries_and_patch_mass() -> None:
    native, logits = _native_attention()
    geometry = torch.randn((2, 5, 5), generator=torch.Generator().manual_seed(9))
    output = patch_relation_weights(native, logits, geometry, 2, ParallelReadoutConfig(mode="joint", beta=0.5))

    assert torch.equal(output[..., :2, :], native[..., :2, :])
    assert torch.equal(output[..., 2:, :2], native[..., 2:, :2])
    assert torch.allclose(output[..., 2:, 2:].sum(-1), native[..., 2:, 2:].sum(-1), atol=1e-6)
    assert torch.allclose(output.sum(-1), torch.ones_like(output.sum(-1)), atol=1e-6)


def test_structural_logits_respect_grid_and_have_diagonal_preference() -> None:
    features = torch.zeros((1, 4, 2))
    features[0, :, 0] = 1.0
    geometry = structural_logits(features, 2, 2, temperature=0.1, spatial_sigma=0.25)

    assert geometry.shape == (1, 4, 4)
    off_diagonal = geometry[0].masked_fill(torch.eye(4, dtype=torch.bool), -torch.inf).max(1).values
    assert torch.all(torch.diagonal(geometry[0]) > off_diagonal)
