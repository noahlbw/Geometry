import torch

from dinotool.ov_adapter import DenseAdapterConfig, DenseTextAdapter


def test_dense_adapter_starts_as_the_lvd_identity_and_accepts_gradients() -> None:
    config = DenseAdapterConfig(feature_dim=8, hidden_dim=16, context_blocks=1, use_satellite_features=True)
    adapter = DenseTextAdapter(config)
    lvd = torch.nn.functional.normalize(torch.randn(2, 8, 3, 4), dim=1)
    satellite = torch.nn.functional.normalize(torch.randn(2, 8, 3, 4), dim=1)

    output = adapter(lvd, satellite)
    assert torch.allclose(output, lvd, atol=1e-6)
    output.square().mean().backward()
    assert adapter.output_projection.weight.grad is not None


def test_dense_adapter_requires_satellite_features_when_configured() -> None:
    adapter = DenseTextAdapter(DenseAdapterConfig(feature_dim=8, hidden_dim=16, use_satellite_features=True))
    lvd = torch.randn(1, 8, 2, 2)
    try:
        adapter(lvd)
    except ValueError as error:
        assert "satellite_features" in str(error)
    else:
        raise AssertionError("Expected missing satellite feature validation")
