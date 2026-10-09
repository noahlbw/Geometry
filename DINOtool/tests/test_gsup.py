import torch

from dinotool.config import GSUPConfig
from dinotool.gsup import GaussianSplatUpsampler


def test_gsup_preserves_constant_values() -> None:
    upsampler = GaussianSplatUpsampler(
        GSUPConfig(steps=0, neighbors=4, optimization_size=32, chunk_pixels=128)
    )
    guidance = torch.full((1, 3, 24, 32), 0.4)
    values = torch.full((1, 2, 4, 5), 2.5)
    parameters, _ = upsampler.fit(guidance, values.shape[-2:])
    result = upsampler.upsample(values, guidance, parameters)
    assert result.shape == (1, 2, 24, 32)
    assert torch.allclose(result, torch.full_like(result, 2.5), atol=1e-5)


def test_gsup_tto_reduces_rgb_reconstruction_error() -> None:
    guidance = torch.zeros((1, 3, 32, 32))
    guidance[:, 0, :, 16:] = 1.0
    guidance[:, 1, 16:, :] = 0.7
    upsampler = GaussianSplatUpsampler(
        GSUPConfig(
            steps=4,
            learning_rate=0.08,
            neighbors=16,
            optimization_size=32,
            chunk_pixels=256,
        )
    )
    _, diagnostics = upsampler.fit(guidance, (4, 4))
    assert diagnostics.final_reconstruction_l1 <= diagnostics.initial_reconstruction_l1 + 1e-6

