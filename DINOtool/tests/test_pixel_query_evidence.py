"""Contracts for the PSDR probe's conditional addressing controls."""
from __future__ import annotations

import pytest
import torch

from dinotool.pixel_query_evidence import PixelQueryEvidenceConfig, PixelQueryEvidenceReader, SharedGridEvidenceBank


def _reader(address="pixel_query", content="member_geometry"):
    config = PixelQueryEvidenceConfig(address_mode=address, content_mode=content, dim=32, heads=4,
        grids=(4, 2), regions_per_read=2, samples_per_region=2, rounds=2, query_chunk=2, visual_blocks=1)
    return PixelQueryEvidenceReader(16, config), config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(20260922)
    torch.set_num_threads(2)


def test_shared_bank_has_identical_geometry_for_all_text_queries():
    bank = SharedGridEvidenceBank((4, 2), 2)(torch.randn(2, 32, 5, 7))
    assert bank["mean"].shape == (2, 20, 32)
    assert bank["members"].shape == (2, 20, 2, 32)
    assert bank["centers"].shape == (20, 2)
    assert bank["scales"].shape == (20, 2)


@pytest.mark.parametrize("address", ("image_only", "query_only", "pixel_only", "pixel_query"))
@pytest.mark.parametrize("content", ("region_mean", "member_geometry", "flat_attention"))
def test_every_control_preserves_cost_shape_and_has_finite_gradients(address, content):
    reader, config = _reader(address, content)
    field = torch.randn(2, config.dim, 4, 6, requires_grad=True)
    bank = SharedGridEvidenceBank(config.grids, config.samples_per_region)(field)
    state = torch.randn(2, 16, 3, 4, 6, requires_grad=True)
    initial = torch.randn_like(state)
    text = torch.randn(3, 1024)
    result = reader(state, initial, field, bank, text, return_candidates=True)
    assert result["cost"].shape == state.shape
    assert result["utility_scores"].shape == (2, 3, 24, config.regions_per_read)
    assert result["candidate_costs"].shape == (config.regions_per_read, 2, 16, 3, 4, 6)
    loss = result["cost"].square().mean() + result["utility_scores"].square().mean()
    loss.backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_address_controls_have_registered_invariances():
    field = torch.randn(1, 32, 4, 4)
    text = torch.randn(3, 1024)
    for mode, equal_pixels, equal_queries in (("image_only", True, True), ("query_only", True, False), ("pixel_only", False, True)):
        reader, _ = _reader(mode)
        request = reader._request(field, text)
        if equal_pixels:
            torch.testing.assert_close(request[:, :, 0], request[:, :, -1])
        else:
            assert not torch.allclose(request[:, :, 0], request[:, :, -1])
        if equal_queries:
            torch.testing.assert_close(request[:, 0], request[:, -1])
        else:
            assert not torch.allclose(request[:, 0], request[:, -1])


def test_zero_initialized_writer_keeps_the_base_cost_exactly():
    reader, config = _reader()
    field = torch.randn(1, config.dim, 4, 4)
    bank = SharedGridEvidenceBank(config.grids, config.samples_per_region)(field)
    state = torch.randn(1, 16, 2, 4, 4)
    result = reader(state, torch.randn_like(state), field, bank, torch.randn(2, 1024))
    torch.testing.assert_close(result["cost"], state, rtol=0, atol=0)
