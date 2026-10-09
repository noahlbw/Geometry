"""Numerical contracts for Q-Lift analysis, synthesis, and conditioning."""
from __future__ import annotations

import pytest
import torch

from dinotool.qlift import QLiftConfig, QLiftDecoder


def config(arm: str, *, query_chunk: int = 3) -> QLiftConfig:
    return QLiftConfig(arm=arm, levels=2, guide_dim=32, hidden_dim=64, kernel=3, query_chunk=query_chunk)


def inputs(*, queries: int = 3, height: int = 7, width: int = 11):
    return (
        torch.randn(1, 8, queries, height, width),
        torch.randn(1, 1024, height, width),
        torch.randn(1, 1024, height, width),
        torch.randn(queries, 1024),
    )


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(291)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False):
        yield


@pytest.mark.parametrize("arm", ("skip", "haar", "image_lift", "query_lift"))
def test_zero_start_reconstructs_odd_rectangular_cost_grid(arm):
    decoder = QLiftDecoder(8, config(arm)).eval()
    cost, x, y, text = inputs()
    with torch.no_grad():
        output, trace = decoder(cost, x, y, text, return_trace=True)
    torch.testing.assert_close(output, cost, rtol=2e-5, atol=2e-6)
    assert trace["max_cost_change"] < 2e-5


def test_every_detail_stream_changes_the_cached_reconstruction():
    decoder = QLiftDecoder(8, config("query_lift"))
    level = decoder.levels[0]
    assert level.lifting is not None
    value = torch.randn(1, 8, 8, 10)
    guide = torch.randn(1, 32, 4, 5)
    text = torch.randn(1, 1024)
    approximation, details, cache = level.lifting.analyze(value, guide, text, query_conditioned=True)
    baseline = level.lifting.synthesize(approximation, details, cache)
    torch.testing.assert_close(baseline, value, rtol=2e-5, atol=2e-6)
    for stream in range(3):
        changed = details.clone()
        changed[:, stream, :, 1, 2] += 0.5
        assert not torch.allclose(level.lifting.synthesize(approximation, changed, cache), baseline)


def test_query_condition_changes_only_query_lift_pu_weights():
    image = QLiftDecoder(8, config("image_lift"))
    query = QLiftDecoder(8, config("query_lift"))
    guide = torch.randn(2, 32, 5, 7)
    text = torch.randn(2, 1024)
    other = torch.randn_like(text)
    assert image.levels[0].lifting is not None and query.levels[0].lifting is not None
    with torch.no_grad():
        p0, u0 = image.levels[0].lifting._weights(guide, text, query_conditioned=False)
        p1, u1 = image.levels[0].lifting._weights(guide, other, query_conditioned=False)
        pq0, uq0 = query.levels[0].lifting._weights(guide, text, query_conditioned=True)
        pq1, uq1 = query.levels[0].lifting._weights(guide, other, query_conditioned=True)
    torch.testing.assert_close(p0, p1, rtol=0, atol=0)
    torch.testing.assert_close(u0, u1, rtol=0, atol=0)
    assert (pq0 - pq1).abs().max() > 1e-6
    assert (uq0 - uq1).abs().max() > 1e-6


def test_pu_counterfactual_keeps_shared_semantic_updates_fixed():
    cost, x, y, text = inputs()
    decoder = QLiftDecoder(8, config("query_lift")).eval()
    image_only = QLiftDecoder(8, config("image_lift")).eval()
    with torch.no_grad():
        for model in (decoder, image_only):
            for level in model.levels:
                level.context.output.weight.normal_(std=0.02)
                level.detail.output.weight.normal_(std=0.02)
        native, _ = decoder(cost, x, y, text)
        shuffled, _ = decoder(cost, x, y, text, lifting_text=text.roll(1, 0))
        no_pu_text, _ = decoder(cost, x, y, text, lifting_query_conditioned=False)
        no_details, _ = decoder(cost, x, y, text, detail_mode="zero")
        image_native, _ = image_only(cost, x, y, text)
        image_other, _ = image_only(cost, x, y, text, lifting_text=text.roll(1, 0))
    assert (native - shuffled).abs().max() > 1e-6
    assert (native - no_pu_text).abs().max() > 1e-6
    assert (native - no_details).abs().max() > 1e-6
    torch.testing.assert_close(image_native, image_other, rtol=0, atol=0)


def test_factorial_pu_and_gain_conditions_are_separable_and_native_is_preserved():
    cost, x, y, text = inputs()
    decoder = QLiftDecoder(8, config("query_lift")).eval()
    with torch.no_grad():
        for level in decoder.levels:
            level.context.output.weight.normal_(std=0.02)
            level.detail.output.weight.normal_(std=0.02)
        native, _ = decoder(cost, x, y, text)
        explicit_native, _ = decoder(
            cost, x, y, text, lifting_query_conditioned=True, gain_query_conditioned=True,
        )
        pu_query_gain_image, _ = decoder(
            cost, x, y, text, lifting_query_conditioned=True, gain_query_conditioned=False,
        )
        pu_image_gain_query, _ = decoder(
            cost, x, y, text, lifting_query_conditioned=False, gain_query_conditioned=True,
        )
        pu_image_gain_image, _ = decoder(
            cost, x, y, text, lifting_query_conditioned=False, gain_query_conditioned=False,
        )
        legacy_image_only, _ = decoder(cost, x, y, text, lifting_query_conditioned=False)
    torch.testing.assert_close(native, explicit_native, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pu_image_gain_image, legacy_image_only, rtol=2e-5, atol=2e-6)
    assert (native - pu_query_gain_image).abs().max() > 1e-6
    assert (native - pu_image_gain_query).abs().max() > 1e-6
    assert (native - pu_image_gain_image).abs().max() > 1e-6


def test_query_order_and_chunk_size_are_equivariant():
    decoder = QLiftDecoder(8, config("query_lift", query_chunk=3)).eval()
    chunked = QLiftDecoder(8, config("query_lift", query_chunk=1)).eval()
    chunked.load_state_dict(decoder.state_dict())
    cost, x, y, text = inputs(queries=3)
    order = torch.tensor((2, 0, 1))
    with torch.no_grad():
        output, _ = decoder(cost, x, y, text)
        reordered, _ = decoder(cost[:, :, order], x, y, text[order])
        split, _ = chunked(cost, x, y, text)
    torch.testing.assert_close(reordered, output[:, :, order], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(split, output, rtol=2e-5, atol=2e-6)


def test_active_query_lift_parameters_receive_finite_gradients_after_opening_update():
    decoder = QLiftDecoder(8, config("query_lift")).train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    cost, x, y, text = inputs()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output, _ = decoder(cost, x, y, text)
        (output.square().mean()).backward()
        optimizer.step()
    active = {name: parameter.grad for name, parameter in decoder.named_parameters() if parameter.requires_grad}
    assert active
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in active.values())
    assert sum(float(gradient.abs().sum()) for gradient in active.values()) > 0
