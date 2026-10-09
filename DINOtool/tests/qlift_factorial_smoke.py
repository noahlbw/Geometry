"""Pytest-free CPU smoke test for the Q-Lift P/U × update-gain intervention."""
from __future__ import annotations

import torch

from dinotool.qlift import QLiftConfig, QLiftDecoder


def main() -> None:
    torch.manual_seed(20260922)
    torch.set_num_threads(2)
    config = QLiftConfig(arm="query_lift", levels=2, guide_dim=32, hidden_dim=64, kernel=3, query_chunk=3)
    decoder = QLiftDecoder(8, config).eval()
    for level in decoder.levels:
        level.context.output.weight.data.normal_(std=0.02)
        level.detail.output.weight.data.normal_(std=0.02)
    cost = torch.randn(1, 8, 3, 7, 11)
    x, y, text = torch.randn(1, 1024, 7, 11), torch.randn(1, 1024, 7, 11), torch.randn(3, 1024)
    with torch.no_grad():
        native, _ = decoder(cost, x, y, text)
        explicit_native, _ = decoder(cost, x, y, text, lifting_query_conditioned=True, gain_query_conditioned=True)
        query_image, _ = decoder(cost, x, y, text, lifting_query_conditioned=True, gain_query_conditioned=False)
        image_query, _ = decoder(cost, x, y, text, lifting_query_conditioned=False, gain_query_conditioned=True)
        image_image, _ = decoder(cost, x, y, text, lifting_query_conditioned=False, gain_query_conditioned=False)
        legacy_image, _ = decoder(cost, x, y, text, lifting_query_conditioned=False)
    torch.testing.assert_close(native, explicit_native, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(image_image, legacy_image, rtol=2e-5, atol=2e-6)
    assert (native - query_image).abs().max() > 1e-6
    assert (native - image_query).abs().max() > 1e-6
    assert (native - image_image).abs().max() > 1e-6
    print("qlift-factorial-cpu-smoke-ok")


if __name__ == "__main__":
    main()
