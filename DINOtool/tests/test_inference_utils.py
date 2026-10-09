import numpy as np
import torch

from dinotool.inference import (
    ProbabilityAccumulator,
    global_anchor_weight,
    hann_blend_window,
    tile_starts,
)


def test_tile_starts_cover_last_pixel() -> None:
    assert tile_starts(1000, 512, 128) == [0, 384, 488]
    assert tile_starts(200, 512, 128) == [0]


def test_hann_window_has_nonzero_edges() -> None:
    window = hann_blend_window(32)
    assert window.shape == (32, 32)
    assert window.min() > 0
    assert window[16, 16] > window[0, 0]


def test_probability_accumulator_blends_and_thresholds(tmp_path) -> None:
    with ProbabilityAccumulator(2, 3, 4, 64, tmp_path) as accumulator:
        probabilities = np.zeros((2, 3, 4), dtype=np.float32)
        probabilities[1] = 0.8
        probabilities[0] = 0.2
        accumulator.add(probabilities, np.ones((3, 4), dtype=np.float32), 0, 0)
        labels, confidence = accumulator.finalize(0.9)
    assert np.all(labels == 255)
    assert np.all(confidence == np.uint8(204))


def test_probability_accumulator_reassigns_low_confidence_to_background(tmp_path) -> None:
    with ProbabilityAccumulator(2, 1, 2, 64, tmp_path) as accumulator:
        probabilities = np.array([[[0.4, 0.1]], [[0.6, 0.9]]], dtype=np.float32)
        accumulator.add(probabilities, np.ones((1, 2), dtype=np.float32), 0, 0)
        plain, _ = accumulator.finalize(None)
        gated, _ = accumulator.finalize(0.7, background_index=0)
    assert plain.tolist() == [[1, 1]]
    assert gated.tolist() == [[0, 1]]


def test_global_anchor_weight_is_positive() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    weight = global_anchor_weight(anchor, anchor, 0, 0, 64, 64, 128, 128)
    assert 0 < weight <= 1

