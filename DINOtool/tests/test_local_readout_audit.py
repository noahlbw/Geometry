import numpy as np
import torch

from dinotool.local_readout_audit import ARMS, error_counts, summarize_error_counts
from dinotool.tcpr import geometry_relation


def test_sparse_relation_keeps_self_and_removes_weak_edges():
    raw = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
    logits = torch.zeros((1, 3, 3))
    dense = geometry_relation(logits, raw, "dense")
    sparse = geometry_relation(logits, raw, "sparse")
    assert torch.allclose(dense, torch.full_like(dense, 1 / 3))
    assert torch.allclose(sparse.sum(-1), torch.ones((1, 3)))
    assert torch.all(torch.diagonal(sparse, dim1=-2, dim2=-1) > 0)
    assert sparse[0, 0, 2] == 0
    assert sparse[0, 2, 0] == 0


def test_intersections_count_common_errors_and_exclusive_rescues():
    target = np.array([[0, 1, 0, 1, 255]])
    predictions = {
        "C_dense": np.array([[0, 0, 1, 0, 0]]),
        "A_dense": np.array([[0, 1, 1, 0, 0]]),
        "C_sparse": np.array([[0, 0, 0, 0, 0]]),
        "A_sparse": np.array([[0, 0, 1, 0, 0]]),
    }
    counts = error_counts(predictions, target, 2)
    summary = summarize_error_counts(counts, ["first", "second"])
    assert tuple(predictions) == ARMS
    assert summary["valid_pixels"] == 4
    assert summary["error_patterns"][0] == 1
    assert summary["error_patterns"][13] == 1  # Only A_dense correct.
    assert summary["error_patterns"][11] == 1  # Only C_sparse correct.
    assert summary["error_patterns"][15] == 1
    assert summary["oracle_any_correct_pixel_accuracy_percent"] == 75
    assert summary["exclusive_correct_pixels"]["A_dense"] == 1
    assert summary["exclusive_correct_pixels"]["C_sparse"] == 1
    assert summary["pairwise"]["C_dense__A_dense"]["first_wrong_second_correct"] == 1
    assert summary["pairwise"]["C_dense__C_sparse"]["first_wrong_second_correct"] == 1
