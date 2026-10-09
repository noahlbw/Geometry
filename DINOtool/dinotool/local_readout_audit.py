"""Exact pixel-level error intersections for matched local readouts."""
from __future__ import annotations

from itertools import combinations

import numpy as np


ARMS = ("C_dense", "A_dense", "C_sparse", "A_sparse")


def error_counts(predictions: dict[str, np.ndarray], target: np.ndarray,
                 class_count: int) -> dict[str, object]:
    valid = (target >= 0) & (target < class_count)
    if any(name not in predictions or predictions[name].shape != target.shape for name in ARMS):
        raise ValueError("All four predictions must match the target shape.")
    truth = target[valid]
    labels = {name: predictions[name][valid] for name in ARMS}
    incorrect = {name: labels[name] != truth for name in ARMS}
    pattern = sum(incorrect[name].astype(np.uint8) << index
                  for index, name in enumerate(ARMS))
    per_class = np.zeros((class_count, 16), dtype=np.int64)
    np.add.at(per_class, (truth, pattern), 1)
    pairs = {}
    for first, second in combinations(ARMS, 2):
        changed = labels[first] != labels[second]
        a, b = incorrect[first], incorrect[second]
        pairs[f"{first}__{second}"] = {
            "changed": int(changed.sum()),
            "first_wrong_second_correct": int((a & ~b).sum()),
            "first_correct_second_wrong": int((~a & b).sum()),
            "different_both_wrong": int((changed & a & b).sum()),
            "same_wrong": int((~changed & a).sum()),
        }
    return {
        "valid_pixels": int(valid.sum()),
        "error_patterns": np.bincount(pattern, minlength=16).tolist(),
        "error_patterns_by_true_class": per_class.tolist(),
        "pairwise": pairs,
    }


def summarize_error_counts(counts: dict[str, object], names: list[str]) -> dict[str, object]:
    patterns = counts["error_patterns"]
    valid = counts["valid_pixels"]
    if sum(patterns) != valid:
        raise ValueError("Error patterns do not cover valid pixels.")
    per_class = counts["error_patterns_by_true_class"]
    if len(per_class) != len(names) or any(len(row) != 16 for row in per_class):
        raise ValueError("Incorrect per-class error pattern shape.")
    if np.asarray(per_class, dtype=np.int64).sum(axis=0).tolist() != patterns:
        raise ValueError("Per-class intersections do not sum to global intersections.")
    return {
        **counts,
        "all_four_wrong_fraction": patterns[15] / max(valid, 1),
        "oracle_any_correct_pixel_accuracy_percent": 100 * (1 - patterns[15] / max(valid, 1)),
        "exclusive_correct_pixels": {
            name: patterns[15 ^ (1 << index)] for index, name in enumerate(ARMS)
        },
        "per_class": [
            {"name": name, "pixels": int(sum(row)), "all_four_wrong": row[15],
             "any_correct_fraction": 1 - row[15] / max(sum(row), 1)}
            for name, row in zip(names, per_class)
        ],
    }
