"""Label-aware error accounting for evaluation only, never model inputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def error_counts(prediction: np.ndarray, target: np.ndarray, classes: int, radii: tuple[int, ...] = (8, 16, 32)) -> dict:
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Expected matching 2D label images")
    valid = (target >= 0) & (target < classes)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= classes)):
        raise ValueError("Prediction label outside vocabulary")
    confusion = np.bincount((target[valid] * classes + prediction[valid]).astype(np.int64), minlength=classes**2).reshape(classes, classes)
    entries = []
    for category in range(classes):
        truth = valid & (target == category)
        predicted = valid & (prediction == category)
        false_positive, false_negative = predicted & ~truth, truth & ~predicted
        other = valid & ~truth
        to_truth = distance_transform_edt(~truth) if truth.any() else np.full(target.shape, np.inf)
        to_other = distance_transform_edt(~other) if other.any() else np.full(target.shape, np.inf)
        entries.append({
            "tp": int((truth & predicted).sum()), "fp": int(false_positive.sum()), "fn": int(false_negative.sum()),
            "fp_within_gt_distance": [int((false_positive & (to_truth <= radius)).sum()) for radius in radii],
            "fn_within_other_gt_distance": [int((false_negative & (to_other <= radius)).sum()) for radius in radii],
            "fp_without_class_in_image": int(false_positive.sum()) if not truth.any() else 0,
        })
    return {"confusion_matrix": confusion.tolist(), "classes": entries, "radii_pixels": list(radii), "valid_pixels": int(valid.sum())}


def summarize_errors(records: list[dict], names: list[str], radii: tuple[int, ...] = (8, 16, 32)) -> dict:
    matrix = sum((np.asarray(record["confusion_matrix"], dtype=np.int64) for record in records), start=np.zeros((len(names), len(names)), dtype=np.int64))
    output = []
    for category, name in enumerate(names):
        rows = [record["classes"][category] for record in records]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        near = [sum(row["fp_within_gt_distance"][i] for row in rows) for i in range(len(radii))]
        output.append({"name": name, "tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                       "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                       "fp_within_gt_distance": near, "fp_beyond_gt_distance": [fp - value for value in near],
                       "fn_within_other_gt_distance": [sum(row["fn_within_other_gt_distance"][i] for row in rows) for i in range(len(radii))],
                       "fp_without_class_in_image": sum(row["fp_without_class_in_image"] for row in rows),
                       "fp_source_classes": {other: int(matrix[i, category]) for i, other in enumerate(names) if i != category}})
    ious = [row["iou"] for row in output if row["iou"] is not None]
    return {"images": len(records), "mean_iou_percent": 100 * float(np.mean(ious)), "per_class": output,
            "confusion_matrix": matrix.tolist(), "radii_pixels": list(radii),
            "distance_definition": "FP: nearest valid GT pixel of this class; FN: nearest valid GT pixel of another class. Ignored pixels are neither targets nor errors. Distances are native pixels, not meters or a causal boundary-error diagnosis."}
