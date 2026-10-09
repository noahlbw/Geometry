#!/usr/bin/env python3
"""Audit fixed Geometry support signals on cached observations; no prediction writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rank_auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], np.sort(scores[~labels])
    if not len(positive) or not len(negative):
        return None, 0
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    pairs = len(positive) * len(negative)
    return float((left + .5 * (right - left)).sum() / pairs), pairs


def support_signals(relation, margin):
    relation, margin = np.asarray(relation, dtype=np.float64), np.asarray(margin, dtype=np.float64)
    if relation.shape != (len(margin), len(margin)) or len(margin) < 2:
        raise ValueError("Requires a square patch relation and matching margins.")
    if not np.isfinite(relation).all() or not np.isfinite(margin).all() or (relation < 0).any():
        raise ValueError("Nonfinite or negative cached relation.")
    weights = relation.copy()
    np.fill_diagonal(weights, 0.)
    mass = weights.sum(-1)
    if (mass <= 0).any():
        raise ValueError("No nonself support for a patch.")
    support = (weights / mass[:, None]) @ margin
    control = (margin.sum() - margin) / (len(margin) - 1)
    return {"final_margin": margin, "support_margin": support,
            "support_minus_window_control": support - control,
            "query_minus_support": margin - support}


def summary(scores, labels, images):
    value, pairs = rank_auc(scores, labels)
    wins, within_pairs, per_image = 0., 0, {}
    for key in sorted(set(images)):
        mask = images == key
        auc, count = rank_auc(scores[mask], labels[mask])
        per_image[key] = {"positive": int(labels[mask].sum()),
                          "negative": int((~labels[mask]).sum()), "auc": auc}
        if count:
            wins += auc * count
            within_pairs += count

    def distribution(values):
        if not len(values):
            return None
        return {"mean": float(values.mean()),
                "positive_percent": 100 * float((values > 0).mean()),
                "zero_percent": 100 * float((values == 0).mean()),
                "q25_q50_q75": np.quantile(values, [.25, .5, .75]).tolist()}

    return {"auc": value, "comparable_pairs": pairs,
            "within_image_pair_weighted_auc": wins / within_pairs if within_pairs else None,
            "within_image_comparable_pairs": within_pairs,
            "positive_scores": distribution(scores[labels]),
            "negative_scores": distribution(scores[~labels]), "per_image": per_image}


def audit(root):
    import torch
    from audit_geometry_readout_cache import PAIRS

    torch.set_num_threads(2)
    result = json.loads((root / "results.json").read_text())
    if (result["status"] != "complete" or result["processed_images"] != 8
            or result["total_images"] != 8):
        raise ValueError("Requires the completed eight-image observation.")
    signature = result["signature"]
    expected = {(row["sample_key"], top, left) for row in result["images"]
                for top, left in row["saved_windows"]}
    seen, rows = set(), {}
    for a, b in PAIRS[signature["dataset"]]:
        rows[(a, b)] = {"scores": {}, "labels": [], "predictions": [], "images": []}
    for path in sorted((root / "features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = (cache["sample_key"], cache["top"], cache["left"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate snapshot.")
        seen.add(key)
        stages = cache["stage_scores"]
        if not torch.equal(stages["final"], stages["Geometry_original"]):
            raise ValueError("Original cached scores changed.")
        score = stages["final"].numpy()[0]
        relation = cache["geometry_relation"].float().numpy()[0]
        truth = cache["target_patch_centers"].numpy()[0]
        valid = cache["valid_patch_centers"].numpy()[0]
        prediction = score.argmax(-1)
        for (a, b), row in rows.items():
            ai, bi = signature["class_names"].index(a), signature["class_names"].index(b)
            fields = support_signals(relation, score[:, ai] - score[:, bi])
            mask = valid & ((truth == ai) | (truth == bi))
            for name, values in fields.items():
                row["scores"].setdefault(name, []).extend(values[mask].tolist())
            row["labels"].extend((truth[mask] == ai).tolist())
            row["predictions"].extend(prediction[mask].tolist())
            row["images"].extend([key[0]] * int(mask.sum()))
        del cache
    if seen != expected or len(seen) != 16:
        raise ValueError("Snapshot coverage incomplete.")
    pairs = {}
    for (a, b), row in rows.items():
        labels, predictions, images = (np.asarray(row[name]) for name in ("labels", "predictions", "images"))
        groups = {}
        for group, mask in (("all_truth_a_or_b", np.ones(len(labels), dtype=bool)),
                            ("original_pred_a", predictions == signature["class_names"].index(a)),
                            ("original_pred_b", predictions == signature["class_names"].index(b))):
            groups[group] = {"positive_truth_a": int(labels[mask].sum()),
                             "negative_truth_b": int((~labels[mask]).sum()),
                             "signals": {name: summary(np.asarray(values)[mask], labels[mask], images[mask])
                                         for name, values in row["scores"].items()}}
        pairs[a + " - " + b] = groups
    return {"dataset": signature["dataset"], "samples": signature["samples"],
            "cache_count": len(seen), "snapshot_coordinates_verified": True,
            "note": "Only cached original Geometry fields. Signal signs are fixed, not label-fitted. "
                    "Nonself Geometry neighbors are compared with uniform nonself window control. "
                    "Labels only audit ranking and natural-zero signs; no threshold selection, fitting, "
                    "model change or prediction output. Query contrast measures relative A evidence, "
                    "not a calibrated A/B probability. "
                    "Overlapping patch centers, not independent samples or full-dataset metrics.",
            "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing an existing audit output.")
    result = audit(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"dataset": result["dataset"], "cache_count": result["cache_count"]}))


if __name__ == "__main__":
    main()
