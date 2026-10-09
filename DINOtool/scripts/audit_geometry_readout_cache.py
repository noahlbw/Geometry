#!/usr/bin/env python3
"""Fixed-pair CPU attribution audit of saved, unchanged Geometry observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dinotool.geometry_readout_trace import class_attribution


PAIRS = {
    "vdd": (("vehicle", "road"), ("vehicle", "other"), ("vehicle", "roof"),
            ("vehicle", "wall"), ("roof", "wall"), ("water", "other")),
    "potsdam": (("car", "impervious surface"), ("car", "low vegetation"),
                ("car", "building"), ("car", "tree")),
}


def accumulate(entry, selected, margins, contributions):
    count = int(selected.sum())
    if not count:
        return
    entry["patch_centers"] = entry.get("patch_centers", 0) + count
    for field, values in (("stage_margins", margins), ("components", contributions)):
        sums = entry.setdefault(field, {})
        for name, value in values.items():
            sums[name] = sums.get(name, 0.) + float(value[selected].double().sum())
    entry["positive_final_margin"] = entry.get("positive_final_margin", 0) + int(
        (margins["final"][selected] > 0).sum())


@torch.inference_mode()
def audit(root):
    results = json.loads((root / "results.json").read_text())
    signature = results["signature"]
    if (results["status"] != "complete" or results["processed_images"] != 8
            or results["total_images"] != 8 or results["windows"] != results["unchanged_windows"]
            or results["maximum_replay_feature_error"] != 0.):
        raise ValueError("Requires the complete, unchanged eight-image diagnostic.")
    expected = {(record["sample_key"], top, left) for record in results["images"]
                for top, left in record["saved_windows"]}
    if len(expected) != 16:
        raise ValueError("Requires exactly two distinct fixed snapshots per image.")
    text_bank = torch.load(root / "text_bank.pt", map_location="cpu", weights_only=True)
    parents = text_bank["parents"]
    names = signature["class_names"]
    if not torch.equal(torch.bincount(parents), torch.full((len(names),), 20)):
        raise ValueError("The fixed 20-alias vocabulary changed.")
    pairs = {f"{a} - {b}": {"a": a, "b": b, "groups": {}}
             for a, b in PAIRS[signature["dataset"]]}
    seen, maximum_error = set(), 0.
    for path in sorted((root / "features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = (cache["sample_key"], cache["top"], cache["left"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate snapshot coordinates.")
        seen.add(key)
        stages = cache["stage_scores"]
        if not torch.equal(stages["final"], stages["Geometry_original"]):
            raise ValueError("Saved final scores differ from original Geometry.")
        components = class_attribution(cache["original_alias_scores"],
                                       cache["alias_contributions"], parents, len(names))
        error = float((sum(components.values()) - stages["final"]).abs().max())
        maximum_error = max(maximum_error, error)
        if error > 2e-5:
            raise ValueError("Conditional components fail to reconstruct saved scores.")
        truth, valid = cache["target_patch_centers"], cache["valid_patch_centers"]
        prediction = stages["final"].argmax(-1)
        for pair in pairs.values():
            a, b = names.index(pair["a"]), names.index(pair["b"])
            margins = {name: value[..., a] - value[..., b] for name, value in stages.items()}
            contributions = {name: value[..., a] - value[..., b]
                             for name, value in components.items()}
            contributions["mlp_plus_norm_bias"] = contributions["mlp"] + contributions["norm_bias"]
            for side, actual in (("truth_a", a), ("truth_b", b)):
                selected = valid & (truth == actual)
                groups = {side: selected,
                          side + "_pred_a": selected & (prediction == a),
                          side + "_pred_b": selected & (prediction == b),
                          side + "_pred_other": selected & (prediction != a) & (prediction != b)}
                for group, mask in groups.items():
                    entry = pair["groups"].setdefault(group, {})
                    accumulate(entry, mask, margins, contributions)
                    image = entry.setdefault("per_image", {}).setdefault(key[0], {})
                    accumulate(image, mask, margins, contributions)
        del cache, components
    if seen != expected:
        raise ValueError("Saved snapshot coverage is incomplete.")

    def finish(entry):
        count = entry.get("patch_centers", 0)
        if not count:
            return {"patch_centers": 0}
        result = {"patch_centers": count,
                  "positive_final_margin_percent": 100 * entry["positive_final_margin"] / count}
        for field in ("stage_margins", "components"):
            result["mean_" + field] = {name: value / count for name, value in entry[field].items()}
        if "per_image" in entry:
            result["per_image"] = {name: finish(image) for name, image in entry["per_image"].items()}
        return result

    for pair in pairs.values():
        pair["groups"] = {name: finish(entry) for name, entry in pair["groups"].items()}
    return {"dataset": signature["dataset"], "implementation": signature["implementation"],
            "seed": signature["seed"], "samples": signature["samples"],
            "vocabulary_sha256": signature["vocabulary_sha256"],
            "cache_count": len(seen), "snapshot_coordinates_verified": True,
            "maximum_class_reconstruction_error": maximum_error,
            "note": "CPU audit only. Margins always A minus B. Conditional components use observed "
                    "normalization and alias responsibilities; not causal removal effects. Counts are "
                    "overlapping patch observation centers in 16 fixed snapshots, not unique pixels "
                    "or all windows. No parameters or thresholds fitted.",
            "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing an existing audit output.")
    torch.set_num_threads(2)
    result = audit(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("dataset", "cache_count",
                                                  "maximum_class_reconstruction_error")}))


if __name__ == "__main__":
    main()
