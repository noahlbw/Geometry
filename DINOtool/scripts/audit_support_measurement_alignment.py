#!/usr/bin/env python3
"""Audit saved support measurements; labels never select or change a readout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geometry_support_erasure import semantic_score_influence, support_operator
from dinotool.support_conditioned_geometry import SupportConfig


@torch.inference_mode()
def main(args):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise ValueError("Refusing existing alignment audit.")
    original = json.loads((source/"results.json").read_text())
    measured = json.loads(Path(args.observations).read_text())
    if measured["status"] != "complete" or measured["snapshots"] != 16 or measured["maximum_control_error"] != 0.:
        raise ValueError("Requires the complete exact-replay intervention cache.")
    images = {row["sample_key"]: row for row in original["images"]}
    lookup = {(row["sample_key"], row["top"], row["left"]): row for row in measured["records"]}
    names = original["signature"]["class_names"]
    config, records, seen = SupportConfig(), [], set()
    maximum_reference_error = 0.
    for path in sorted((source/"features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = cache["sample_key"], cache["top"], cache["left"]
        if key not in lookup or key in seen:
            raise ValueError("Unexpected or duplicate support cache.")
        seen.add(key)
        image = images[key[0]]
        yy, xx = key[1]+torch.arange(32)*16+8, key[2]+torch.arange(32)*16+8
        valid = (yy[:, None] < image["height"]) & (xx[None] < image["width"])
        raw = F.normalize(cache["features"]["backbone_tokens"][:, cache["prefix_tokens"]:].float(), dim=-1)[0]
        local = cache["stage_scores"]["Geometry_original"][0]
        operator = support_operator(raw, cache["geometry_relation"][0], local, valid, config)
        diagnostics = lookup[key]["diagnostics"]
        reference = torch.tensor(diagnostics["support_reference"])
        error = float((operator @ local-reference).abs().max()) if len(operator) else 0.
        maximum_reference_error = max(maximum_reference_error, error)
        if error > 1e-6 or len(operator) != diagnostics["regions"]:
            raise ValueError("The original image-only support operator changed.")
        observations = diagnostics["semantic_observations"]
        gain = torch.tensor(observations["SupportGlobal"])-torch.tensor(observations["SupportErasedGlobal"])
        area = operator.sum(-1)/operator.amax(-1).clamp_min(1e-12)/valid.sum()
        corrected = semantic_score_influence(local, operator, gain, valid.reshape(-1))
        # Targets enter only after the frozen operator, measurement and readout.
        truth = cache["target_patch_centers"][0]
        scored = valid.reshape(-1) & (truth >= 0) & (truth < len(names))
        one_hot = F.one_hot(truth.clamp(0, len(names)-1).long(), len(names)).float()*scored[:, None]
        mass = operator @ one_hot
        mass = mass/mass.sum(-1, keepdim=True).clamp_min(1e-12)
        support_corrected = operator @ corrected
        for index in range(len(operator)):
            records.append({"sample_key": key[0], "top": key[1], "left": key[2], "region": index,
                "target_distribution": mass[index].tolist(), "target_purity": float(mass[index].max()),
                "target_dominant": names[int(mass[index].argmax())],
                "Geometry_dominant": names[int(reference[index].argmax())],
                "native_gain_dominant": names[int(gain[index].argmax())],
                "ScoreInfluence_dominant": names[int(support_corrected[index].argmax())],
                "soft_erasure_fraction": float(area[index]),
                "native_gain": gain[index].tolist(),
                "local_reference": reference[index].tolist(),
                "corrected_reference": support_corrected[index].tolist()})
        del cache
    if seen != set(lookup) or len(seen) != 16:
        raise ValueError("Incomplete diagnostic-window coverage.")
    alignments = {}
    for method in ("Geometry", "native_gain", "ScoreInfluence"):
        confusion = np.zeros((len(names), len(names)), np.int64)
        expected_mass = []
        for row in records:
            selected = names.index(row[method+"_dominant"])
            confusion[names.index(row["target_dominant"]), selected] += 1
            expected_mass.append(row["target_distribution"][selected])
        alignments[method] = {"dominant_matches": int(np.trace(confusion)),
                              "total_supports": len(records),
                              "mean_target_mass_of_selected_class": float(np.mean(expected_mass)),
                              "dominant_confusion": confusion.tolist()}
    result = {"status": "complete", "dataset": measured["dataset"], "windows": len(seen),
              "supports": len(records), "maximum_support_reference_error": maximum_reference_error,
              "classes": names, "alignment": alignments,
              "mean_target_purity": float(np.mean([row["target_purity"] for row in records])),
              "records": records,
              "note": "Audit of existing measurements only, no new model forwards or readout changes. Geometry support and scores are reconstructed before target access. Dominant gain is an influence direction, not a calibrated posterior. Overlapping fixed diagnostic windows, not full-image results; no fitted gate or selection."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
