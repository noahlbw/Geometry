"""Fixed sign-removal counterfactuals; these are diagnostics, not a final model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geometry_localized_likelihood import PRIMARY, IMPLEMENTATION as SOURCE_IMPLEMENTATION
from eval_cached_localized_likelihood import audit, save_json


IMPLEMENTATION = "geometry-localized-likelihood-sign-audit-20261002"
METHODS = ("Geometry", PRIMARY, "PositiveOnly_Geometry", "NegativeOnly_Geometry",
           "PositiveOnly_Box", "NegativeOnly_Box")


def signed_scores(base, candidate):
    delta = candidate-base
    return base+delta.clamp_min(0), base+delta.clamp_max(0)


def predict(raw, classes):
    output = {}
    for method in METHODS:
        scores = torch.from_numpy(raw[method])
        dense = F.interpolate(scores.T.reshape(1, classes, 32, 32), (512, 512),
                              mode="bilinear", align_corners=False)[0]
        output[method] = dense.argmax(0).numpy()
    return output


@torch.inference_mode()
def main(args):
    if args.output_dir.exists():
        raise ValueError("Refusing existing sign-audit output.")
    source = json.loads((args.source/"verified.json").read_text())
    old = source["signature"]
    if (source["implementation"] != SOURCE_IMPLEMENTATION or source["status"] != "complete"
            or not source["coverage_verified"] or not source["original_geometry_exact"]
            or old["candidate_scores_loaded_masks"] or old["dataset"] != args.dataset
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() !=
                old["source_signature"]["vocabulary_sha256"]):
        raise ValueError("Verified frozen source score cache required.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "sample_keys": old["sample_keys"], "source_signature": old, "source_results": source,
        "candidate_scores_loaded_masks": False,
        "note": "Fixed sign-removal counterfactual audit, not a selected new model. "
                "Zero is the neutral score increment, not a fitted threshold. No parameter search."}
    manifests = json.loads((args.source/"window_manifest.json").read_text())
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"scores").mkdir()
    save_json(args.output_dir/"signature.json", {key: value for key, value in signature.items() if key != "source_results"})
    for number, row in enumerate(manifests):
        with np.load(args.source/row["scores_file"], allow_pickle=False) as raw:
            base = torch.from_numpy(raw["Geometry"]).to(args.device)
            candidate = torch.from_numpy(raw[PRIMARY]).to(args.device)
            box = torch.from_numpy(raw["BoxLocalLikelihood"]).to(args.device)
        positive, negative = signed_scores(base, candidate)
        box_positive, box_negative = signed_scores(base, box)
        values = {"Geometry": base, PRIMARY: candidate, "PositiveOnly_Geometry": positive,
                  "NegativeOnly_Geometry": negative, "PositiveOnly_Box": box_positive, "NegativeOnly_Box": box_negative}
        row["scores_file"] = f"scores/w{number:03d}.npz"
        np.savez_compressed(args.output_dir/row["scores_file"], **{method: value.cpu().numpy() for method, value in values.items()})
    save_json(args.output_dir/"window_manifest.json", manifests)
    save_json(args.output_dir/"collection_status.json", {"status": "collected", "target_masks_loaded": False,
        "windows": len(manifests), "parameter_search": False, "source_model_rerun": False})
    audit(args, signature, manifests, args.output_dir, methods=METHODS, predictor=predict, implementation=IMPLEMENTATION)
    result = json.loads((args.output_dir/"results.json").read_text())
    for key, group in result["metrics"].items():
        for method in ("Geometry", PRIMARY):
            if not np.array_equal(group[method]["confusion_matrix"], source["metrics"][key][method]["confusion_matrix"]):
                raise RuntimeError("Sign audit changed an authoritative source prediction.")
    result["source_predictions_exact"] = True
    result["promotion_or_model_selection"] = False
    save_json(args.output_dir/"results.json", result)
    print(json.dumps({"dataset": args.dataset, "source_predictions_exact": True,
        "window_miou": {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                        for key, group in result["metrics"].items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    for field in ("data-root", "vocabulary-config"):
        parser.add_argument("--"+field, required=True)
    for field in ("source", "output-dir"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
