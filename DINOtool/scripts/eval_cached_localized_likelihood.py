"""Freeze image-only likelihood readouts from raw caches, then audit window masks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geometry_localized_likelihood import (
    IMPLEMENTATION, PRIMARY, METHODS, LocalLikelihoodConfig,
    class_localization_density, apply_localized_likelihood)
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_gear_ov import protocol


def save_json(path, row):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(row, indent=2)+"\n")
    temp.replace(path)


def dense_predictions(raw, classes):
    probabilities, predictions = {}, {}
    for method in METHODS:
        if method == "MeanProb_BoxLikelihood":
            continue
        value = torch.from_numpy(raw[method])
        dense = F.interpolate(value.T.reshape(1, classes, 32, 32), (512, 512),
                              mode="bilinear", align_corners=False)[0]
        probabilities[method] = (dense/.07).softmax(0)
        predictions[method] = dense.argmax(0).numpy()
    probabilities["MeanProb_BoxLikelihood"] = .5*(probabilities["Geometry"]+probabilities["BoxLocalLikelihood"])
    predictions["MeanProb_BoxLikelihood"] = probabilities["MeanProb_BoxLikelihood"].argmax(0).numpy()
    return predictions


def audit(args, signature, manifests, output, *, methods=METHODS, predictor=dense_predictions,
          implementation=IMPLEMENTATION, primary=PRIMARY, note=None):
    from eval_geometry_vip_reliability import summary
    vocabulary = load_class_specs(args.vocabulary_config)
    samples, specs, _, load_mask = protocol(args, vocabulary)
    lookup = {sample.key: sample for sample in samples}
    confusion = {key: {method: np.zeros((len(classes),)*2, np.int64) for method in methods}
                 for key, classes in specs.items()}
    transitions = {key: {method: np.zeros((len(classes),)*3, np.int64) for method in methods if method != "Geometry"}
                   for key, classes in specs.items()}
    current_key, target = None, None
    for row in manifests:
        sample = lookup[row["sample_key"]]
        key, classes = row["protocol"], len(specs[row["protocol"]])
        if current_key != (sample.key, key):
            target = load_mask(sample, key, tuple(row["image_shape"]))
            current_key = sample.key, key
        top, left = row["top"], row["left"]
        ah, aw = min(512, target.shape[0]-top), min(512, target.shape[1]-left)
        crop = np.full((512, 512), -1, np.int64)
        crop[:ah, :aw] = target[top:top+ah, left:left+aw]
        valid = (crop >= 0) & (crop < classes)
        with np.load(output/row["scores_file"], allow_pickle=False) as raw:
            predictions = predictor(raw, classes)
        for method, prediction in predictions.items():
            encoded = crop[valid]*classes+prediction[valid]
            confusion[key][method] += np.bincount(encoded, minlength=classes**2).reshape(classes, classes)
            if method != "Geometry":
                transitions[key][method] += transition_counts(predictions["Geometry"], prediction, crop, classes)
    metrics = {key: {method: summary(cm, tuple(spec.name for spec in specs[key]), 0) for method, cm in group.items()}
               for key, group in confusion.items()}
    changes = {key: {method: {"counts": counts.tolist(), **transition_summary(counts)} for method, counts in group.items()}
               for key, group in transitions.items()}
    for key, group in metrics.items():
        previous = signature["source_results"]["metrics"][key]["Geometry"]["confusion_matrix"]
        if not np.array_equal(previous, group["Geometry"]["confusion_matrix"]):
            raise RuntimeError("Cached original Geometry window confusion changed.")
        for method, metric in group.items():
            cm = np.asarray(metric["confusion_matrix"], np.int64)
            if method != "Geometry":
                if (not np.array_equal(changes[key][method]["base_confusion"], previous)
                        or not np.array_equal(changes[key][method]["proposal_confusion"], cm)):
                    raise RuntimeError("Transition reconstruction failed.")
            for c, entry in enumerate(metric["per_class"]):
                entry.update(precision_percent=100*int(cm[c, c])/max(int(cm[:, c].sum()), 1),
                             recall_percent=100*int(cm[c, c])/max(int(cm[c].sum()), 1))
    signature = {key: value for key, value in signature.items() if key != "source_results"}
    result = {"status": "complete", "implementation": implementation, "primary": primary,
        "signature": signature, "processed_images": len(signature["sample_keys"]),
        "total_images": len(signature["sample_keys"]), "coverage_verified": True,
        "window_count": len(manifests), "original_geometry_exact": True,
        "metrics": metrics, "transitions": changes,
        "note": note or "Window-only development audit, not full-image dataset metrics. "
            "Revised likelihood hypothesis motivated by prior labeled audit; not untouched validation."}
    save_json(output/"results.json", result)
    print(json.dumps({"dataset": args.dataset, "window_miou": {key: {method: metric["mean_iou_percent"]
        for method, metric in group.items()} for key, group in metrics.items()}}), flush=True)


@torch.inference_mode()
def main(args):
    output = args.output_dir
    if output.exists():
        raise ValueError("Refusing existing cached-readout output.")
    source = json.loads((args.source/"verified.json").read_text())
    old = source["signature"]
    if (not source["coverage_verified"] or source["status"] != "complete" or not old["weights_frozen"]
            or old["target_masks_loaded_during_observation"] or old["dataset"] != args.dataset
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != old["vocabulary_sha256"]):
        raise ValueError("Verified original image-only observation cache required.")
    keys = old["sample_keys"]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source images.")
    manifests = json.loads((args.source/"raw_manifest.json").read_text())
    if set(row["sample_key"] for row in manifests) != set(keys):
        raise ValueError("Incomplete cache coverage.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "sample_keys": keys,
        "methods": METHODS, "config": LocalLikelihoodConfig().signature(), "source_signature": old,
        "candidate_scores_loaded_masks": False, "source_results": source}
    output.mkdir(parents=True)
    (output/"scores").mkdir()
    save_json(output/"signature.json", {key: value for key, value in signature.items() if key != "source_results"})
    started = time.perf_counter()
    diag = {key: {} for key in old["vocabularies"]}
    torch.cuda.reset_peak_memory_stats()
    for number, row in enumerate(manifests):
        key = row["protocol"]
        bank = old["vocabularies"][key]
        classes = len(bank["classes"])
        parents = np.repeat(np.arange(classes), bank["counts"]).tolist()
        if any(count != 20 for count in bank["counts"]):
            raise ValueError("All20 aliases must remain.")
        with np.load(args.source/row["file"], allow_pickle=False) as raw:
            geometry_scores = torch.from_numpy(raw["geometry_logits"]).to(args.device)
            relation = torch.from_numpy(raw["geometry_relation"]).to(args.device)
            observations = [{name: raw[f"{name}_{chunk}"] for name in ("boxes", "alias_scores", "alias_indices")}
                            for chunk in range(int(raw["chunks"]))]
        height, width = row["image_shape"]
        centers = (torch.arange(32, device=args.device)+.5)*16
        valid = ((centers[:, None] < min(512, height-row["top"])) &
                 (centers[None] < min(512, width-row["left"]))).reshape(-1)
        permutation = torch.randperm(int(valid.sum()), generator=torch.Generator().manual_seed(20261002+number)).to(args.device)
        indices = valid.nonzero().flatten()
        shuffled = relation.clone()
        shuffled[indices[:, None], indices[None]] = relation[indices[permutation]][:, indices[permutation]]
        scores = {"Geometry": geometry_scores}
        for method, graph in (("BoxLocalLikelihood", None), ("ShuffledLocalLikelihood", shuffled), (PRIMARY, relation)):
            density, diagnostic = class_localization_density(observations, parents, classes, valid, graph)
            scores[method] = apply_localized_likelihood(geometry_scores, density)
            for field, value in diagnostic.items():
                slot = diag[key].setdefault(method, {})
                slot[field] = slot.get(field, 0)+value
        identity, _ = class_localization_density(observations, parents, classes, valid,
                                                 torch.eye(len(valid), device=args.device))
        control, _ = class_localization_density(observations, parents, classes, valid)
        if not torch.equal(identity, control):
            raise RuntimeError("Actual-cache identity failed to recover same-source control.")
        row["scores_file"] = f"scores/w{number:03d}.npz"
        np.savez_compressed(output/row["scores_file"], **{method: value.cpu().numpy() for method, value in scores.items()})
        save_json(output/"collection_status.json", {"status": "collected" if number+1 == len(manifests) else "collecting",
            "processed_windows": number+1, "total_windows": len(manifests), "target_masks_loaded": False,
            "identity_control_exact": True, "diagnostics_sum": diag,
            "wall_seconds": time.perf_counter()-started, "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576})
    save_json(output/"window_manifest.json", manifests)
    audit(args, signature, manifests, output)


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
