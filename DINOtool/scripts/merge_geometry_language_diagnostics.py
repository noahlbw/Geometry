"""Verify language observations against untouched original Geometry snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_geometry_language_observation import query_metric
from dinotool.geometry_language_observation import IMPLEMENTATION, METHODS, LanguageObservationConfig, consensus_predictions, probe_indices


def merge(root, source, dataset, device="cpu"):
    output = root/f"{dataset}_merged.json"
    if output.exists():
        raise ValueError("Preserve an already merged diagnostic.")
    reference = json.loads((source/"results.json").read_text())
    rows = [json.loads((root/f"{dataset}_s{shard}"/"results.json").read_text()) for shard in range(2)]
    identity = rows[0]["signature"]
    names = identity["class_names"]
    classes = len(names)
    expected_files = sorted(path.name for path in (source/"features").glob("*.pt"))
    expected_config = json.loads(json.dumps(LanguageObservationConfig().signature()))
    matrices = {method: np.zeros((classes, classes+1), np.int64) for method in METHODS}
    changes = {method: dict(beneficial=0, harmful=0, wrong_to_wrong=0, unchanged=0) for method in METHODS[1:]}
    images = {row["sample_key"]: row for row in reference["images"]}
    records, queries, snapshots = [], [], []
    for shard, row in enumerate(rows):
        signature = row["signature"]
        comparison = {key: value for key, value in signature.items() if key != "shard_index"}
        first = {key: value for key, value in identity.items() if key != "shard_index"}
        if (row["status"] != "complete" or row["processed_windows"] != row["total_windows"]
                or comparison != first or signature["shard_index"] != shard or signature["num_shards"] != 2
                or signature["smoke"] or signature["implementation"] != IMPLEMENTATION
                or signature["source_geometry_signature"] != reference["signature"] or signature["dataset"] != dataset
                or signature["config"] != expected_config
                or signature["global_snapshot_files"] != expected_files):
            raise ValueError("Incomplete or mismatched fixed diagnostic.")
        actual_files = [record["snapshot"] for record in row["records"]]
        if actual_files != expected_files[shard::2]:
            raise ValueError("Shard window sequence differs.")
        local_matrices = {method: np.zeros_like(matrix) for method, matrix in matrices.items()}
        for record in row["records"]:
            name = record["snapshot"]
            cached = torch.load(source/"features"/name, map_location="cpu", weights_only=True)
            observed = torch.load(root/f"{dataset}_s{shard}"/"observations"/name, map_location="cpu", weights_only=True)
            image = images[cached["sample_key"]]
            yy = cached["top"]+torch.arange(32)*16+8
            xx = cached["left"]+torch.arange(32)*16+8
            valid = ((yy[:, None] < image["height"]) & (xx[None] < image["width"])).flatten()
            scores = cached["stage_scores"]["Geometry_original"][0].float().to(device)
            expected = probe_indices(scores, valid.to(device)).cpu()
            if (not torch.equal(expected, observed["indices"]) or expected.tolist() != record["query_indices"]
                    or observed["sample_key"] != cached["sample_key"] or observed["top"] != cached["top"]
                    or observed["left"] != cached["left"] or observed["diagnostics"] != record["diagnostics"]):
                raise ValueError("Probe sampling/coordinates or original image identity differs.")
            logits = observed["option_logits"].to(device)
            if logits.shape != (2, 2, len(expected), classes+1) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Invalid four-way raw language observations.")
            log_probability = logits.log_softmax(-1)
            consensus, accepted = consensus_predictions(scores[expected].argmax(-1), logits.argmax(-1), classes)
            derived = {"Geometry": scores[expected].argmax(-1),
                "LanguageContext": log_probability[0].mean(0).argmax(-1),
                "LanguageFoveal": log_probability[1].mean(0).argmax(-1),
                "LanguageGrounded": log_probability.mean((0, 1)).argmax(-1), "GroundedConsensus": consensus}
            derived = {method: prediction.cpu() for method, prediction in derived.items()}
            accepted = accepted.cpu()
            mismatches = {method: (prediction != observed["predictions"][method]).nonzero().flatten().tolist()
                          for method, prediction in derived.items() if not torch.equal(prediction, observed["predictions"][method])}
            if not torch.equal(accepted, observed["accepted"]) or mismatches:
                raise ValueError(f"Raw scores do not reconstruct saved predictions: {name} {mismatches}")
            indices = expected.numpy()
            truth = cached["target_patch_centers"][0].numpy()[indices]
            evaluated = (truth >= 0) & (truth < classes)
            old = derived["Geometry"].numpy()
            for method, tensor in derived.items():
                prediction = tensor.numpy()
                encoded = truth[evaluated].astype(np.int64)*(classes+1)+prediction[evaluated]
                matrix = np.bincount(encoded, minlength=classes*(classes+1)).reshape(classes, classes+1)
                matrices[method] += matrix
                local_matrices[method] += matrix
                if method != "Geometry":
                    changed = prediction != old
                    beneficial = changed & (prediction == truth) & evaluated
                    harmful = changed & (old == truth) & evaluated
                    changes[method]["beneficial"] += int(beneficial.sum())
                    changes[method]["harmful"] += int(harmful.sum())
                    changes[method]["wrong_to_wrong"] += int((changed & ~beneficial & ~harmful & evaluated).sum())
                    changes[method]["unchanged"] += int((~changed & evaluated).sum())
            snapshots.append(name)
            queries.extend([f"{cached['sample_key']}/{cached['top']}/{cached['left']}/{index}" for index in indices])
            records.append(record)
        for method, matrix in local_matrices.items():
            if not np.array_equal(matrix, row["metrics"][method]["confusion_matrix"]):
                raise ValueError("Saved shard metrics do not reproduce actual query outcomes.")
    if len(queries) != len(set(queries)) or sorted(snapshots) != expected_files or len(expected_files) != 16:
        raise ValueError("Incomplete or duplicate snapshot/query coverage.")
    metrics = {method: query_metric(matrix, names) for method, matrix in matrices.items()}
    primary, geometry = metrics["GroundedConsensus"], metrics["Geometry"]
    indexed = lambda row: {item["name"]: item for item in row["per_class"]}
    p, g = indexed(primary), indexed(geometry)
    small, coverage = ("vehicle", "water") if identity["dataset"] == "vdd" else ("car", "low vegetation")
    checks = {"query_miou_retained": primary["mean_iou_percent"] >= geometry["mean_iou_percent"],
        "beneficial_exceeds_harmful": changes["GroundedConsensus"]["beneficial"] > changes["GroundedConsensus"]["harmful"],
        "small_class_fp_falls": p[small]["fp"] < g[small]["fp"],
        "small_class_tp_retained": p[small]["tp"] >= g[small]["tp"],
        "coverage_class_tp_retained": p[coverage]["tp"] >= g[coverage]["tp"],
        "focus_classes_observed": g[small]["tp"]+g[small]["fn"] > 0 and g[coverage]["tp"]+g[coverage]["fn"] > 0}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "dataset": identity["dataset"],
              "signature": {key: value for key, value in identity.items() if key != "shard_index"},
              "coverage_verified": True, "raw_prediction_reconstruction_verified": True,
              "reconstruction_device": device,
              "original_geometry_query_identity_verified": True, "processed_windows": 16, "total_windows": 16,
              "unique_query_keys": queries, "query_count": len(queries), "records": records,
              "metrics": metrics, "changes": changes, "decision": {"passed": all(checks.values()), "checks": checks},
              "parallel_wall_seconds": max(row["wall_seconds"] for row in rows),
              "aggregate_gpu_seconds": sum(row["wall_seconds"] for row in rows),
              "peak_cuda_memory_mb": max(row["peak_cuda_memory_mb"] for row in rows),
              "note": "Selected query-center development diagnostic only. Raw scores reconstruct every prediction "
                      "and original Geometry identity. No official VIP or full eight-domain performance claim."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"dataset": result["dataset"], "query_count": result["query_count"], "decision": result["decision"],
                      "query_miou": {method: metric["mean_iou_percent"] for method, metric in metrics.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    merge(args.root, args.source, args.dataset, args.device)
