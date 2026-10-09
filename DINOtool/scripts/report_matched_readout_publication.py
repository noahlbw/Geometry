"""Verify matched results and write scene-paired statistics for the manuscript."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


DATASETS = ("loveda", "udd5", "oem", "vdd", "potsdam", "vaihingen", "landcoverai", "flair1")
EXPECTED = dict(zip(DATASETS, (1669, 40, 384, 80, 504, 113, 1602, 15700)))


def group_key(dataset, key):
    if dataset in ("potsdam", "vaihingen"):
        return re.sub(r"_y\d+_x\d+\.tif$", "", key)
    if dataset == "landcoverai":
        return re.sub(r"_\d+$", "", key)
    if dataset in ("oem", "flair1"):
        return key.split("/")[0]
    return key


def miou(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    union = matrix.sum(-1)+matrix.sum(-2)-tp
    scores = np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union > 0)
    return np.nanmean(scores, axis=-1)*100


def class_errors(matrix, names):
    matrix = np.asarray(matrix, dtype=np.int64)
    target, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    rows = []
    for index, name in enumerate(names):
        rows.append({"name": name, "tp": int(tp[index]),
                     "fp": int(predicted[index]-tp[index]), "fn": int(target[index]-tp[index]),
                     "target_pixels": int(target[index]), "predicted_pixels": int(predicted[index]),
                     "precision_percent": float(100*tp[index]/predicted[index]) if predicted[index] else None,
                     "recall_percent": float(100*tp[index]/target[index]) if target[index] else None,
                     "predicted_target_ratio": float(predicted[index]/target[index]) if target[index] else None})
    return rows


def write_error_report(inputs, output, results):
    lines = ["# Class competition and error evidence", "",
             "Pixel-confusion evidence separates false activation from missed coverage. "
             "Class identity is not object size: these matrices cannot establish instance-level small-object or boundary errors.", ""]
    audit = {}
    for group, datasets in results.items():
        audit[group] = {}
        for dataset, row in datasets.items():
            merged = json.loads((inputs/group/dataset/"merged.json").read_text())
            audit[group][dataset] = {}
            for protocol, methods in merged["metrics"].items():
                errors = {method: class_errors(metric["confusion_matrix"], [entry["name"] for entry in metric["per_class"]])
                          for method, metric in methods.items()}
                audit[group][dataset][protocol] = errors
                lines += [f"## {group}/{dataset}/{protocol}", "",
                          "| Geometry class | IoU | Precision | Recall | Predicted/target area | FP | FN |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
                for entry, metric in zip(errors["Geometry"], methods["Geometry"]["per_class"]):
                    def display(value):
                        return "undefined" if value is None else f"{value:.3f}"
                    lines.append(f"| {entry['name']} | {metric['iou_percent']:.4f} | "
                                 f"{display(entry['precision_percent'])} | {display(entry['recall_percent'])} | "
                                 f"{display(entry['predicted_target_ratio'])} | {entry['fp']} | {entry['fn']} |")
                comparisons = ("SCLIP_Two", "VIPProxy_Two") if group == "core" else ("Native_Spatial", "SpatialOnly", "NACLIP_Last")
                lines += ["", "Changes below are comparator minus Geometry; negative FP/FN means fewer errors.", "",
                          "| Comparator | Class | IoU change | TP change | FP change | FN change |",
                          "| --- | --- | ---: | ---: | ---: | ---: |"]
                for method in comparisons:
                    for base, new, metric, base_metric in zip(errors["Geometry"], errors[method],
                                                             methods[method]["per_class"], methods["Geometry"]["per_class"]):
                        lines.append(f"| {method} | {base['name']} | "
                                     f"{metric['iou_percent']-base_metric['iou_percent']:+.4f} | "
                                     f"{new['tp']-base['tp']:+d} | {new['fp']-base['fp']:+d} | {new['fn']-base['fn']:+d} |")
                matrix = np.asarray(methods["Geometry"]["confusion_matrix"], dtype=np.int64).copy()
                np.fill_diagonal(matrix, 0)
                names = [entry["name"] for entry in errors["Geometry"]]
                biggest = np.argsort(matrix.ravel())[-3:][::-1]
                pairs = [f"{names[index//len(names)]} -> {names[index%len(names)]}: {matrix.ravel()[index]:,}"
                         for index in biggest if matrix.ravel()[index] > 0]
                lines += ["", "Largest true-to-predicted confusion flows: "+"; ".join(pairs)+".", ""]
    lines += ["## Limits", "",
              "These are post-prediction labeled audits, not signals used by the frozen readout. "
              "Do not turn class-specific observations into target-fitted inference thresholds. "
              "Boundary leakage, scale erasure and spectral mismatch require separate spatial/instance/input evidence; "
              "they are hypotheses unless independently measured."]
    (output/"class_errors.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    (output/"ERROR_ANALYSIS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze(directory, dataset, replicates):
    result = json.loads((directory/"merged.json").read_text())
    if (not result["coverage_verified"] or not result["exact_reference_geometry_confusion"]
            or not result["per_image_coverage_verified"] or result["processed_images"] != EXPECTED[dataset]):
        raise ValueError(f"Unverified full result: {dataset}")
    keys = result["signature"]["sample_keys"]
    groups = [group_key(dataset, key) for key in keys]
    unique = sorted(set(groups))
    lookup = {key: index for index, key in enumerate(unique)}
    group_ids = np.asarray([lookup[key] for key in groups])
    rng = np.random.default_rng(20261001)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=replicates)
    fields = {}
    with np.load(directory/"per_image_confusions.npz", allow_pickle=False) as arrays:
        if arrays["sample_keys"].tolist() != keys:
            raise ValueError("Per-image keys differ")
        for protocol, metrics in result["metrics"].items():
            samples = {}
            for method, metric in metrics.items():
                matrices = arrays[protocol+"__"+method]
                if not np.array_equal(matrices.sum(0), metric["confusion_matrix"]):
                    raise ValueError("Per-image confusion sum differs")
                grouped = np.zeros((len(unique), *matrices.shape[-2:]), np.int64)
                np.add.at(grouped, group_ids, matrices)
                boot = np.einsum("bg,gij->bij", weights, grouped, optimize=True)
                samples[method] = miou(boot)
            fields[protocol] = {method: {
                "miou": metric["mean_iou_percent"],
                "delta_vs_Geometry": metric["mean_iou_percent"]-metrics["Geometry"]["mean_iou_percent"],
                "paired_delta_95_interval": np.quantile(samples[method]-samples["Geometry"], [.025, .975]).tolist(),
                "per_class": metric["per_class"],
                "foreground_mean_iou": metric.get("foreground_mean_iou_percent"),
                "non_residual_mean_iou": metric.get("non_residual_mean_iou_percent")}
                for method, metric in metrics.items()}
    return {"images": len(keys), "groups": len(unique), "group_ids": unique,
            "bootstrap_replicates": replicates,
            "grouping": "parent/acquisition region" if dataset not in ("loveda", "udd5", "vdd") else
                        "image-level proxy; independent parent metadata unavailable",
            "metrics": fields, "diagnostics": result["diagnostics"],
            "combined_evaluator_cost": {
                "max_shard_elapsed_seconds": result["parallel_wall_seconds"],
                "sum_shard_elapsed_seconds": result["aggregate_gpu_seconds"],
                "peak_allocated_mib": result["peak_cuda_memory_mb"],
                "note": "All arms together. Shards were queued; maximum shard time is not end-to-end suite wall time. "
                        "Sum of elapsed times is not hardware-kernel active time."},
            "transitions": {protocol: {method: {key: val for key, val in values.items() if key != "counts"}
                            for method, values in methods.items()}
                            for protocol, methods in result["transitions"].items()}}


def main(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for group in ("core", "neighborhood"):
        source = args.inputs/group
        if source.exists():
            results[group] = {dataset: analyze(source/dataset, dataset, args.replicates) for dataset in DATASETS}
    if not results:
        raise ValueError("No complete matched suite")
    (output/"paired_statistics.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    lines = ["# Full matched readout comparison", "", "All values are mIoU percentages; changes are percentage points.",
             "These are frozen DINO.text operator adaptations with matched inputs/text, not complete official CLIP systems.",
             "Each Geometry confusion exactly matches its historical full-set reference. All per-image matrices reproduce full metrics.", ""]
    averages = {}
    for group, datasets in results.items():
        methods = list(next(iter(datasets.values()))["metrics"].values())[0].keys()
        lines += ["## "+group.capitalize(), "", "| Dataset/protocol | Images | "+" | ".join(methods)+" |",
                  "| --- | ---: | "+" | ".join("---:" for _ in methods)+" |"]
        for dataset, row in datasets.items():
            for protocol, metrics in row["metrics"].items():
                lines.append(f"| {dataset}/{protocol} | {row['images']} | "+
                             " | ".join(f"{metrics[method]['miou']:.4f}" for method in methods)+" |")
        averages[group] = {method: float(np.mean([
            row["metrics"]["D" if dataset == "loveda" else dataset][method]["miou"]
            for dataset, row in datasets.items()])) for method in methods}
        lines += ["", "Eight-domain equal mean uses LoveDA D once; LoveDA P is separately reported.", "",
                  "| Method | Eight-domain mean |", "| --- | ---: |"]
        lines += [f"| {method} | {value:.4f} |" for method, value in averages[group].items()]
        lines += ["", "### Foreground and non-residual metrics", "",
                  "| Dataset/protocol | Metric | "+" | ".join(methods)+" |",
                  "| --- | --- | "+" | ".join("---:" for _ in methods)+" |"]
        for dataset, protocol, field, label in (
                ("loveda", "D", "foreground_mean_iou", "D foreground mIoU"),
                ("udd5", "udd5", "non_residual_mean_iou", "non-residual mIoU"),
                ("landcoverai", "landcoverai", "foreground_mean_iou", "foreground mIoU")):
            metrics = datasets[dataset]["metrics"][protocol]
            lines.append(f"| {dataset}/{protocol} | {label} | "+
                         " | ".join(f"{metrics[method][field]:.4f}" for method in methods)+" |")
        lines += ["", "### Paired uncertainty", "",
                  "| Dataset | Resampling groups | Comparator minus Geometry | Delta | 95% paired interval |",
                  "| --- | ---: | --- | ---: | --- |"]
        for dataset, row in datasets.items():
            protocol = "D" if dataset == "loveda" else dataset
            for method, metric in row["metrics"][protocol].items():
                if method == "Geometry":
                    continue
                low, high = metric["paired_delta_95_interval"]
                lines.append(f"| {dataset} | {row['groups']} | {method} | {metric['delta_vs_Geometry']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        lines += ["", "Image-level intervals for LoveDA/UDD5/VDD are conditional proxies; adjacent scenes/frames may be correlated.",
                  "Potsdam/Vaihingen use parent scenes, LandCover.ai uses parent orthophotos, OEM uses regions, FLAIR uses acquisition domains.", ""]
        lines += ["### Combined evaluation cost", "",
                  "These are all-arm evaluation costs, not standalone inference. Queuing means max shard duration is not suite wall time.", "",
                  "| Dataset | Max shard elapsed s | Sum shard elapsed s | Peak allocated MiB |",
                  "| --- | ---: | ---: | ---: |"]
        for dataset, row in datasets.items():
            cost = row["combined_evaluator_cost"]
            lines.append(f"| {dataset} | {cost['max_shard_elapsed_seconds']:.3f} | "
                         f"{cost['sum_shard_elapsed_seconds']:.3f} | {cost['peak_allocated_mib']:.3f} |")
        lines.append("")
    lines += ["## Interpretation constraints", "",
              "Paired bootstrap uses 2000 replicates by default with fixed seed 20261001. "
              "Intervals are exploratory nominal 95% intervals, without multiple-comparison correction.",
              "Statistical separation supports an incremental performance claim, not a priority/novelty claim. "
              "The eight datasets have informed development. Do not select a different control or inference setting by dataset.",
              "The standalone ProxyCLIP/NACLIP attention-only controls test transfer of published operators to this head; "
              "their failure does not invalidate the original CLIP systems. Guarded VIP proxy excludes its official text scoring, "
              "physical field of view, thresholds and transductive distillation.",
              "Class IoUs, precision/recall/area, foreground metrics and beneficial/harmful transitions are available in paired_statistics.json."]
    (output/"MATCHED_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"domain_means.json").write_text(json.dumps(averages, indent=2)+"\n", encoding="utf-8")
    write_error_report(args.inputs, output, results)
    print(json.dumps(averages), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    main(parser.parse_args())
