#!/usr/bin/env python3
"""Render CAFe-DINO depth-diagnosis tables as publication-ready heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--semantic-method", choices=("raw", "procrustes", "ridge"), default="ridge")
    parser.add_argument(
        "--semantic-metric",
        choices=("region", "pixel"),
        default="region",
        help="Use GT-interior pooled recognition or the legacy pixel mIoU readout.",
    )
    parser.add_argument("--spatial-affinity", choices=("feature", "query", "key", "value", "qkv"), default="qkv")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def annotated_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    output: Path,
    *,
    colorbar_label: str,
) -> None:
    width = max(7.0, len(column_labels) * 1.05)
    height = max(4.0, len(row_labels) * 0.48 + 1.8)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if value < np.nanmedian(values) else "black"
                axis.text(column, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def comparison_plot(result: dict, output: Path) -> None:
    labels = ["Static multilayer fusion"]
    values = [result["static_multilayer_fusion"]["mean_iou_percent"]]
    locked = result.get("locked_source_selection")
    if locked:
        labels.extend(["OEM-locked same depth", "OEM-locked dual depth"])
        values.extend(
            [
                locked["fixed_same_depth"]["target_metrics"]["mean_iou_percent"],
                locked["fixed_dual_depth"]["target_metrics"]["mean_iou_percent"],
            ]
        )
    oracle = result["oracle_pairs"]["per_image_oracle"]
    labels.extend(["Image oracle same depth", "Image oracle dual depth"])
    values.extend(
        [
            oracle["adaptive_same_depth_miou_percent"],
            oracle["adaptive_dual_depth_miou_percent"],
        ]
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=("#4c78a8", "#72b7b2", "#f58518", "#b279a2", "#e45756")[: len(labels)])
    axis.set_xticks(positions, labels=labels, rotation=18, ha="right")
    axis.set_ylabel("mIoU (%)")
    axis.set_title("Fixed, fused, and explicitly oracle depth selection")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0, max(values) * 1.18 if values else 1)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    result_path = Path(args.results).expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = result["config"]["layers"]
    class_names = result["dataset"]["classes"]
    semantic_key = "region_semantic" if args.semantic_metric == "region" else "semantic"
    if semantic_key not in result:
        raise ValueError(f"Result has no {semantic_key!r} semantic readout.")
    semantic = result[semantic_key][args.semantic_method]
    spatial = result["spatial"][args.spatial_affinity]
    combined_rows = []
    combined_labels = []
    for class_name in class_names:
        combined_rows.append(
            [semantic[str(layer)]["per_class_iou_percent"][class_name] for layer in layers]
        )
        combined_labels.append(f"{class_name} semantic-{args.semantic_metric}")
        combined_rows.append(
            [100.0 * spatial[str(layer)]["mean_per_class_metrics"][class_name]["region_iou"] for layer in layers]
        )
        combined_labels.append(f"{class_name} spatial")
    annotated_heatmap(
        np.asarray(combined_rows, dtype=np.float64),
        combined_labels,
        [f"L{layer}" for layer in layers],
        f"Semantic-{args.semantic_metric} ({args.semantic_method}) and spatial ({args.spatial_affinity}) depth readout",
        output_dir / "semantic_spatial_heatmap.png",
        colorbar_label="score (%)",
    )
    pair_matrix = np.asarray(result["oracle_pairs"]["matrix"], dtype=np.float64)
    annotated_heatmap(
        pair_matrix,
        [f"semantic L{layer}" for layer in layers],
        [f"spatial L{layer}" for layer in layers],
        "Semantic-spatial layer-pair mIoU",
        output_dir / "oracle_pair_matrix.png",
        colorbar_label="mIoU (%)",
    )
    comparison_plot(result, output_dir / "depth_selection_comparison.png")


if __name__ == "__main__":
    main()
