"""Evaluate the frozen Geov2 Anchored reader on iSAID validation triples.

The evaluator discovers official ``val/images`` RGB files and their two mask
files, runs Qwen/Kev-frozen vocabulary inference, and only then decodes masks
for scoring.  No image or mask is used by routing, alias selection, or the
Geov2 prediction itself.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinotool.anchored_slot_memory import anchored_slot_wide_observations
from dinotool.development_readout import biased_probabilities
from dinotool.efficient_deployment import reconstruction
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.isaid import (
    ISaidConfusionMatrix,
    ISaidSample,
    discover_isaid_validation,
    instance_size_recall_totals,
    isaid_target_ids,
)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


@torch.inference_mode()
def predict(model: FrozenGeometry, image: torch.Tensor) -> np.ndarray:
    source = anchored_slot_wide_observations(
        image,
        model.compact_geometry,
        model.query_slot_observer,
        model.strengths,
        wide_policy=model.profile.wide_policy,
        temperature=model.query_slot_temperature,
        beta=model.query_slot_beta,
        reconstruction=reconstruction,
        mode="anchored",
    )
    probability = model.reader.probabilities(
        source,
        model.profile,
        {},
        wide_scorer=model.wide_scorer,
        fusion="coupled",
    )
    probability = biased_probabilities(
        probability,
        model.config["background_bias"],
        model.config["background_index"],
    )
    confidence, labels = probability.max(0)
    background = model.config["background_index"]
    threshold = model.config["background_threshold"]
    if background is not None and threshold > 0:
        labels = labels.masked_fill(confidence < threshold, background)
    return labels.to(torch.uint8).cpu().numpy()


def main(args: argparse.Namespace) -> None:
    # The pinned DINO tokenizer helper hard-codes an empty path; bind it to
    # the frozen local BPE before FrozenGeometry constructs its backbone.
    from dinov3.eval.text import tokenizer as dino_tokenizer
    bpe_path = Path(args.checkpoint_dir).resolve() / "bpe_simple_vocab_16e6.txt.gz"
    dino_tokenizer.get_tokenizer = lambda *_a, **_k: dino_tokenizer.Tokenizer(vocab_path=str(bpe_path))
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    tiled = bool(args.samples_manifest)
    if tiled:
        manifest = json.loads(Path(args.samples_manifest).read_text())
        samples = [ISaidSample(row["key"], Path(row["image_path"]), Path(row["mask_path"]), Path(row["mask_path"]))
                   for row in manifest["samples"]]
    else:
        samples = discover_isaid_validation(args.data_root)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    if not samples:
        raise RuntimeError("No iSAID validation triples discovered")
    # The first invocation may create the frozen CPU text cache.  Once that
    # cache exists, share the visual backbone between local and wide readers;
    # this is the deployment path used for the memory/latency measurement.
    shared_visual = Path(args.text_cache).is_file()
    model = FrozenGeometry(
        args.config,
        checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,
        text_cache=args.text_cache,
        device=args.device,
        compact=True,
        efficient=True,
        wide_branch="query_slot",
        query_slot_capacity=args.query_slot_capacity,
        writeback="coupled",
        shared_visual=shared_visual,
    )
    model.profile = replace(model.profile, coupling=args.g0)
    if args.cost_json:
        # Cost protocol: fixed first seven official tiles, one warm-up and
        # seven timed repetitions per tile.  Targets are never loaded.
        cost_rows = []
        for sample in samples[:7]:
            image = load_rgb(sample.image_path)
            predict(model, image)
            timings = []
            peaks = []
            for _ in range(args.cost_repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                predict(model, image)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    peak = None
                timings.append((time.perf_counter() - start) * 1000.0)
                peaks.append(peak)
            cost_rows.append({
                "key": sample.key,
                "milliseconds": timings,
                "median_ms": float(np.median(timings)),
                "peak_allocated_mib": None if peaks[0] is None else float(max(peaks)),
            })
        output_cost = Path(args.cost_json)
        output_cost.parent.mkdir(parents=True, exist_ok=True)
        output_cost.write_text(json.dumps({
            "status": "complete",
            "model": "Geov2 Anchored",
            "g0": args.g0,
            "tile_manifest": args.samples_manifest,
            "images": cost_rows,
            "median_ms": float(np.median([r["median_ms"] for r in cost_rows])),
            "peak_allocated_mib": float(max(r["peak_allocated_mib"] for r in cost_rows)),
            "repeats": args.cost_repeats,
            "includes": "RGB load excluded; model already initialized; Geov2 inference only",
        }, indent=2) + "\n")
        print(json.dumps({"status": "complete", "cost_json": str(output_cost)}), flush=True)
        return
    matrix = ISaidConfusionMatrix()
    size_totals = np.zeros((3, 3), dtype=np.int64)
    started = time.perf_counter()
    rows = []
    for index, sample in enumerate(samples, 1):
        image = load_rgb(sample.image_path)
        prediction = predict(model, image)
        target = isaid_target_ids(sample.semantic_mask_path)
        matrix.update(prediction, target)
        if not tiled:
            size_totals += instance_size_recall_totals(prediction, target, sample.instance_id_path)
        rows.append(sample.key)
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(json.dumps({"processed": index, "total": len(samples)}), flush=True)
    result = {
        "status": "complete",
        "dataset": "isaid",
        "model": "Geov2 Anchored, DonorBefore local path, QuerySlot K=24, beta=0.5, coupled writer",
        "g0": args.g0,
        "query_slot_capacity": args.query_slot_capacity,
        "shared_visual_storage": shared_visual,
        "qwen_kev_vocabulary": str(Path(args.config).resolve()),
        "samples": rows,
        "processed_images": len(rows),
        "metric_protocol": "standard 16-class iSAID IDs: background=0, foreground=1..15, ignore=255",
        "metrics": matrix.summary(),
        "instance_size_recall": {
            "bins": ["small", "medium", "large"],
            "instance_count": size_totals[:, 0].tolist(),
            "target_pixels": size_totals[:, 1].tolist(),
            "correct_pixels": size_totals[:, 2].tolist(),
            "pixel_recall": [
                None if int(row[1]) == 0 else float(row[2] / row[1]) for row in size_totals
            ],
        },
        "wall_seconds": time.perf_counter() - started,
        "target_used_only_for_scoring": True,
        "text_selection": "Qwen candidates generated before evaluation; Kev taxonomy admission; no iSAID labels loaded",
        "limitations": "The input must contain official RGB validation images plus semantic and instance masks.",
        "sample_manifest": args.samples_manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "mean_iou_percent": result["metrics"]["mean_iou_percent"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "config",
        "data-root",
        "checkpoint-dir",
        "dinov3-repo",
        "upstream-root",
        "text-cache",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--g0", type=float, default=0.5)
    parser.add_argument("--query-slot-capacity", type=int, default=24)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--samples-manifest")
    parser.add_argument("--cost-json")
    parser.add_argument("--cost-repeats", type=int, default=7)
    main(parser.parse_args())
