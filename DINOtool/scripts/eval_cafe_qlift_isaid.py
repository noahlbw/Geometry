#!/usr/bin/env python3
"""Locked, target-only native-resolution iSAID validation evaluation for Q-Lift.

This evaluator deliberately accepts source-selected Q-Lift checkpoints only.
It does not train, tune prompts, calibrate a threshold, crop from target labels,
or select a checkpoint from iSAID.  The sole target vocabulary is the official
15-category iSAID list plus a fixed ``background`` prompt.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import build_model, image_tensor
from dinotool.cafe_qlift import CafeQLift, config_from_checkpoint
from dinotool.isaid import (
    ISAID_CLASS_NAMES,
    ISaidForegroundConfusionMatrix,
    discover_isaid_validation,
    instance_size_recall_totals,
    isaid_target_ids,
)
from dinotool.ov_train import _write_json_atomic
from train_cafe_rc import sliding_logits


ISAID_WINDOW_SIZE = 448
ISAID_STRIDE = 224
ISAID_VOCABULARY = ("background", *ISAID_CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-root", "base-checkpoint", "bpe-path", "data-root", "weights", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_download_bindings(data_root: Path) -> dict[str, object]:
    """Bind evaluation to a completed target-only download without guessing layout."""
    root = data_root.resolve()
    marker_root = next((candidate for candidate in (root, *root.parents) if (candidate / "DOWNLOAD_COMPLETE").is_file()), None)
    if marker_root is None:
        raise ValueError(
            "iSAID evaluation data must be under a completed target-only download root with DOWNLOAD_COMPLETE."
        )
    bindings: dict[str, object] = {"download_root": str(marker_root), "download_complete": str(marker_root / "DOWNLOAD_COMPLETE")}
    for name in ("DOWNLOAD_COMPLETE", "download_manifest.tsv", "download_sha256.tsv", "official_source_url.txt"):
        path = marker_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required target-download provenance file: {path}")
        bindings[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return bindings


def _sample_keys_sha256(samples: list) -> str:
    return hashlib.sha256("\n".join(sample.key for sample in samples).encode("utf-8")).hexdigest()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    samples: list,
    text: torch.Tensor,
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    metrics = ISaidForegroundConfusionMatrix()
    local_samples = samples[rank::world]
    unlabeled_totals = np.zeros(2, dtype=np.int64)  # all unlabeled pixels, foreground predictions there
    size_totals = np.zeros((3, 3), dtype=np.int64)  # instances, target pixels, correct pixels
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, sample in enumerate(local_samples, 1):
        image = image_tensor(sample.image_path, 0, device)
        logits = sliding_logits(model, image, text, ISAID_WINDOW_SIZE, ISAID_STRIDE, args.amp)
        prediction = logits.argmax(0).cpu().numpy().astype(np.uint8, copy=False)
        target = isaid_target_ids(sample.semantic_mask_path)
        metrics.update(prediction, target)
        unlabeled = target == 255
        unlabeled_totals[0] += int(unlabeled.sum())
        unlabeled_totals[1] += int((prediction[unlabeled] != 0).sum())
        size_totals += instance_size_recall_totals(prediction, target, sample.instance_id_path)
        if rank == 0 and (index % 8 == 0 or index == len(local_samples)):
            _write_json_atomic(
                output / "status.json",
                {
                    "status": "evaluating",
                    "rank0_images": index,
                    "rank0_total": len(local_samples),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    matrix = torch.as_tensor(metrics.matrix, device=device)
    ignored = torch.tensor(metrics.ignored_pixels, dtype=torch.int64, device=device)
    unlabeled_tensor = torch.as_tensor(unlabeled_totals, device=device)
    size_tensor = torch.as_tensor(size_totals, device=device)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
    if world > 1:
        dist.all_reduce(matrix)
        dist.all_reduce(ignored)
        dist.all_reduce(unlabeled_tensor)
        dist.all_reduce(size_tensor)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    torch.cuda.synchronize(device)
    metrics.matrix = matrix.cpu().numpy()
    metrics.ignored_pixels = int(ignored.item())
    summary = metrics.summary()
    unlabeled_pixels, unlabeled_foreground = (int(value) for value in unlabeled_tensor.cpu().numpy())
    size_rows = size_tensor.cpu().numpy()
    summary.update(
        images=len(samples),
        seconds=time.perf_counter() - started,
        max_rank_peak_gib=float(peak.item()),
        unlabeled_pixels=unlabeled_pixels,
        unlabeled_foreground_prediction_pixels=unlabeled_foreground,
        unlabeled_foreground_prediction_percent=(
            None if not unlabeled_pixels else round(100.0 * unlabeled_foreground / unlabeled_pixels, 4)
        ),
        instance_size_target_recall=[
            {
                "name": name,
                "instances": int(row[0]),
                "target_pixels": int(row[1]),
                "correct_pixels": int(row[2]),
                "target_pixel_recall": None if not row[1] else float(row[2] / row[1]),
                "target_pixel_recall_percent": None if not row[1] else round(100.0 * row[2] / row[1], 4),
            }
            for name, row in zip(("small", "medium", "large"), size_rows)
        ],
    )
    return summary


def main() -> None:
    args = parse_args()
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("--memory-fraction must be in (0, 1].")
    rank, local_rank, world = (
        int(os.getenv(key, default))
        for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    output = Path(args.output_dir).resolve()
    try:
        root = Path(args.data_root).resolve()
        samples = discover_isaid_validation(root)
        target_bindings = _target_download_bindings(root)
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(hours=8), device_id=device)
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Evaluation output must be fresh: {output}")
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()

        payload = torch.load(args.weights, map_location="cpu", weights_only=False)
        if payload.get("format") != "cafe_qlift_v1":
            raise ValueError("iSAID target evaluator accepts only a source-selected cafe_qlift_v1 checkpoint.")
        config = config_from_checkpoint(payload)
        recorded_base = payload.get("base_checkpoint", {}).get("checkpoint", {})
        base = Path(args.base_checkpoint).resolve()
        if recorded_base.get("path") != str(base) or recorded_base.get("bytes") != base.stat().st_size:
            raise ValueError("Base checkpoint differs from the Q-Lift source-training base.")
        cafe, _, base_manifest = build_model(argparse.Namespace(
            official_root=args.official_root,
            checkpoint=args.base_checkpoint,
            bpe_path=args.bpe_path,
            device=str(device),
            window_size=ISAID_WINDOW_SIZE,
            model_mode="eval",
        ))
        model = CafeQLift(cafe, config).to(device).eval()
        model.load_adapted_state_dict(payload["adapted_state"])
        with torch.no_grad():
            text = cafe.build_text_embeddings([list(ISAID_VOCABULARY)]).detach()

        selected = Path(args.weights).resolve()
        record = {
            "status": "configured",
            "dataset": "iSAID validation target-only",
            "target_used_for_training_or_selection": False,
            "data_root": str(root),
            "target_download_bindings": target_bindings,
            "source_selected_checkpoint": {
                "path": str(selected), "bytes": selected.stat().st_size, "sha256": file_sha256(selected),
                "source_selection": payload.get("source", {}).get("selection"),
                "source_protocol": payload.get("source", {}).get("protocol"),
            },
            "base_checkpoint": base_manifest,
            "architecture": payload.get("architecture"),
            "model_variant": config.arm,
            "world_size": world,
            "amp": args.amp,
            "evaluation_size": 0,
            "window_size": ISAID_WINDOW_SIZE,
            "stride": ISAID_STRIDE,
            "vocabulary": list(ISAID_VOCABULARY),
            "sample_count": len(samples),
            "sample_keys_sha256": _sample_keys_sha256(samples),
        }
        if rank == 0:
            _write_json_atomic(output / "evaluation_config.json", record)
        if world > 1:
            dist.barrier()
        result = evaluate(model, samples, text, args, rank, world, device, output)
        result.update(status="complete", dataset=record["dataset"], model_variant=config.arm, world_size=world)
        if rank == 0:
            _write_json_atomic(output / "results.json", result)
            _write_json_atomic(output / "status.json", {key: result[key] for key in ("status", "images", "mean_iou_percent")})
            print(json.dumps({key: result[key] for key in ("status", "dataset", "model_variant", "images", "mean_iou_percent")}), flush=True)
        if world > 1:
            dist.barrier()
    except BaseException as error:
        if rank == 0 and output.is_dir() and not (output / "results.json").exists():
            _write_json_atomic(output / "failure.json", {"error": repr(error)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
