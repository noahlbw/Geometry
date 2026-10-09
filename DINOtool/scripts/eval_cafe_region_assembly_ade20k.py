#!/usr/bin/env python3
"""Fixed external ADE20K-150 evaluation for a CAFe region-assembly checkpoint."""
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
from PIL import Image
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from cafedino_locked_loveda import ConfusionMatrix, build_model, image_tensor
from dinotool.cafe_region_assembly import CafeRegionAssembly, config_from_checkpoint
from dinotool.ov_train import _write_json_atomic
from train_cafe_rc import sliding_logits


ADE_CLASS_COUNT = 150
EVALUATION_SIZE = 512
WINDOW_SIZE = 224
STRIDE = 112
EXPECTED_IMAGES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-images", type=int, help="Development-only bounded smoke test.")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world


def canonical_classes(root: Path) -> tuple[tuple[str, ...], str]:
    mapping_path = root / "object_id2label.json"
    mapping_bytes = mapping_path.read_bytes()
    raw = json.loads(mapping_bytes)
    expected = {str(index) for index in range(ADE_CLASS_COUNT)}
    if set(raw) != expected:
        raise ValueError("ADE20K object_id2label.json must contain exactly contiguous IDs 0..149")
    names = []
    for index in range(ADE_CLASS_COUNT):
        value = raw[str(index)]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid ADE20K label at ID {index}")
        names.append(value.split(",", 1)[0].strip())
    if len(set(names)) != ADE_CLASS_COUNT:
        raise ValueError("Canonical ADE20K prompts must be unique")
    return tuple(names), hashlib.sha256(mapping_bytes).hexdigest()


def discover_samples(root: Path) -> list[tuple[Path, Path, str]]:
    image_dir = root / "images" / "validation"
    mask_dir = root / "annotations" / "validation"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError("ADE20K root must contain images/validation and annotations/validation")
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.casefold() in {".jpg", ".jpeg"}}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.casefold() == ".png"}
    if images.keys() != masks.keys():
        raise ValueError(f"ADE20K image/mask pairing mismatch: images={len(images)}, masks={len(masks)}")
    samples = [(images[key], masks[key], key) for key in sorted(images)]
    if len(samples) != EXPECTED_IMAGES:
        raise ValueError(f"ADE20K validation must contain {EXPECTED_IMAGES} samples, found {len(samples)}")
    return samples


def target_ids(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as source:
        raw = np.asarray(source).copy()
    if raw.ndim == 3:
        if raw.shape[2] < 3 or not np.array_equal(raw[..., :3], np.repeat(raw[..., :1], 3, axis=2)):
            raise ValueError(f"ADE20K mask must be scalar IDs: {mask_path}")
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"ADE20K mask must be 2D: {mask_path}")
    resized = np.asarray(Image.fromarray(raw).resize((EVALUATION_SIZE, EVALUATION_SIZE), Image.Resampling.NEAREST)).copy()
    # ADE20K PNG masks use one-based semantic IDs: 1..150. The model logits
    # are zero-based, so map them to 0..149. Raw 0 is unlabeled/background
    # and 255 is the usual ignore value; neither contributes to mIoU.
    valid = (resized >= 1) & (resized <= ADE_CLASS_COUNT)
    ignored = (resized == 0) | (resized == 255)
    invalid = np.unique(resized[~(valid | ignored)])
    if len(invalid):
        raise ValueError(f"Unexpected ADE20K labels at {mask_path}: {invalid.tolist()}")
    target = np.full(resized.shape, 255, dtype=np.int64)
    target[valid] = resized[valid].astype(np.int64) - 1
    return target


def validate_args(args: argparse.Namespace, root: Path, world: int) -> None:
    if root.name != "ade20k":
        raise ValueError("data-root must be the fixed processed ADE20K root")
    if args.max_images is not None and not world <= args.max_images <= EXPECTED_IMAGES:
        raise ValueError("max-images must be at least world-size and at most the full validation size")
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in (0, 1]")


@torch.inference_mode()
def evaluate(model, samples, text: torch.Tensor, args: argparse.Namespace, rank: int, world: int,
             device: torch.device, output: Path, class_names: tuple[str, ...]) -> dict:
    metrics = ConfusionMatrix(class_names)
    local_samples = samples[rank::world]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, (image_path, mask_path, _) in enumerate(local_samples, start=1):
        image = image_tensor(image_path, EVALUATION_SIZE, device)
        logits = sliding_logits(model, image, text, WINDOW_SIZE, STRIDE, args.amp)
        prediction = logits.argmax(0).cpu().numpy()
        metrics.update(prediction, target_ids(mask_path))
        if rank == 0 and (index % 24 == 0 or index == len(local_samples)):
            report = {"status": "evaluating", "rank0_images": index, "rank0_total": len(local_samples),
                      "elapsed_seconds": time.perf_counter() - started}
            _write_json_atomic(output / "status.json", report)
            print(json.dumps(report), flush=True)
    matrix = torch.as_tensor(metrics.matrix, device=device)
    ignored = torch.tensor(metrics.ignored_pixels, dtype=torch.int64, device=device)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
    if world > 1:
        dist.all_reduce(matrix)
        dist.all_reduce(ignored)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    torch.cuda.synchronize(device)
    metrics.matrix = matrix.cpu().numpy()
    metrics.ignored_pixels = int(ignored.item())
    return {
        **metrics.summary(),
        "images": len(samples),
        "seconds": time.perf_counter() - started,
        "max_rank_peak_gib": float(peak.item()),
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world = distributed_context()
    output = Path(args.output_dir).resolve()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("ADE20K evaluation requires CUDA")
        root = Path(args.data_root).resolve()
        validate_args(args, root, world)
        class_names, mapping_sha256 = canonical_classes(root)
        samples = discover_samples(root)
        if args.max_images is not None:
            samples = samples[:args.max_images]
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Evaluation output must be fresh: {output}")
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        payload = torch.load(args.weights, map_location="cpu", weights_only=False)
        if payload.get("format") != "cafe_region_assembly_v1":
            raise ValueError("ADE20K evaluator accepts only a CAFe region-assembly checkpoint")
        config = config_from_checkpoint(payload)
        recorded = payload["base_checkpoint"]["checkpoint"]
        base_checkpoint = Path(args.base_checkpoint).resolve()
        if base_checkpoint != Path(recorded["path"]).resolve() or base_checkpoint.stat().st_size != recorded["bytes"]:
            raise ValueError("Base checkpoint differs from the source-training base")
        cafe, _, base_manifest = build_model(argparse.Namespace(
            official_root=args.official_root,
            checkpoint=str(base_checkpoint),
            bpe_path=args.bpe_path,
            device=str(device),
            window_size=WINDOW_SIZE,
            model_mode="eval",
        ))
        model = CafeRegionAssembly(cafe, config).to(device).eval()
        model.load_adapted_state_dict(payload["adapted_state"])
        text = cafe.build_text_embeddings([list(class_names)]).detach()
        record = {
            "protocol": "ADE20K-150 external development: square-512, window-224, stride-112",
            "protocol_is_published_cafe_reproduction": False,
            "data_root": str(root),
            "images": len(samples),
            "evaluation_size": EVALUATION_SIZE,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "label_indexing": "raw 1..150 -> model 0..149; raw 0/255 ignored",
            "class_name_source": "object_id2label.json first comma-separated alias",
            "object_id2label_sha256": mapping_sha256,
            "classes": list(class_names),
            "amp": args.amp,
            "world_size": world,
            "max_images": args.max_images,
            "weights": str(Path(args.weights).resolve()),
            "base_checkpoint": base_manifest,
            "model_variant": config.arm,
            "checkpoint_selection": payload.get("source", {}).get("selection", "mean(COCO source-dev, OEM native-val) mIoU"),
            "source_training_protocol": payload.get("source", {}).get("protocol", "locked_COCO41_OEM"),
            "ADE20K_used_for_training_or_selection": False,
            "architecture": payload["architecture"],
        }
        if rank == 0:
            _write_json_atomic(output / "evaluation_config.json", record)
        result = evaluate(model, samples, text, args, rank, world, device, output, class_names)
        result.update(status="complete", model_variant=config.arm, weights=record["weights"], world_size=world,
                      protocol=record["protocol"], evaluation_size=EVALUATION_SIZE,
                      window_size=WINDOW_SIZE, stride=STRIDE)
        if rank == 0:
            _write_json_atomic(output / "results.json", result)
            _write_json_atomic(output / "status.json", {key: result[key] for key in ("status", "images", "mean_iou_percent")})
            print(json.dumps({key: result[key] for key in ("status", "model_variant", "images", "mean_iou_percent")}), flush=True)
        if world > 1:
            dist.barrier()
    except BaseException as error:
        if rank == 0 and output.is_dir() and not (output / "results.json").exists():
            _write_json_atomic(output / "failure.json", {"error": repr(error)})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
