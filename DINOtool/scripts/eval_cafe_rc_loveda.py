#!/usr/bin/env python3
"""External zero-shot CAFe-RC evaluation on labeled LoveDA validation data."""
from __future__ import annotations

import argparse
from datetime import timedelta
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
sys.path.insert(0, str(ROOT))

from cafedino_locked_loveda import build_model
from dinotool.cafe_rc import CafeRC, CafeRCConfig
from dinotool.loveda import LOVEDA_CLASS_NAMES, LoveDAConfusionMatrix, _target_to_ids, discover_loveda_samples
from train_cafe_rc import normalize_image, sliding_logits


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--crop-size", type=int, help="Must equal the saved training crop size.")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument("--max-images", type=int, help="Development-only bounded run; omitted means full validation.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the untouched CAFe base used to initialize CAFe-RC; do not load adapted weights.",
    )
    return parser


def distributed_context() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, local_rank, world


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def model_label(payload: dict, frozen: bool) -> str:
    if frozen:
        return "CAFe-DINO frozen published baseline"
    variant = payload["rc_config"]["variant"]
    if variant == "baseline":
        return "CAFe-DINO source-trained plain baseline"
    return "CAFe-RC" if variant == "regional" else f"CAFe {variant}"


def validate_args(args: argparse.Namespace, payload: dict[str, object]) -> int:
    root = Path(args.data_root).expanduser().resolve()
    if root.name.casefold() not in {"val", "validation"} or "loveda" not in str(root).casefold():
        raise ValueError("Evaluation is restricted to a LoveDA val/validation directory.")
    if not 0.0 < args.memory_fraction <= 1.0 or args.progress_every < 1 or (args.max_images is not None and args.max_images < 1):
        raise ValueError("Invalid memory fraction, progress interval, or image limit.")
    if payload.get("format") != "cafe_rc_v1":
        raise ValueError("Not a CAFe-RC checkpoint.")
    saved_crop = int(payload["args"]["crop_size"])
    crop_size = saved_crop if args.crop_size is None else args.crop_size
    if crop_size != saved_crop or crop_size < 112 or crop_size % 16:
        raise ValueError("Use the saved divisible-by-16 training crop size.")
    checkpoint = Path(args.base_checkpoint).resolve()
    if checkpoint.stat().st_size != payload["base_checkpoint"]["bytes"]:
        raise ValueError("Frozen base checkpoint size differs from the training checkpoint.")
    return crop_size


@torch.inference_mode()
def evaluate(
    model: CafeRC,
    samples,
    text: torch.Tensor,
    crop_size: int,
    amp: str,
    rank: int,
    world: int,
    device: torch.device,
    progress_every: int,
) -> dict[str, object]:
    model.eval()
    metrics = LoveDAConfusionMatrix()
    local_samples = samples[rank::world]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, sample in enumerate(local_samples, start=1):
        with Image.open(sample.image_path) as image_file:
            rgb = np.asarray(image_file.convert("RGB")).copy()
        with Image.open(sample.mask_path) as mask_file:
            raw_target = np.asarray(mask_file).copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device).float().div_(255.0)
        logits = sliding_logits(
            model,
            normalize_image(image),
            text,
            crop_size,
            crop_size // 2,
            amp,
        )
        metrics.update(logits.argmax(0).cpu().numpy(), _target_to_ids(raw_target))
        if rank == 0 and (index % progress_every == 0 or index == len(local_samples)):
            print(json.dumps({"rank0_images": index, "rank0_total": len(local_samples)}), flush=True)
    torch.cuda.synchronize(device)
    matrix = torch.from_numpy(metrics.matrix).to(device)
    ignored = torch.tensor(metrics.ignored_pixels, device=device, dtype=torch.int64)
    if world > 1:
        dist.all_reduce(matrix)
        dist.all_reduce(ignored)
    metrics.matrix = matrix.cpu().numpy()
    metrics.ignored_pixels = int(ignored.item())
    return {
        **metrics.summary(),
        "images": len(samples),
        "native_resolution": True,
        "window_size": crop_size,
        "stride": crop_size // 2,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = args_parser().parse_args()
    rank, local_rank, world = distributed_context()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CAFe-RC LoveDA evaluation requires CUDA.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        payload = torch.load(args.weights, map_location="cpu", weights_only=False)
        crop_size = validate_args(args, payload)
        output = Path(args.output_dir).expanduser().resolve()
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing evaluation output: {output}")
            output.mkdir(parents=True, exist_ok=True)
            write_json(
                output / "evaluation_config.json",
                {
                    "protocol": "source-selected checkpoint, external LoveDA validation evaluation only",
                    "weights": str(Path(args.weights).resolve()),
                    "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                    "data_root": str(Path(args.data_root).resolve()),
                    "classes": list(LOVEDA_CLASS_NAMES),
                    "crop_size": crop_size,
                    "stride": crop_size // 2,
                    "amp": args.amp,
                    "world_size": world,
                    "max_images": args.max_images,
                    "model_variant": model_label(payload, args.baseline),
                    "checkpoint_selection": "OEM full native validation mIoU only",
                    "LoveDA_used_for_training_or_selection": False,
                },
            )
        if world > 1:
            dist.barrier()
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        model_args = argparse.Namespace(
            official_root=args.official_root,
            checkpoint=str(Path(args.base_checkpoint).resolve()),
            bpe_path=args.bpe_path,
            device=str(device),
            window_size=crop_size,
            model_mode="eval",
        )
        cafe, _, _ = build_model(model_args)
        config_data = dict(payload["rc_config"])
        if args.baseline:
            config_data["variant"] = "baseline"
        model = CafeRC(cafe, CafeRCConfig(**config_data), teacher=False).to(device).eval()
        if not args.baseline:
            model.load_adapted_state_dict(payload["adapted_state"])
        with torch.no_grad():
            text = cafe.build_text_embeddings([list(LOVEDA_CLASS_NAMES)]).detach()
        samples = discover_loveda_samples(args.data_root)
        if args.max_images is not None:
            samples = samples[:args.max_images]
        result = evaluate(
            model, samples, text, crop_size, args.amp, rank, world, device, args.progress_every
        )
        if rank == 0:
            result["model_variant"] = model_label(payload, args.baseline)
            result["checkpoint"] = str(Path(args.weights).resolve())
            result["checkpoint_selection"] = "OEM full native validation mIoU only"
            result["LoveDA_used_for_training_or_selection"] = False
            write_json(output / "results.json", result)
            print(json.dumps(result), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
