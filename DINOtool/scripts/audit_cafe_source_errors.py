#!/usr/bin/env python3
"""Full native OEM-val error decomposition of a frozen/source-trained CAFe."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cafedino_locked_loveda import build_model
from dinotool.cafe_rc import CafeRC, CafeRCConfig
from dinotool.oem import OEM_CLASSES, discover_oem_samples, _read_rgb, _read_mask, _validate_raw_labels, oem_source_manifest
from dinotool.segmentation_errors import error_counts, summarize_errors
from train_cafe_rc import normalize_image, sliding_logits
from dinotool.ov_train import _write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--allow-missing-source-images", action="store_true")
    args = parser.parse_args()
    if "loveda" in str(Path(args.data_root).resolve()).lower():
        raise ValueError("Source error audit cannot use LoveDA")
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.65, device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        output = Path(args.output_dir).resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"Refusing existing audit directory: {output}")
            output.mkdir(parents=True)
        if world > 1:
            dist.barrier()
        payload = torch.load(args.weights, map_location="cpu", weights_only=False)
        if payload.get("format") != "cafe_rc_v1":
            raise ValueError("Unexpected checkpoint format")
        crop = int(payload["args"]["crop_size"])
        if Path(args.base_checkpoint).stat().st_size != payload["base_checkpoint"]["bytes"]:
            raise ValueError("Base checkpoint mismatch")
        cafe, _, _ = build_model(argparse.Namespace(official_root=args.official_root, checkpoint=args.base_checkpoint, bpe_path=args.bpe_path, device=str(device), window_size=crop, model_mode="eval"))
        config = dict(payload["rc_config"])
        if args.frozen:
            config["variant"] = "baseline"
        model = CafeRC(cafe, CafeRCConfig(**config), teacher=False).to(device).eval()
        if not args.frozen:
            model.load_adapted_state_dict(payload["adapted_state"])
        samples = discover_oem_samples(args.data_root, "val", allow_missing_images=args.allow_missing_source_images)
        names = [item.name for item in OEM_CLASSES]
        records = []
        started = time.perf_counter()
        with torch.inference_mode():
            text = cafe.build_text_embeddings([names]).detach()
            for index, sample in enumerate(samples[rank::world], 1):
                rgb, raw = _read_rgb(sample.image_path), _read_mask(sample.mask_path)
                _validate_raw_labels(raw, sample.key)
                if rgb.shape[:2] != raw.shape:
                    raise ValueError(f"Image/label shape mismatch: {sample.key}")
                target = np.full(raw.shape, 255, dtype=np.int64)
                valid = (raw >= 1) & (raw <= 8)
                target[valid] = raw[valid].astype(np.int64) - 1
                image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].to(device).float() / 255
                logits = sliding_logits(model, normalize_image(image), text, crop, crop // 2, args.amp)
                prediction = logits.argmax(0).cpu().numpy()
                records.append({"key": sample.key, "height": raw.shape[0], "width": raw.shape[1], **error_counts(prediction, target, len(names))})
                if rank == 0 and index % 12 == 0:
                    print(json.dumps({"rank0_images": index, "rank0_total": len(samples[rank::world])}), flush=True)
        all_records = [None] * world
        if world > 1:
            dist.all_gather_object(all_records, records)
        else:
            all_records[0] = records
        if rank == 0:
            records = sorted([item for partition in all_records for item in partition], key=lambda item: item["key"])
            if len(records) != len(samples) or len({item["key"] for item in records}) != len(samples):
                raise RuntimeError("Incomplete or duplicated audit samples")
            result = {**summarize_errors(records, names), "seconds": time.perf_counter() - started,
                      "variant": "frozen" if args.frozen else config["variant"], "weights": args.weights,
                      "crop_size": crop, "stride": crop // 2, "amp": args.amp, "target_data_used": False,
                      "source_manifest": oem_source_manifest(args.data_root, "val", samples, allow_missing_images=args.allow_missing_source_images)}
            _write_json_atomic(output / "per_image.json", {"records": records})
            _write_json_atomic(output / "results.json", result)
            print(json.dumps({"status": "complete", "images": len(records), "mean_iou_percent": result["mean_iou_percent"]}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
