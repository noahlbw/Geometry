#!/usr/bin/env python3
"""Source-only OpenEarthMap training of CAFe-RC with torchrun DDP.

LoveDA is neither discovered nor loaded. Source validation uses full native
images with fixed sliding windows. Audit cohorts/results are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cafedino_locked_loveda import build_model, IMAGENET_MEAN, IMAGENET_STD
from dinotool.cafe_rc import CafeRC, CafeRCConfig
from dinotool.oem import (OEM_CLASSES, OPEN_VOCABULARY_PRESERVATION_CLASSES, OpenEarthMapDataset,
                         discover_oem_samples, oem_source_manifest, oem_class_histogram, _read_rgb, _read_mask, _validate_raw_labels)
from dinotool.ov_train import (_ensure_disjoint_splits, _seed_everything, _seed_worker, _class_balanced_weights,
                              _segmentation_loss, _SegmentationConfusion, _atomic_torch_save, _write_json_atomic, _append_json_line)
from dinotool.inference import tile_starts


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("official-root", "checkpoint", "bpe-path", "data-root", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--crop-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=2, help="Per GPU microbatch")
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--cafe-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--preservation-weight", type=float, default=0.05)
    p.add_argument("--dice-weight", type=float, default=0.3)
    p.add_argument("--content-dim", type=int, default=128)
    p.add_argument("--variant", choices=("regional", "content", "baseline", "mask_plain", "mask_dual", "mask_fixed"), default="regional")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--allow-missing-source-images", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--smoke", action="store_true", help="Separate two-step DDP test, no model selection")
    p.add_argument("--memory-fraction", type=float, default=0.65)
    p.add_argument("--log-every", type=int, default=10)
    return p


def validate_args(args: argparse.Namespace) -> None:
    root = Path(args.data_root).expanduser().resolve()
    if any("loveda" in part.casefold() for part in root.parts):
        raise ValueError("LoveDA is prohibited in source training and checkpoint selection")
    if args.allow_missing_source_images and not (root / "xbd_files.csv").is_file():
        raise ValueError("Missing-image opt-in only supports the official wo_xBD mirror")
    if args.crop_size < 112 or args.crop_size % 16:
        raise ValueError("crop_size must be >=112 and divisible by 16")
    if min(args.epochs, args.batch_size, args.accum_steps, args.log_every) < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch size, accumulation, logging interval, or workers")
    if not 0 < args.memory_fraction <= 1 or not 0 <= args.dice_weight <= 1:
        raise ValueError("Invalid memory fraction or Dice weight")
    if min(args.lr, args.cafe_lr) <= 0 or min(args.weight_decay, args.warmup_steps, args.preservation_weight) < 0:
        raise ValueError("Invalid optimizer/loss settings")


def normalize_image(rgb: Tensor) -> Tensor:
    return (rgb - rgb.new_tensor(IMAGENET_MEAN)[None, :, None, None]) / rgb.new_tensor(IMAGENET_STD)[None, :, None, None]


def amp_context(args: argparse.Namespace):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")


def paired_loss(logits: Tensor, target: Tensor, class_weights: Tensor, dice_weight: float) -> Tensor:
    # An ignored-only crop must still participate in the same DDP graph.
    if not (target != 255).any():
        return logits.sum() * 0.0
    return _segmentation_loss(logits.float(), target, dice_weight=dice_weight, class_weights=class_weights)


@torch.inference_mode()
def sliding_logits(model: CafeRC, image: Tensor, text: Tensor, crop_size: int, stride: int, amp: str) -> Tensor:
    if not 0 < stride <= crop_size:
        raise ValueError("stride must be in (0, crop_size]")
    _, _, height, width = image.shape
    ph, pw = max(height, crop_size), max(width, crop_size)
    image = F.pad(image, (0, pw - width, 0, ph - height), mode="replicate")
    scores = torch.zeros((len(text), ph, pw), device=image.device, dtype=torch.float32)
    weights = torch.zeros((ph, pw), device=image.device, dtype=torch.float32)
    for top in tile_starts(ph, crop_size, crop_size - stride):
        for left in tile_starts(pw, crop_size, crop_size - stride):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp == "bf16"):
                logits = model(image[:, :, top:top + crop_size, left:left + crop_size], text)["logits"]
            scores[:, top:top + crop_size, left:left + crop_size] += logits[0].float()
            weights[top:top + crop_size, left:left + crop_size] += 1
    if not (weights > 0).all():
        raise RuntimeError("Sliding inference left uncovered pixels")
    return (scores / weights)[:, :height, :width]


def evaluate(model: CafeRC, samples, text: Tensor, args, rank: int, world: int, device) -> dict:
    model.eval()
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    started = time.perf_counter()
    local_samples = samples[rank::world]  # No DistributedSampler padding/duplicate validation images.
    for index, sample in enumerate(local_samples):
        rgb, raw = _read_rgb(sample.image_path), _read_mask(sample.mask_path)
        _validate_raw_labels(raw, sample.key)
        if rgb.shape[:2] != raw.shape:
            raise ValueError(f"Mismatched image/mask shapes: {sample.key}")
        target = np.full(raw.shape, 255, dtype=np.int64)
        valid = (raw >= 1) & (raw <= 8)
        target[valid] = raw[valid].astype(np.int64) - 1
        image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].to(device).float() / 255
        logits = sliding_logits(model, normalize_image(image), text, args.crop_size, args.crop_size // 2, args.amp)
        metrics.update(logits.argmax(0).cpu(), torch.from_numpy(target))
        if rank == 0 and (index + 1) % 12 == 0:
            print(json.dumps({"validation_rank0_images": index + 1, "rank0_total": len(local_samples)}), flush=True)
    matrix = metrics.matrix.to(device)
    if world > 1:
        dist.all_reduce(matrix)
    metrics.matrix = matrix.cpu()
    return {**metrics.summary(), "images": len(samples), "native_resolution": True, "seconds": time.perf_counter() - started}


def save_checkpoint(model, optimizer, scaler_state, args, epoch, step, best, rng, output, validation) -> None:
    state = {"format": "cafe_rc_v1", "architecture": model.architecture(), "rc_config": model.config.__dict__,
             "base_checkpoint": {"path": args.checkpoint, "bytes": Path(args.checkpoint).stat().st_size,
                                 "mtime_ns": Path(args.checkpoint).stat().st_mtime_ns},
             "adapted_state": model.adapted_state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "step": step, "best_source_miou": best, "validation": validation,
             "rng_by_rank": rng, "world_size": len(rng), "args": vars(args), "amp_state": scaler_state}
    _atomic_torch_save(output / "last.pt", state)
    if validation["mean_iou"] >= best:
        inference = {k: v for k, v in state.items() if k not in {"optimizer", "rng_by_rank", "amp_state"}}
        _atomic_torch_save(output / "best.pt", state)
        _atomic_torch_save(output / "best_inference.pt", inference)


def run(args, rank: int, local_rank: int, world: int) -> None:
    validate_args(args)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(4)
    _seed_everything(args.seed)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Use a fresh output directory or --resume; existing runs are never replaced")
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / "status.json", {"status": "preparing", "world_size": world, "started": time.time()})
    if world > 1:
        dist.barrier()
    train_samples = discover_oem_samples(args.data_root, "train", allow_missing_images=args.allow_missing_source_images)
    val_samples = discover_oem_samples(args.data_root, "val", allow_missing_images=args.allow_missing_source_images)
    _ensure_disjoint_splits(train_samples, val_samples)
    if rank == 0:
        counts = oem_class_histogram(train_samples)
        source_manifest = {
            "train": oem_source_manifest(args.data_root, "train", train_samples, allow_missing_images=args.allow_missing_source_images),
            "val": oem_source_manifest(args.data_root, "val", val_samples, allow_missing_images=args.allow_missing_source_images),
            "class_pixel_counts": counts.tolist(), "target_data_used": False,
        }
        manifest_path = output / "source_manifest.json"
        if args.resume and (not manifest_path.is_file() or json.loads(manifest_path.read_text()) != source_manifest):
            raise ValueError("Source image/split provenance changed since the saved run")
        _write_json_atomic(manifest_path, source_manifest)
    if world > 1:
        dist.barrier()
    counts = np.asarray(json.loads((output / "source_manifest.json").read_text())["class_pixel_counts"])
    class_weights = _class_balanced_weights(counts, power=0.5, maximum=4.0).to(device)
    if args.smoke:
        train_samples = train_samples[:max(world * args.batch_size * 4, 16)]
        val_samples = val_samples[:world]
    dataset = OpenEarthMapDataset(train_samples, crop_size=args.crop_size, training=True)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=False)
    loader_generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, worker_init_fn=_seed_worker, generator=loader_generator, drop_last=False,
                        persistent_workers=False)
    model_args = argparse.Namespace(official_root=args.official_root, checkpoint=args.checkpoint, bpe_path=args.bpe_path,
                                    device=str(device), window_size=args.crop_size, model_mode="eval")
    cafe, _, base_manifest = build_model(model_args)
    config = CafeRCConfig(content_dim=args.content_dim, variant=args.variant)
    model = CafeRC(cafe, config, teacher=args.preservation_weight > 0).to(device)
    with torch.no_grad():
        source_text = cafe.build_text_embeddings([[spec.name for spec in OEM_CLASSES]]).detach()
        preservation_text = cafe.build_text_embeddings([[spec.name for spec in OPEN_VOCABULARY_PRESERVATION_CLASSES]]).detach()
    training_text = torch.cat((source_text, preservation_text)) if args.preservation_weight > 0 else source_text
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("cafe.")], "initial_lr": args.cafe_lr, "lr": args.cafe_lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cafe.")], "initial_lr": args.lr, "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=args.weight_decay)
    start_epoch, global_step, best = 1, 0, -1.0
    resumed_rng = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["format"] != "cafe_rc_v1" or payload["rc_config"] != config.__dict__ or payload["world_size"] != world:
            raise ValueError("Resume architecture/world size mismatch")
        locked = ("data_root", "checkpoint", "crop_size", "batch_size", "accum_steps", "seed", "amp", "lr", "cafe_lr", "epochs", "preservation_weight")
        if any(payload["args"][key] != getattr(args, key) for key in locked):
            raise ValueError("Resume training protocol changed")
        model.load_adapted_state_dict(payload["adapted_state"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, global_step, best = payload["epoch"] + 1, payload["step"], payload["best_source_miou"]
        resumed_rng = payload["rng_by_rank"][rank]
    if rank == 0:
        _write_json_atomic(output / "training_config.json", {"args": vars(args), "world_size": world,
            "global_batch": world * args.batch_size * args.accum_steps, "architecture": model.architecture(),
            "backbone": base_manifest, "source_queries": [s.name for s in OEM_CLASSES],
            "preservation_queries": [s.name for s in OPEN_VOCABULARY_PRESERVATION_CLASSES],
            "selection": "OEM full native val mIoU only; no LoveDA or target fitting", "torch": torch.__version__,
            "protocol_note": "New supervised dense model run, independent of immutable 224-pixel readout audit"})
        print(json.dumps({"event": "model_ready", "world_size": world, **model.architecture()}), flush=True)
    training_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    _seed_everything(args.seed + rank)
    if resumed_rng is not None:
        torch.set_rng_state(resumed_rng["torch"])
        torch.cuda.set_rng_state(resumed_rng["cuda"], device)
        random.setstate(resumed_rng["python"])
        np.random.set_state(resumed_rng["numpy"])
    updates_per_epoch = math.ceil(len(loader) / args.accum_steps)
    total_updates = updates_per_epoch * args.epochs
    started = time.perf_counter()
    epochs = range(start_epoch, (start_epoch if args.smoke else args.epochs) + 1)
    for epoch in epochs:
        sampler.set_epoch(epoch)
        loader_generator.manual_seed(args.seed + epoch * 1000 + rank)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        num_batches = min(len(loader), args.accum_steps * 2) if args.smoke else len(loader)
        losses = torch.zeros(3, dtype=torch.float64, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        for batch_index, (rgb, target) in enumerate(loader):
            if batch_index >= num_batches:
                break
            group_start = (batch_index // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, num_batches - group_start)
            synchronize = (batch_index + 1) % args.accum_steps == 0 or batch_index + 1 == num_batches
            fraction = min(1.0, (global_step + 1) / max(args.warmup_steps, 1)) if global_step < args.warmup_steps else max(1e-3, 0.5 * (1 + math.cos(math.pi * min(1.0, (global_step - args.warmup_steps) / max(total_updates - args.warmup_steps, 1)))))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * fraction
            image = normalize_image(rgb.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)
            context = training_model.no_sync() if isinstance(training_model, DDP) and not synchronize else nullcontext()
            with context:
                with amp_context(args):
                    result = training_model(image, training_text, output_count=8, preserve_from=8 if args.preservation_weight > 0 else None)
                    segmentation = paired_loss(result["logits"], target, class_weights, args.dice_weight)
                    loss = segmentation + args.preservation_weight * result["preservation_loss"]
                valid_loss = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(valid_loss, op=dist.ReduceOp.MIN)
                if not valid_loss.item():
                    raise RuntimeError("Nonfinite loss on at least one rank")
                (loss / group_size).backward()
            losses += torch.stack((segmentation.detach().double(), result["preservation_loss"].detach().double(), loss.detach().double()))
            if synchronize:
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0 or args.smoke or global_step == 1:
                    average = losses.clone()
                    if world > 1:
                        dist.all_reduce(average)
                    average /= world * (batch_index + 1)
                    if rank == 0:
                        progress = {"status": "training", "epoch": epoch, "step": global_step,
                                    "updates_per_epoch": updates_per_epoch, "source_loss": average[0].item(),
                                    "preservation_loss": average[1].item(), "loss": average[2].item(),
                                    "grad_norm": float(norm), "elapsed_seconds": time.perf_counter() - started,
                                    "world_size": world, "rank0_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30}
                        if "mean_member_change" in result:
                            progress["member_change_last_batch_by_round"] = result["mean_member_change"].float().cpu().tolist()
                        _write_json_atomic(output / "status.json", progress)
                        _append_json_line(output / "steps.jsonl", progress)
                        print(json.dumps(progress), flush=True)
        validation = evaluate(model, val_samples, source_text, args, rank, world, device)
        best = max(best, validation["mean_iou"])
        rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
               "python": random.getstate(), "numpy": np.random.get_state()}
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
        peaks = [torch.empty_like(peak) for _ in range(world)]
        if world > 1:
            dist.all_gather(peaks, peak)
        else:
            peaks[0] = peak
        if rank == 0:
            save_checkpoint(model, optimizer, {}, args, epoch, global_step, best, all_rng, output, validation)
            record = {"status": "smoke_complete" if args.smoke else "complete" if epoch == args.epochs else "running",
                      "epoch": epoch, "step": global_step, "validation": validation, "best_source_miou": best,
                      "per_rank_peak_gib": [p.item() for p in peaks], "elapsed_seconds": time.perf_counter() - started,
                      "world_size": world}
            _append_json_line(output / "history.jsonl", record)
            _write_json_atomic(output / "status.json", record)
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()


def main() -> None:
    args = args_parser().parse_args()
    rank, local_rank, world = (int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this CAFe training entry point")
        torch.cuda.set_device(local_rank)
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
        run(args, rank, local_rank, world)
    except BaseException as error:
        if rank == 0:
            output = Path(args.output_dir)
            if output.is_dir():
                _write_json_atomic(output / "failure.json", {"error": repr(error), "time": time.time()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
