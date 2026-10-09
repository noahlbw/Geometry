#!/usr/bin/env python3
"""Eight-GPU source-only COCO-Stuff training for matched CAFe-VC experiments.

LoveDA is intentionally neither discovered nor loaded. Checkpoints are selected
only by the fixed COCO source-dev manifest produced by prepare_coco_stuff.py.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cafedino_locked_loveda import IMAGENET_MEAN, IMAGENET_STD, build_model
from dinotool.cafe_vc import CafeVC, CafeVCConfig
from dinotool.cafe_rs import CafeRS, CafeRSConfig, rs_losses
from dinotool.cafe_query_reconstruction import (
    CafeQueryReconstruction,
    QueryReconstructionConfig,
    query_reconstruction_loss,
)
from dinotool.cafe_support_reobservation import (
    CafeSupportReobservation,
    SupportReobservationConfig,
    support_reobservation_loss,
)
from dinotool.cafe_pca import CafePCA, PCADINOConfig
from dinotool.coco_stuff import COCO_CAFE41_NAMES, CocoStuffCafe41Dataset, samples_from_manifest, validate_cafe41_mapping
from dinotool.ov_train import _append_json_line, _atomic_torch_save, _seed_everything, _seed_worker, _write_json_atomic


def parse_args(defaults=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-root", "checkpoint", "bpe-path", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--arm",
        choices=("plain", "cost_only", "concat", "vc", "rs", "query_reconstruction",
                 "support_reobservation", "serial", "parallel", "pca_epl", "pca_epl_fod",
                 "task_parallel"),
        default="vc",
    )
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--region-loss-weight", type=float, default=0.05)
    parser.add_argument("--affinity-loss-weight", type=float, default=0.02)
    parser.add_argument("--kd-weight", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means complete all requested epochs.")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU microbatch.")
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4, help="New module learning rate.")
    parser.add_argument("--cafe-lr", type=float, default=2e-5, help="Existing CAFe aggregation learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--content-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--relation-dim", type=int, default=32)
    parser.add_argument("--spatial-weight", type=float, default=0.05)
    parser.add_argument("--class-weight", type=float, default=0.05)
    parser.add_argument("--solver-steps", type=int, default=4)
    parser.add_argument("--solver-step-size", type=float, default=2.0 / 2.85)
    parser.add_argument("--relation-weight", type=float, default=0.05)
    parser.add_argument("--guide-dim", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--channel-init", choices=("pretrained", "fresh"), default="pretrained")
    parser.add_argument("--corrected-class-attention", action="store_true")
    parser.add_argument("--visual-tune-blocks", type=int, choices=(0, 2), default=0)
    parser.add_argument("--fod-weight", type=float, default=0.001)
    parser.add_argument("--reobservation-rounds", type=int, default=2)
    parser.add_argument("--detail-scale", type=int, default=2)
    parser.add_argument("--support-hidden-dim", type=int, default=32)
    parser.add_argument("--support-weight", type=float, default=0.05)
    parser.add_argument("--propagation-init", type=float, default=0.25)
    parser.add_argument("--feedback-init", type=float, default=0.50)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true", help="Run exactly two optimizer updates per rank.")
    parser.set_defaults(**(defaults or {}))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    root = Path(args.data_root).expanduser().resolve()
    if any("loveda" in part.casefold() for part in root.parts):
        raise ValueError("LoveDA is prohibited in source training and checkpoint selection")
    if root.name.casefold() != "cocostuff2017":
        raise ValueError("data-root must be the locked COCOStuff2017 directory")
    if args.crop_size < 112 or args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16 and at least 112")
    if min(args.epochs, args.batch_size, args.accum_steps, args.log_every) < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch size, accumulation, workers, or log interval")
    if args.max_updates < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("Invalid max-updates or label-smoothing")
    if min(args.lr, args.cafe_lr) <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
        raise ValueError("Invalid optimization settings")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory-fraction must be in (0, 1]")
    if args.arm in {"serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"}:
        PCADINOConfig(
            arm=args.arm, stages=6, corrected_class_attention=args.corrected_class_attention,
            channel_init=args.channel_init, visual_tune_blocks=args.visual_tune_blocks,
        ).validate()
        if args.fod_weight < 0:
            raise ValueError("fod-weight must be nonnegative")
    elif args.arm == "support_reobservation":
        SupportReobservationConfig(
            rounds=args.reobservation_rounds,
            detail_scale=args.detail_scale,
            hidden_dim=args.support_hidden_dim,
            support_weight=args.support_weight,
            propagation_init=args.propagation_init,
            feedback_init=args.feedback_init,
            residual_init=args.residual_scale,
        ).validate()
        if args.crop_size % (16 * args.detail_scale):
            raise ValueError("support-reobservation crop-size must divide both DINO scales")
    elif args.arm == "query_reconstruction":
        QueryReconstructionConfig(
            relation_dim=args.relation_dim,
            spatial_weight=args.spatial_weight,
            class_weight=args.class_weight,
            solver_steps=args.solver_steps,
            solver_step_size=args.solver_step_size,
            relation_weight=args.relation_weight,
            guide_dim=args.guide_dim,
            residual_scale=args.residual_scale,
            consistency_weight=args.consistency_weight,
        ).validate()
    elif args.arm == "rs":
        CafeRSConfig(content_dim=args.content_dim, heads=args.heads, stages=args.stages, modes=args.modes).validate()
        if min(args.region_loss_weight, args.affinity_loss_weight, args.kd_weight) < 0:
            raise ValueError("Auxiliary loss weights must be nonnegative")
    else:
        blocks = 0 if args.arm == "plain" else args.blocks
        CafeVCConfig(args.arm, args.content_dim, args.heads, args.kernel_size, blocks).validate()


def normalize_image(images: Tensor) -> Tensor:
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (images - mean) / std


def amp_context(args: argparse.Namespace):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")


def loss_with_ignore(logits: Tensor, target: Tensor, label_smoothing: float) -> Tensor:
    if not bool((target != 255).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float(), target, ignore_index=255, label_smoothing=label_smoothing)


def source_manifest(root: Path) -> dict[str, Any]:
    train_path = root / "manifests" / "train2017_cafe41.json"
    dev_path = root / "manifests" / "val2017_source_dev.json"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("Run prepare_coco_stuff.py for train and val before training.")
    train = samples_from_manifest(train_path, relocate_root=root, verify_paths=True)
    dev = samples_from_manifest(dev_path, relocate_root=root, verify_paths=True)
    if {sample.key for sample in train} & {sample.key for sample in dev}:
        raise ValueError("COCO train/source-dev image IDs overlap")
    return {
        "train_manifest": str(train_path),
        "source_dev_manifest": str(dev_path),
        "train_images": len(train),
        "source_dev_images": len(dev),
        "target_data_used": False,
    }


def summarize_matrix(matrix: Tensor) -> dict[str, Any]:
    diagonal = matrix.diag().to(torch.float64)
    union = matrix.sum(1).to(torch.float64) + matrix.sum(0).to(torch.float64) - diagonal
    iou = torch.where(union > 0, diagonal / union, torch.full_like(union, float("nan")))
    total = matrix.sum().to(torch.float64)
    return {
        "mean_iou": round(float(torch.nanmean(iou)), 6),
        "pixel_accuracy": round(float(diagonal.sum() / total) if total else 0.0, 6),
        "labeled_pixels": int(total),
        "per_class_iou": {name: None if torch.isnan(score) else round(float(score), 6) for name, score in zip(COCO_CAFE41_NAMES, iou)},
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate(model: CafeVC, samples, text: Tensor, args: argparse.Namespace, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    dataset = CocoStuffCafe41Dataset(samples[rank::world], crop_size=args.crop_size, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    matrix = torch.zeros((len(COCO_CAFE41_NAMES), len(COCO_CAFE41_NAMES)), dtype=torch.int64, device=device)
    started = time.perf_counter()
    for images, targets in loader:
        with amp_context(args):
            logits = model(normalize_image(images.to(device, non_blocking=True)), text)["logits"]
        prediction = logits.argmax(1)
        target = targets.to(device, non_blocking=True)
        valid = target != 255
        if bool(valid.any()):
            encoded = target[valid].to(torch.int64) * len(COCO_CAFE41_NAMES) + prediction[valid].to(torch.int64)
            matrix += torch.bincount(encoded, minlength=matrix.numel()).reshape_as(matrix)
    if world > 1:
        dist.all_reduce(matrix)
    summary = summarize_matrix(matrix.cpu())
    summary.update({"images": len(samples), "evaluation_size": args.crop_size, "seconds": time.perf_counter() - started})
    return summary


def save_checkpoint(model: CafeVC, optimizer: torch.optim.Optimizer, args: argparse.Namespace, output: Path, epoch: int, step: int, best: float, validation: dict[str, Any], source: dict[str, Any], base: dict[str, Any]) -> None:
    state = {
        "format": (model.checkpoint_format if args.arm in {"query_reconstruction", "support_reobservation",
                                                     "serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"} else
                   "cafe_rs_coco_v1" if args.arm == "rs" else "cafe_vc_coco_v2"),
        "architecture": model.architecture(),
        "adapted_state": model.adapted_state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_source_dev_miou": best,
        "validation": validation,
        "source": source,
        "base_checkpoint": base,
        "args": vars(args),
    }
    _atomic_torch_save(output / "last.pt", state)
    if validation["mean_iou"] >= best:
        _atomic_torch_save(output / "best.pt", state)
        inference = {key: value for key, value in state.items() if key not in {"optimizer"}}
        _atomic_torch_save(output / "best_inference.pt", inference)


def run(args: argparse.Namespace, rank: int, local_rank: int, world: int) -> None:
    validate_args(args)
    validate_cafe41_mapping()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    _seed_everything(args.seed)
    root, output = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    train_samples = samples_from_manifest(
        root / "manifests" / "train2017_cafe41.json", relocate_root=root,
        verify_paths=rank == 0,
    )
    dev_samples = samples_from_manifest(
        root / "manifests" / "val2017_source_dev.json", relocate_root=root,
        verify_paths=rank == 0,
    )
    if len(dev_samples) != 2500:
        raise ValueError("Locked COCO source-development manifest must contain exactly 2,500 images")
    if {sample.key for sample in train_samples} & {sample.key for sample in dev_samples}:
        raise ValueError("COCO train/source-development image IDs overlap")
    source = {
        "train_manifest": str(root / "manifests" / "train2017_cafe41.json"),
        "source_dev_manifest": str(root / "manifests" / "val2017_source_dev.json"),
        "train_images": len(train_samples), "source_dev_images": len(dev_samples),
        "target_data_used": False,
    }
    if args.smoke:
        train_samples = train_samples[:max(world * args.batch_size * args.accum_steps * 2, 16)]
        dev_samples = dev_samples[:max(world * args.batch_size, 8)]
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Use a fresh output directory or --resume; existing runs are never replaced")
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / "status.json", {"status": "preparing", "world_size": world, "started": time.time()})
        _write_json_atomic(output / "source_manifest.json", source)
    if world > 1:
        dist.barrier()
    train_set = CocoStuffCafe41Dataset(train_samples, crop_size=args.crop_size, training=True)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=False)
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=False, persistent_workers=args.workers > 0, worker_init_fn=_seed_worker, generator=generator)
    model_args = argparse.Namespace(official_root=args.official_root, checkpoint=args.checkpoint, bpe_path=args.bpe_path, device=str(device), window_size=args.crop_size, model_mode="eval")
    cafe, _, base = build_model(model_args)
    teacher = None
    if args.arm in {"serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"}:
        config = PCADINOConfig(
            arm=args.arm, stages=6, corrected_class_attention=args.corrected_class_attention,
            channel_init=args.channel_init, visual_tune_blocks=args.visual_tune_blocks,
        )
        model = CafePCA(cafe, config).to(device)
    elif args.arm == "support_reobservation":
        config = SupportReobservationConfig(
            rounds=args.reobservation_rounds,
            detail_scale=args.detail_scale,
            hidden_dim=args.support_hidden_dim,
            support_weight=args.support_weight,
            propagation_init=args.propagation_init,
            feedback_init=args.feedback_init,
            residual_init=args.residual_scale,
        )
        model = CafeSupportReobservation(cafe, config).to(device)
    elif args.arm == "query_reconstruction":
        teacher = CafeVC(copy.deepcopy(cafe), CafeVCConfig("plain", args.content_dim, args.heads,
                                                          args.kernel_size, 0)).requires_grad_(False).eval()
        config = QueryReconstructionConfig(
            relation_dim=args.relation_dim,
            spatial_weight=args.spatial_weight,
            class_weight=args.class_weight,
            solver_steps=args.solver_steps,
            solver_step_size=args.solver_step_size,
            relation_weight=args.relation_weight,
            guide_dim=args.guide_dim,
            residual_scale=args.residual_scale,
            consistency_weight=args.consistency_weight,
        )
        model = CafeQueryReconstruction(cafe, config).to(device)
    elif args.arm == "rs":
        config = CafeRSConfig(content_dim=args.content_dim, heads=args.heads, stages=args.stages, modes=args.modes)
        model = CafeRS(cafe, config, official_root=args.official_root).to(device)
    else:
        blocks = 0 if args.arm == "plain" else args.blocks
        config = CafeVCConfig(args.arm, args.content_dim, args.heads, args.kernel_size, blocks)
        model = CafeVC(cafe, config).to(device)
    with torch.no_grad():
        text = cafe.build_text_embeddings([list(COCO_CAFE41_NAMES)]).detach()
    groups = (model.optimizer_groups(args.lr, args.cafe_lr, args.cafe_lr * 0.1)
              if args.arm in {"query_reconstruction", "support_reobservation", "serial", "parallel",
                              "pca_epl", "pca_epl_fod", "task_parallel"} else [
                  {"params": [p for name, p in model.named_parameters()
                              if p.requires_grad and name.startswith("cafe.")],
                   "lr": args.cafe_lr, "initial_lr": args.cafe_lr},
                  {"params": [p for name, p in model.named_parameters()
                              if p.requires_grad and not name.startswith("cafe.")],
                   "lr": args.lr, "initial_lr": args.lr},
              ])
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=args.weight_decay)
    global_step, start_epoch, best = 0, 1, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_format = (model.checkpoint_format
                           if args.arm in {"query_reconstruction", "support_reobservation", "serial", "parallel",
                                           "pca_epl", "pca_epl_fod", "task_parallel"} else
                           "cafe_rs_coco_v1" if args.arm == "rs" else "cafe_vc_coco_v2")
        if state.get("format") != expected_format or state.get("architecture") != model.architecture():
            raise ValueError("Resume checkpoint architecture differs from this requested run")
        model.load_adapted_state_dict(state["adapted_state"])
        optimizer.load_state_dict(state["optimizer"])
        global_step, start_epoch, best = int(state["step"]), int(state["epoch"]) + 1, float(state["best_source_dev_miou"])
    if rank == 0:
        _write_json_atomic(output / "training_config.json", {
            "args": vars(args), "world_size": world, "global_batch": world * args.batch_size * args.accum_steps,
            "architecture": model.architecture(), "source": source, "backbone": base,
            "queries": list(COCO_CAFE41_NAMES),
            "selection": "COCO source-dev mIoU only; all external datasets prohibited",
            "teacher": ("independent frozen published CAFe including its original visual encoder"
                        if teacher is not None else None),
        })
        print(json.dumps({"event": "model_ready", "world_size": world, **model.architecture()}), flush=True)
    # Zero-initialized residual outputs deliberately defer gradients to the
    # visual read path until the first optimizer update.
    pca_arms = {"serial", "parallel", "pca_epl", "pca_epl_fod", "task_parallel"}
    training_model = (DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                          find_unused_parameters=args.arm not in pca_arms)
                      if world > 1 else model)
    updates_per_epoch = math.ceil(len(loader) / args.accum_steps)
    total_updates = 2 if args.smoke else args.max_updates or updates_per_epoch * args.epochs
    started = time.perf_counter()
    complete = False
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        generator.manual_seed(args.seed + rank + epoch * 1000)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        component_names = (("gate_anchor", "gate_spatial", "gate_class", "branch_cosine")
                           if args.arm == "task_parallel" else
                           ("ce_loss", "fod_loss", "branch_cosine")
                           if args.arm == "pca_epl_fod" else
                           ("ce_loss", "ownership_loss", "support_entropy",
                            "fine_revision", "feedback_delta")
                           if args.arm == "support_reobservation" else
                           ("ce_loss", "region_loss", "affinity_loss", "kd_loss",
                            "coarse_consistency_loss", "feature_preservation_error")
                           if args.arm == "query_reconstruction" else
                           ("ce_loss", "region_loss", "affinity_loss", "kd_loss")
                           if args.arm == "rs" else ())
        accumulated = torch.zeros(2 + len(component_names), dtype=torch.float64, device=device)
        batch_count = 0
        for batch_index, (images, targets) in enumerate(loader):
            if global_step >= total_updates:
                complete = True
                break
            synchronize = (batch_index + 1) % args.accum_steps == 0 or batch_index + 1 == len(loader)
            group_start = (batch_index // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, len(loader) - group_start)
            progress = min(1.0, (global_step + 1) / max(args.warmup_steps, 1)) if global_step < args.warmup_steps else max(1e-3, 0.5 * (1.0 + math.cos(math.pi * min(1.0, (global_step - args.warmup_steps) / max(total_updates - args.warmup_steps, 1)))))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * progress
            images, targets = normalize_image(images.to(device, non_blocking=True)), targets.to(device, non_blocking=True)
            context = training_model.no_sync() if isinstance(training_model, DDP) and not synchronize else nullcontext()
            with context:
                with amp_context(args):
                    if args.arm == "task_parallel":
                        outputs = training_model(images, text, return_aux=True)
                        loss = loss_with_ignore(outputs["logits"], targets, args.label_smoothing)
                        components = {key: outputs[key] for key in component_names}
                    elif args.arm == "pca_epl_fod":
                        outputs = training_model(images, text, return_aux=True)
                        ce_loss = loss_with_ignore(outputs["logits"], targets, args.label_smoothing)
                        loss = ce_loss + args.fod_weight * outputs["fod_loss"]
                        components = {"ce_loss": ce_loss, "fod_loss": outputs["fod_loss"],
                                      "branch_cosine": outputs["branch_cosine"]}
                    elif args.arm == "support_reobservation":
                        outputs = training_model(images, text, ownership_target=targets)
                        loss, components = support_reobservation_loss(
                            outputs, targets,
                            smoothing=args.label_smoothing,
                            support_weight=args.support_weight,
                        )
                    elif args.arm == "query_reconstruction":
                        with torch.no_grad():
                            reference = teacher(images, text)["logits"]
                        outputs = training_model(images, text, relation_target=targets)
                        loss, components = query_reconstruction_loss(
                            outputs, targets, reference,
                            smoothing=args.label_smoothing,
                            kd_weight=args.kd_weight,
                            relation_weight=args.relation_weight,
                            consistency_weight=args.consistency_weight,
                        )
                    elif args.arm == "rs":
                        outputs = training_model(images, text, return_aux=True, teacher=args.kd_weight > 0)
                        loss, components = rs_losses(outputs, targets, args.label_smoothing, args.region_loss_weight, args.affinity_loss_weight, args.kd_weight)
                    else:
                        logits = training_model(images, text)["logits"]
                        loss = loss_with_ignore(logits, targets, args.label_smoothing)
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Nonfinite loss on at least one DDP rank")
                (loss / group_size).backward()
            values = [loss.detach().double(), (targets != 255).sum().detach().double()]
            if component_names:
                values += [components[key].detach().double() for key in component_names]
            accumulated += torch.stack(values)
            batch_count += 1
            if synchronize:
                if args.arm in {"rs", "query_reconstruction", "support_reobservation", "serial", "parallel",
                                "pca_epl", "pca_epl_fod", "task_parallel"} and args.smoke and global_step == 1:
                    gradients = {name: float(p.grad.float().norm()) if p.grad is not None else None
                                 for name, p in model.named_parameters() if p.requires_grad}
                    missing = [key for key, value in gradients.items()
                               if value is None or not math.isfinite(value)]
                    if missing:
                        raise RuntimeError(f"Disconnected/nonfinite gradients: {missing}")
                    if teacher is not None and any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
                        raise RuntimeError("Frozen teacher received gradients")
                    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                        raise RuntimeError("Frozen student parameters received gradients")
                    if rank == 0:
                        _write_json_atomic(output / "gradient_report.json", gradients)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step >= total_updates:
                    complete = True
                if global_step % args.log_every == 0 or global_step == 1 or args.smoke:
                    values = accumulated.clone()
                    if world > 1:
                        dist.all_reduce(values)
                    if rank == 0:
                        record = {
                            "status": "training", "epoch": epoch, "step": global_step, "total_updates": total_updates,
                            "loss": float(values[0] / max(world * batch_count, 1)), "valid_pixels": int(values[1]),
                            "grad_norm": float(norm), "elapsed_seconds": time.perf_counter() - started,
                            "rank0_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        }
                        if component_names:
                            record["components"] = {
                                key: float(values[i + 2] / max(world * batch_count, 1))
                                for i, key in enumerate(component_names)
                            }
                        _append_json_line(output / "steps.jsonl", record)
                        _write_json_atomic(output / "status.json", record)
                        print(json.dumps(record), flush=True)
        validation = evaluate(model, dev_samples, text, args, rank, world, device)
        best = max(best, validation["mean_iou"])
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
        peaks = [torch.empty_like(peak) for _ in range(world)]
        if world > 1:
            dist.all_gather(peaks, peak)
        else:
            peaks[0] = peak
        if rank == 0:
            save_checkpoint(model, optimizer, args, output, epoch, global_step, best, validation, source, base)
            status = "smoke_complete" if args.smoke else "complete" if complete or epoch == args.epochs else "running"
            record = {"status": status, "epoch": epoch, "step": global_step, "best_source_dev_miou": best, "validation": validation, "per_rank_peak_gib": [float(value) for value in peaks], "elapsed_seconds": time.perf_counter() - started}
            _append_json_line(output / "history.jsonl", record)
            _write_json_atomic(output / "status.json", record)
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()
        if complete or args.smoke:
            break


def main(defaults=None) -> None:
    args = parse_args(defaults)
    rank, local_rank, world = (int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CAFe-VC training")
        torch.cuda.set_device(local_rank)
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
        run(args, rank, local_rank, world)
    except BaseException as error:
        if rank == 0 and Path(args.output_dir).is_dir():
            _write_json_atomic(Path(args.output_dir) / "failure.json", {"error": repr(error), "time": time.time()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
