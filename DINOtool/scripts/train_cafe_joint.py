#!/usr/bin/env python3
"""Matched COCO/OEM joint fine-tuning; target datasets never enter this trainer."""
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

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import build_model
from train_cafe_coco import evaluate as evaluate_coco, normalize_image, source_manifest
from train_cafe_rc import evaluate as evaluate_oem
from dinotool.cafe_joint import JointCafe, JointConfig, joint_loss
from dinotool.cafe_vc import CafeVC, CafeVCConfig
from dinotool.coco_stuff import COCO_CAFE41_NAMES, CocoStuffCafe41Dataset, samples_from_manifest, validate_cafe41_mapping
from dinotool.oem import OEM_CLASSES, OpenEarthMapDataset, discover_oem_samples
from dinotool.joint_sampling import SeededSourceDataset, SourceBatchSampler, consumed_updates, domain_at
from dinotool.ov_train import _ensure_disjoint_splits, _seed_everything, _write_json_atomic, _append_json_line, _atomic_torch_save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("official-root", "checkpoint", "bpe-path", "coco-root", "oem-root", "output-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--arm", choices=("plain", "concat", "rs", "rs_aux"), required=True)
    p.add_argument("--updates", type=int, default=20896)
    p.add_argument("--validate-every", type=int, default=2612)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--new-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=2e-5)
    p.add_argument("--visual-lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--kd-weight", type=float, default=0.02)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--memory-fraction", type=float, default=0.70)
    p.add_argument("--smoke", action="store_true", help="Two warmup and two joint updates, covering both source domains")
    p.add_argument("--resume")
    args = p.parse_args()
    if args.smoke:
        args.updates, args.validate_every, args.warmup_steps = 4, 4, 2
    if min(args.updates, args.validate_every, args.batch_size, args.accum_steps) < 1 or args.workers < 0:
        raise ValueError("Invalid training budget or loader settings")
    if not 0 <= args.warmup_steps < args.updates or min(args.new_lr, args.head_lr, args.visual_lr) <= 0:
        raise ValueError("Invalid warmup or learning rates")
    if not 0 <= args.label_smoothing < 1 or min(args.weight_decay, args.kd_weight) < 0 or not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid loss or memory settings")
    return args


def load_sources(args):
    coco, oem = Path(args.coco_root).resolve(), Path(args.oem_root).resolve()
    if coco.name != "COCOStuff2017" or oem.name != "OpenEarthMap_wo_xBD" or any("loveda" in str(p).lower() for p in (coco, oem)):
        raise ValueError("Only the locked COCO and OEM source roots are permitted")
    if not (oem / "xbd_files.csv").is_file():
        raise ValueError("OEM missing-image opt-in requires the recorded wo_xBD mirror")
    validate_cafe41_mapping()
    record = {"coco": source_manifest(coco)}
    train = {"coco": samples_from_manifest(coco / "manifests/train2017_cafe41.json"),
             "oem": discover_oem_samples(oem, "train", allow_missing_images=True)}
    dev = {"coco": samples_from_manifest(coco / "manifests/val2017_source_dev.json"),
           "oem": discover_oem_samples(oem, "val", allow_missing_images=True)}
    _ensure_disjoint_splits(train["oem"], dev["oem"])
    counts = (len(train["coco"]), len(dev["coco"]), len(train["oem"]), len(dev["oem"]))
    if counts != (83559, 2500, 2303, 384):
        raise ValueError(f"Locked source counts changed: {counts}")
    record.update(oem={"root": str(oem), "allow_missing_xbd": True,
                       "train_keys": [s.key for s in train["oem"]], "val_keys": [s.key for s in dev["oem"]]},
                  target_data_used=False, selection="mean(COCO source-dev mIoU, OEM native-val mIoU)",
                  domain_schedule="COCO/OEM alternating complete optimizer updates, ratio 1:1",
                  coco_crop=224, oem_crop=448, augmentation="existing paired transforms; deterministic sample/cycle RNG")
    if args.smoke:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        train = {key: samples[:max(32, world * args.batch_size * 4)] for key, samples in train.items()}
        dev = {key: samples[:world * 2] for key, samples in dev.items()}
    return train, dev, record


def make_loaders(train, args, rank, world, completed):
    loaders = {}
    for i, domain in enumerate(("coco", "oem")):
        dataset = (CocoStuffCafe41Dataset(train[domain], crop_size=224, training=True) if domain == "coco"
                   else OpenEarthMapDataset(train[domain], crop_size=448, training=True))
        seed = args.seed + i * 10000019
        seeded = SeededSourceDataset(dataset, seed, rank)
        used = consumed_updates(completed, domain) * args.accum_steps
        total = consumed_updates(args.updates, domain) * args.accum_steps
        sampler = SourceBatchSampler(seeded, rank=rank, world=world, batch_size=args.batch_size,
                                     seed=seed, start_batch=used, batches=total - used)
        loaders[domain] = DataLoader(seeded, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                                     persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(seed + rank))
    return loaders


def set_rates(model, optimizer, args, step):
    warmup = step < args.warmup_steps
    model.set_warmup(warmup)
    if warmup:
        factor = (step + 1) / max(1, args.warmup_steps)
    else:
        factor = max(1e-3, 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, args.updates - args.warmup_steps))))
    for group in optimizer.param_groups:
        group["lr"] = 0.0 if warmup and group["role"] != "new" else group["initial_lr"] * factor


def amp(args):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")


def validate_resume(payload, args, source, base, architecture, world):
    if payload.get("format") != "cafe_joint_v1" or payload["architecture"] != architecture or payload["world_size"] != world:
        raise ValueError("Joint resume architecture/world-size mismatch")
    locked = [key for key in vars(args) if key not in {"resume", "output_dir"}]
    saved_base = {key: value for key, value in payload["base_checkpoint"].items() if key != "device"}
    current_base = {key: value for key, value in base.items() if key != "device"}
    if any(payload["args"].get(key) != getattr(args, key) for key in locked) or payload["source"] != source or saved_base != current_base:
        raise ValueError("Joint resume protocol or source provenance changed")


def validate(model, dev, text, args, rank, world, device):
    coco_args = argparse.Namespace(crop_size=224, batch_size=args.batch_size, workers=args.workers, amp=args.amp)
    oem_args = argparse.Namespace(crop_size=448, amp=args.amp)
    coco = evaluate_coco(model, dev["coco"], text["coco"], coco_args, rank, world, device)
    with torch.inference_mode():
        oem = evaluate_oem(model, dev["oem"], text["oem"], oem_args, rank, world, device)
    score = 0.5 * (coco["mean_iou"] + oem["mean_iou"])
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite source validation score")
    return dict(coco=coco, oem=oem, selection_score=score)


def save(model, teacher, optimizer, args, output, step, best, validation, source, base, rank, world):
    local_rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                 "python": random.getstate(), "numpy": np.random.get_state()}
    rng = [None] * world
    if world > 1:
        dist.all_gather_object(rng, local_rng)
    else:
        rng[0] = local_rng
    if rank == 0:
        payload = dict(format="cafe_joint_v1", architecture=model.architecture(), adapted_state=model.adapted_state_dict(),
                       step=step, best_source_score=best, validation=validation, source=source, base_checkpoint=base,
                       args=vars(args), world_size=world, optimizer=optimizer.state_dict(), rng_by_rank=rng)
        _atomic_torch_save(output / "last.pt", payload)
        if validation["selection_score"] >= best:
            _atomic_torch_save(output / "best_inference.pt", {k: v for k, v in payload.items() if k not in {"optimizer", "rng_by_rank"}})


def run(args, rank, local_rank, world):
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(4)
    _seed_everything(args.seed)
    train, dev, source = load_sources(args)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Use a fresh output directory; old experiments are preserved")
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / "status.json", dict(status="preparing", world_size=world, started=time.time()))
    if world > 1:
        dist.barrier()
    cafe, _, base = build_model(argparse.Namespace(official_root=args.official_root, checkpoint=args.checkpoint,
                               bpe_path=args.bpe_path, device=str(device), window_size=224, model_mode="eval"))
    # A separate original encoder is necessary once the student's visual blocks change.
    teacher = CafeVC(copy.deepcopy(cafe), CafeVCConfig(arm="plain", blocks=0)).requires_grad_(False).eval()
    model = JointCafe(cafe, JointConfig(arm=args.arm), official_root=args.official_root).to(device)
    if model.tuned_indices != (22, 23):
        raise ValueError(f"Unexpected official visual blocks: {model.tuned_indices}")
    with torch.no_grad():
        text = {"coco": cafe.build_text_embeddings([list(COCO_CAFE41_NAMES)]).detach(),
                "oem": cafe.build_text_embeddings([[c.name for c in OEM_CLASSES]]).detach()}
    optimizer = torch.optim.AdamW(model.optimizer_groups(args.new_lr, args.head_lr, args.visual_lr), weight_decay=args.weight_decay)
    completed, best, resume_rng = 0, -1.0, None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(payload, args, source, base, model.architecture(), world)
        model.load_adapted_state_dict(payload["adapted_state"])
        optimizer.load_state_dict(payload["optimizer"])
        completed, best, resume_rng = payload["step"], payload["best_source_score"], payload["rng_by_rank"][rank]
        if completed >= args.updates:
            raise ValueError("Checkpoint already completed the requested budget")
    loaders = make_loaders(train, args, rank, world, completed)
    streams = {domain: iter(loader) for domain, loader in loaders.items()}
    if rank == 0:
        _write_json_atomic(output / "source_manifest.json", source)
        _write_json_atomic(output / "training_config.json", dict(args=vars(args), architecture=model.architecture(),
                           base_checkpoint=base, source=source, world_size=world, max_global_batch=world * args.batch_size * args.accum_steps,
                           optimizer_groups=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
                           teacher="independent frozen official CAFe, including original visual encoder"))
        print(json.dumps(dict(event="model_ready", world_size=world, **model.architecture())), flush=True)
    training = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True) if world > 1 else model
    _seed_everything(args.seed + rank)
    if resume_rng:
        torch.set_rng_state(resume_rng["torch"])
        torch.cuda.set_rng_state(resume_rng["cuda"], device)
        random.setstate(resume_rng["python"])
        np.random.set_state(resume_rng["numpy"])
    visual_before = {str(i): {name: p.detach().clone() for name, p in model.cafe.backbone.visual_model.backbone.blocks[i].named_parameters()}
                     for i in model.tuned_indices} if args.smoke else None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    aggregate = torch.zeros((2, 6), dtype=torch.float64, device=device)
    gradient_report = {}
    for step in range(completed, args.updates):
        domain = domain_at(step)
        set_rates(model, optimizer, args, step)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.accum_steps):
            rgb, target = next(streams[domain])
            images = normalize_image(rgb.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)
            context = training.no_sync() if isinstance(training, DDP) and micro + 1 != args.accum_steps else nullcontext()
            with context:
                with amp(args):
                    with torch.no_grad():
                        reference = teacher(images, text[domain])["logits"]
                    outputs = training(images, text[domain])
                    loss, terms = joint_loss(outputs, target, reference, auxiliary=args.arm == "rs_aux",
                                            smoothing=args.label_smoothing, kd_weight=args.kd_weight)
                finite = torch.isfinite(loss).int()
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Nonfinite joint loss on at least one rank")
                (loss / args.accum_steps).backward()
            aggregate[0 if domain == "coco" else 1] += torch.stack([
                loss.detach().double(), terms["ce_loss"].detach().double(), terms["kd_loss"].detach().double(),
                terms["region_loss"].detach().double(), terms["affinity_loss"].detach().double(), loss.new_tensor(1).double()])
        if args.smoke and step == args.updates - 1:
            gradient_report = {name: float(p.grad.float().norm()) if p.grad is not None else None
                               for name, p in model.named_parameters() if p.requires_grad}
            missing = [name for name, value in gradient_report.items() if value is None or not math.isfinite(value)]
            if missing:
                raise RuntimeError(f"Disconnected/nonfinite joint gradients: {missing}")
            for i in model.tuned_indices:
                prefix = f"student.cafe.backbone.visual_model.backbone.blocks.{i}."
                if sum(v for k, v in gradient_report.items() if k.startswith(prefix)) <= 0:
                    raise RuntimeError(f"Visual block {i} received no effective gradients")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad) or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Frozen parameters or teacher received gradients")
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 20 == 0 or step == completed or args.smoke:
            values = aggregate.clone()
            if world > 1:
                dist.all_reduce(values)
            if rank == 0:
                report = dict(status="training", step=step + 1, total_updates=args.updates, world_size=world,
                              current_domain=domain, warmup=step < args.warmup_steps, grad_norm=float(norm),
                              elapsed_seconds=time.perf_counter() - started,
                              rank0_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                              learning_rates={g["role"]: g["lr"] for g in optimizer.param_groups},
                              domains={d: dict(zip(("loss", "ce", "kd", "region", "affinity"), (values[i, :5] / values[i, 5].clamp_min(1)).tolist())) for i, d in enumerate(("coco", "oem"))})
                _write_json_atomic(output / "status.json", report)
                _append_json_line(output / "steps.jsonl", report)
                print(json.dumps(report), flush=True)
            aggregate.zero_()
        if (step + 1) % args.validate_every == 0 or step + 1 == args.updates:
            validation = validate(model, dev, text, args, rank, world, device)
            best = max(best, validation["selection_score"])
            save(model, teacher, optimizer, args, output, step + 1, best, validation, source, base, rank, world)
            if args.smoke:
                for i in model.tuned_indices:
                    if all(torch.equal(visual_before[str(i)][name], p) for name, p in model.cafe.backbone.visual_model.backbone.blocks[i].named_parameters()):
                        raise RuntimeError(f"Visual block {i} did not actually update")
            if rank == 0:
                if args.smoke:
                    _write_json_atomic(output / "gradient_report.json", gradient_report)
                report = dict(status="smoke_complete" if args.smoke else "complete" if step + 1 == args.updates else "training",
                              step=step + 1, total_updates=args.updates, world_size=world, best_source_score=best,
                              validation=validation, elapsed_seconds=time.perf_counter() - started,
                              rank0_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)
                _write_json_atomic(output / "status.json", report)
                _append_json_line(output / "history.jsonl", report)
                print(json.dumps({k: v for k, v in report.items() if k != "validation"}), flush=True)
            if world > 1:
                dist.barrier()
            model.train()


def main():
    args = parse_args()
    rank, local_rank, world = (int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    try:
        torch.cuda.set_device(local_rank)
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
        run(args, rank, local_rank, world)
    except BaseException as error:
        if rank == 0 and Path(args.output_dir).is_dir():
            _write_json_atomic(Path(args.output_dir) / "failure.json", dict(error=repr(error), time=time.time()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
