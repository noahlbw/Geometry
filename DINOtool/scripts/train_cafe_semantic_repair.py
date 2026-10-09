#!/usr/bin/env python3
"""Explicit COCO-171 + OEM repair recipe; target labels never enter training."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import ConfusionMatrix, build_model
from dinotool.cafe_region_assembly import CafeRegionAssembly, config_from_checkpoint, assembly_loss
from dinotool.cafe_ped import CafePED
from dinotool.parallel_evidence import ParallelEvidenceConfig
from dinotool.coco_full import ACTIVE_IDS, class_names, raw_samples, CocoStuffFullDataset
from dinotool.oem import OEM_CLASSES, OpenEarthMapDataset, discover_oem_samples
from dinotool.joint_sampling import SeededSourceDataset, SourceBatchSampler, domain_at, consumed_updates
from dinotool.ov_train import _write_json_atomic, _append_json_line, _atomic_torch_save, _seed_everything
from dinotool.proposal_supervision import proposal_partition_loss
from train_cafe_rc import normalize_image, evaluate as evaluate_oem
from eval_cafe_region_assembly_ade20k import distributed_context


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "official-root", "base-checkpoint", "bpe-path", "coco-root", "oem-root", "output-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--validate-every", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--source-coco-images", type=int, default=512)
    p.add_argument("--source-oem-images", type=int, default=64)
    p.add_argument("--new-lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--proposal-weight", type=float, default=0.0)
    return p.parse_args()


def evenly(samples, count):
    if not 1 <= count <= len(samples):
        raise ValueError("Invalid source validation subset size")
    return [samples[i] for i in np.linspace(0, len(samples) - 1, count, dtype=int)]


@torch.inference_mode()
def validate(model, coco_dev, oem_dev, text, names, rank, world, device):
    model.eval()
    metric = ConfusionMatrix(names)
    loader = DataLoader(CocoStuffFullDataset(coco_dev[rank::world], crop_size=224, training=False), batch_size=1)
    for rgb, target in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(normalize_image(rgb.to(device)), text["coco"])["logits"].argmax(1)
        metric.update(pred.cpu().numpy()[0], target.numpy()[0])
    matrix = torch.as_tensor(metric.matrix, device=device)
    dist.all_reduce(matrix)
    metric.matrix = matrix.cpu().numpy()
    coco = metric.summary()
    oem = evaluate_oem(model, oem_dev, text["oem"], argparse.Namespace(crop_size=448, amp="bf16"), rank, world, device)
    score = (coco["mean_iou"] + oem["mean_iou"]) / 2
    return dict(coco=coco, oem=oem, selection_score=score)


def main():
    args = parse_args()
    rank, local, world = distributed_context()
    if world < 2:
        raise ValueError("Launch the repair with torchrun on at least two GPUs")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85, device)
    _seed_everything(args.seed)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    if Path(args.coco_root).name != "COCOStuff2017" or Path(args.oem_root).name != "OpenEarthMap_wo_xBD":
        raise ValueError("Only the original COCO/OEM image roots are allowed")
    names = class_names(args.official_root)
    train_coco = raw_samples(args.coco_root, "train2017_cafe41.json")
    all_coco_dev = raw_samples(args.coco_root, "val2017_source_dev.json")
    train_oem = discover_oem_samples(Path(args.oem_root), "train", allow_missing_images=True)
    all_oem_dev = discover_oem_samples(Path(args.oem_root), "val", allow_missing_images=True)
    assert not {s.key for s in train_coco} & {s.key for s in all_coco_dev}
    assert not {s.key for s in train_oem} & {s.key for s in all_oem_dev}
    dev_coco = evenly(all_coco_dev, args.source_coco_images)
    dev_oem = evenly(all_oem_dev, args.source_oem_images)
    if args.proposal_weight < 0:
        raise ValueError("proposal-weight must be nonnegative")
    source = dict(protocol="COCO171_OEM_semantic_repair_v1", coco_train_images=len(train_coco),
                  oem_train_images=len(train_oem), coco_raw_ids=ACTIVE_IDS, coco_classes=names,
                  coco_dev_keys=[s.key for s in dev_coco], oem_dev_keys=[s.key for s in dev_oem],
                  target_used=False, selection="mean(COCO171 source-dev, OEM native-val)",
                  changes="same source image lists, full raw COCO labels, frozen visual encoder")
    source["proposal_supervision_weight"] = args.proposal_weight
    initial = torch.load(args.weights, map_location="cpu", weights_only=False)
    cafe, _, base = build_model(argparse.Namespace(official_root=args.official_root, checkpoint=args.base_checkpoint,
        bpe_path=args.bpe_path, device=str(device), window_size=224, model_mode="eval"))
    teacher = CafePED(copy.deepcopy(cafe), ParallelEvidenceConfig(arm="plain", stages=6)).eval().requires_grad_(False)
    model = CafeRegionAssembly(cafe, config_from_checkpoint(initial)).to(device)
    model.load_adapted_state_dict(initial["adapted_state"])
    model.cafe.backbone.requires_grad_(False)
    # The visual encoder stays in eval mode during every decoder update below.
    with torch.no_grad():
        text = {"coco": cafe.build_text_embeddings([list(names)]).detach(),
                "oem": cafe.build_text_embeddings([[c.name for c in OEM_CLASSES]]).detach()}
    groups = model.optimizer_groups(args.new_lr, args.head_lr, 0)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    training = DDP(model, device_ids=[local], broadcast_buffers=False)
    datasets = {"coco": CocoStuffFullDataset(train_coco, crop_size=224, training=True),
                "oem": OpenEarthMapDataset(train_oem, crop_size=448, training=True)}
    loaders = {}
    for i, (domain, dataset) in enumerate(datasets.items()):
        seeded = SeededSourceDataset(dataset, args.seed + i * 10000019, rank)
        sampler = SourceBatchSampler(seeded, rank=rank, world=world, batch_size=args.batch_size,
            seed=args.seed + i * 10000019, start_batch=0, batches=consumed_updates(args.updates, domain) * args.accum_steps)
        loaders[domain] = DataLoader(seeded, batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
    streams = {key: iter(loader) for key, loader in loaders.items()}
    if rank == 0:
        _write_json_atomic(out / "protocol.json", dict(args=vars(args), source=source, init_weights=args.weights,
                                                     architecture=model.architecture(), world_size=world))
    best = -1.0
    started = time.perf_counter()
    for step in range(args.updates + 1):
        if step % args.validate_every == 0 or step == args.updates:
            value = validate(model, dev_coco, dev_oem, text, names, rank, world, device)
            value["validation_step"] = step
            selected = value["selection_score"] > best
            best = max(best, value["selection_score"])
            if rank == 0:
                architecture = dict(model.architecture(), source_training=source["protocol"],
                                    proposal_supervision_weight=args.proposal_weight)
                payload = dict(format=model.checkpoint_format, architecture=architecture,
                    adapted_state=model.adapted_state_dict(), step=step, best_source_score=best,
                    validation=value, source=source, base_checkpoint=base, args=vars(args), world_size=world,
                    init_weights=args.weights)
                if selected:
                    _atomic_torch_save(out / "best_inference.pt", payload)
                _atomic_torch_save(out / "last.pt", dict(**payload, optimizer=optimizer.state_dict()))
                report = dict(step=step, coco_miou=value["coco"]["mean_iou_percent"],
                    oem_miou=100*value["oem"]["mean_iou"], source_score=100*value["selection_score"],
                    selected=selected, seconds=time.perf_counter()-started,
                    status="complete" if step == args.updates else "paused" if step == args.stop_after else "training")
                _append_json_line(out / "history.jsonl", report)
                _write_json_atomic(out / "status.json", report)
                print(json.dumps(report), flush=True)
            dist.barrier()
            if step == args.updates or step == args.stop_after:
                break
        training.train()
        model.cafe.backbone.eval()
        domain = domain_at(step)
        optimizer.zero_grad(set_to_none=True)
        factor = min(1.0, (step+1)/40) * max(0.1, 1 - step/args.updates)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
        for micro in range(args.accum_steps):
            rgb, target = next(streams[domain])
            image, target = normalize_image(rgb.to(device)), target.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    reference = teacher(image, text[domain])["logits"]
                output = training(image, text[domain], return_aux=True)
                loss, terms = assembly_loss(output, target, reference, smoothing=0.1, kd_weight=0.02, membership_weight=0.1)
                proposal_loss = loss.new_zeros(())
                if args.proposal_weight:
                    proposal_loss = proposal_partition_loss(output["proposal_logits"], target, *output["proposal_grid"])
                    loss = loss + args.proposal_weight * proposal_loss
            (loss / args.accum_steps).backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        if rank == 0 and (step+1) % 20 in (19, 0):
            report = dict(step=step+1, domain=domain, loss=float(loss), grad_norm=float(norm),
                          proposal_loss=float(proposal_loss), seconds=time.perf_counter()-started,
                          peak_gib=torch.cuda.max_memory_allocated()/2**30)
            _append_json_line(out / "steps.jsonl", report)
            print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
