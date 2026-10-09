#!/usr/bin/env python3
"""Paired component ablations on an explicit, fixed ADE development subset."""
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
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from eval_cafe_region_assembly_ade20k import canonical_classes, discover_samples, target_ids, distributed_context
from cafedino_locked_loveda import ConfusionMatrix, build_model, image_tensor
from dinotool.cafe_region_assembly import CafeRegionAssembly, config_from_checkpoint
from dinotool.cafe_ped import CafePED
from dinotool.parallel_evidence import ParallelEvidenceConfig
from dinotool.ov_train import _write_json_atomic
from train_cafe_rc import sliding_logits


class RawReadout(nn.Module):
    def __init__(self, model, use_head=False):
        super().__init__()
        self.model = model
        self.use_head = use_head

    def forward(self, image, text):
        x, y = self.model._encode(image)
        x = y if self.use_head else x
        scores = torch.einsum("bdhw,cd->bchw", F.normalize(x, dim=1), F.normalize(text, dim=-1))
        return {"logits": F.interpolate(scores.float(), image.shape[-2:], mode="bilinear", align_corners=False)}


def plain_state(assembly):
    return {k: v for k, v in assembly.items() if k.startswith("cafe.")}


def load_variant(cafe, payload, variant):
    if variant in ("official", "adapted_plain", "raw", "raw_y", "raw_pretrained_y"):
        model = CafePED(cafe, ParallelEvidenceConfig(arm="plain", stages=len(cafe.aggregator))).eval()
        if variant == "adapted_plain":
            model.load_adapted_state_dict(plain_state(payload["adapted_state"]))
        return RawReadout(model, use_head=variant != "raw") if variant.startswith("raw") else model
    model = CafeRegionAssembly(cafe, config_from_checkpoint(payload)).eval()
    if variant == "restore_visual":
        blocks = cafe.backbone.visual_model.backbone.blocks
        saved = {i: copy.deepcopy(blocks[i].state_dict()) for i in model.tuned_indices}
    model.load_adapted_state_dict(payload["adapted_state"])
    if variant == "restore_visual":
        for i, state in saved.items():
            blocks[i].load_state_dict(state)
    return model


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "data-root", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--images", type=int, default=64)
    p.add_argument("--window", type=int, default=224)
    p.add_argument("--stride", type=int, default=112)
    p.add_argument("--dinotxt-weights")
    p.add_argument("--lvd-weights")
    p.add_argument("--variants", nargs="+", default=["official", "adapted_plain", "adapted", "restore_visual", "raw"],
                   choices=["official", "adapted_plain", "adapted", "restore_visual", "raw", "raw_y", "raw_pretrained_y", "adapted_original_text"])
    args = p.parse_args()
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    names, _ = canonical_classes(Path(args.data_root))
    all_samples = discover_samples(Path(args.data_root))
    assert world <= args.images <= len(all_samples)
    indices = np.linspace(0, len(all_samples) - 1, args.images, dtype=int)
    samples = [all_samples[i] for i in indices]
    if rank == 0:
        _write_json_atomic(out / "protocol.json", dict(
            args=vars(args), sample_keys=[s[2] for s in samples],
            label_indexing="raw 1..150 -> 0..149, 0/255 ignored",
            selection="uniform image index, no metric-based selection", resolution=512, window=args.window, stride=args.stride,
            purpose="diagnosis only; ADE labels cannot select a training checkpoint"))
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    for variant in args.variants:
        cafe, _, _ = build_model(argparse.Namespace(
            official_root=args.official_root, checkpoint=args.base_checkpoint, bpe_path=args.bpe_path,
            device=str(device), window_size=args.window, model_mode="eval"))
        model = load_variant(cafe, payload, variant).to(device).eval().requires_grad_(False)
        if variant in ("raw_pretrained_y", "adapted_original_text"):
            before_text = cafe.build_text_embeddings([list(names)]).detach()
            original = torch.load(args.dinotxt_weights, map_location="cpu", weights_only=False)
            restored = {k: v for k, v in original.items() if variant == "raw_pretrained_y" or k.startswith("text_model.")}
            incompat = cafe.backbone.load_state_dict(restored, strict=False)
            allowed_missing = ("visual_model.backbone.",) if variant == "raw_pretrained_y" else ("visual_model.", "logit_")
            assert not incompat.unexpected_keys, incompat.unexpected_keys
            assert all(k.startswith(allowed_missing) for k in incompat.missing_keys), incompat.missing_keys
            if variant == "raw_pretrained_y":
                cafe.backbone.visual_model.backbone.load_state_dict(torch.load(args.lvd_weights, map_location="cpu", weights_only=False), strict=True)
            del original, restored
        text = cafe.build_text_embeddings([list(names)]).detach()
        if rank == 0 and variant in ("raw_pretrained_y", "adapted_original_text"):
            cosine = F.cosine_similarity(before_text, text).cpu().tolist()
            _write_json_atomic(out / (variant + "_text_drift.json"), dict(
                mean_cosine=float(np.mean(cosine)), per_class=dict(zip(names, cosine))))
        metrics = ConfusionMatrix(names)
        started = time.perf_counter()
        for index, (ip, mp, key) in enumerate(samples[rank::world]):
            image = image_tensor(ip, 512, device)
            pred = sliding_logits(model, image, text, args.window, args.stride, "bf16").argmax(0).cpu().numpy()
            metrics.update(pred, target_ids(mp))
            if rank == 0 and (index + 1) % 4 == 0:
                print(json.dumps(dict(variant=variant, rank0_done=index + 1, seconds=time.perf_counter() - started)), flush=True)
        matrix = torch.as_tensor(metrics.matrix, device=device)
        ignored = torch.tensor(metrics.ignored_pixels, device=device)
        if world > 1:
            dist.all_reduce(matrix)
            dist.all_reduce(ignored)
        metrics.matrix = matrix.cpu().numpy()
        metrics.ignored_pixels = int(ignored.item())
        result = dict(**metrics.summary(), variant=variant, images=len(samples), seconds=time.perf_counter() - started)
        if rank == 0:
            _write_json_atomic(out / (variant + ".json"), result)
            print(json.dumps({k: result[k] for k in ("variant", "images", "mean_iou_percent", "pixel_accuracy_percent", "seconds")}), flush=True)
        del model, cafe, text
        torch.cuda.empty_cache()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
