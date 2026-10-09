#!/usr/bin/env python3
"""Audit VIP Potsdam's per-class alias-count offset on identical probabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.vip_official_adapter import VIPOfficialAdapter, upstream_aliases, upstream_settings
from eval_vip_official_eight import Confusion, PINNED_COMMIT, checkpoint_config, protocol


ARMS = ("LSE_off", "LME_off", "LSE_on", "LME_on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "upstream-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", choices=("potsdam",), default="potsdam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    if not 0 < args.memory_fraction <= 1 or args.max_images < 0 or args.progress_every < 1:
        raise ValueError("Invalid evaluation parameters.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing existing output: {output}")
    upstream = Path(args.upstream_root).resolve()
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"VIP source revision changed: {commit}")
    all_samples, scored, load_image, load_mask = protocol(args)
    samples = all_samples[:args.max_images] if args.max_images else all_samples
    keys = [sample.key for sample in samples]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate sample keys.")
    names = tuple(scored[args.dataset])
    vocab = upstream / "configs/cls_potsdam.txt"
    aliases = upstream_aliases(vocab)
    counts = [len(group) for group in aliases]
    if counts != [2, 1, 3, 2, 1, 1]:
        raise ValueError("Pinned VIP Potsdam query counts changed.")
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = checkpoint_config(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=True)
    vip = VIPOfficialAdapter(backbone, upstream)
    bank = vip.encode_queries(names, aliases)
    settings = upstream_settings(args.dataset)
    offset = torch.tensor([math.log(count) / settings.tau for count in counts],
                          device=backbone.device)[:, None, None]
    matrices = {arm: Confusion(names, len(names)) for arm in ARMS}
    signature = {
        "implementation": "vip-potsdam-alias-count-offset-20260930",
        "upstream_commit": commit, "dataset": args.dataset, "arms": ARMS,
        "class_names": names, "alias_counts": counts,
        "offset": "subtract log(alias_count)/tau from each class logit",
        "tau": settings.tau, "threshold": settings.prob_thd,
        "vocabulary_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "sample_keys": keys,
        "sample_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        target = load_mask(sample, args.dataset, tuple(image.shape[-2:]))
        probabilities, _ = vip.predict_probabilities(image, bank, settings)
        adjusted = torch.softmax(probabilities.clamp_min(1e-30).log() - offset, dim=0)
        for prefix, values in (("LSE", probabilities), ("LME", adjusted)):
            confidence, prediction = values.max(0)
            plain = prediction.to(torch.uint8).cpu().numpy()
            gated = prediction.masked_fill(confidence < settings.prob_thd,
                                           settings.bg_idx).to(torch.uint8).cpu().numpy()
            matrices[prefix + "_off"].update(plain, target)
            matrices[prefix + "_on"].update(gated, target)
        if number % args.progress_every == 0 or number == len(samples):
            result = {
                "status": "complete" if number == len(samples) else "running",
                "processed_images": number, "total_images": len(samples),
                "metrics": {arm: matrix.summary() for arm, matrix in matrices.items()},
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(backbone.device) / 1048576
                                        if backbone.device.type == "cuda" else 0.0),
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                                 encoding="utf-8")
            print(json.dumps({"processed": number, "total": len(samples),
                              "miou": {arm: result["metrics"][arm]["mean_iou_percent"]
                                       for arm in ARMS}}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
