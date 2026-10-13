#!/usr/bin/env python3
"""Save paired qualitative masks from the pinned upstream VIP implementation.

This script is deliberately separate from Geov2's ``wide`` endpoint.  UDD5
uses the same fixed cross-dataset VIP vocabulary as the official full run;
VOC21 reads the pinned upstream ``cls_voc21.txt`` aliases and the upstream
natural-image resize/threshold settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinotool.config import CheckpointConfig
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPSettings, upstream_aliases, upstream_settings


PINNED_COMMIT = "5bd25ee03ec25c1538622cf7da661e8c0461e769"
UDD5_CLASSES = ("vegetation", "building", "road", "vehicle", "other")
VOC21_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("udd5", "voc21"), required=True)
    p.add_argument("--dinov3-repo", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--vocabulary-config", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def digest(key: str, dataset: str) -> int:
    return int(hashlib.sha256(f"qualitative:{dataset}:{key}".encode()).hexdigest()[:12], 16)


def miou(pred: np.ndarray, target: np.ndarray, classes: int) -> float:
    values = []
    for cls in range(classes):
        union = np.logical_or(pred == cls, target == cls).sum()
        if union:
            values.append(float(np.logical_and(pred == cls, target == cls).sum()) / float(union))
    return 100.0 * float(np.mean(values)) if values else 0.0


def checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(
        dinov3_repo=Path(args.dinov3_repo).resolve(),
        checkpoint_dir=root,
        dinotxt_weights=root / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        lvd_weights=root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        sat_weights=root / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        bpe_path=root / "bpe_simple_vocab_16e6.txt.gz",
    )


def load_udd5(root: Path):
    from eval_cver_external import discover_samples, load_rgb, load_target

    samples = discover_samples("udd5", root)
    return samples, lambda sample: load_rgb(sample, "udd5"), lambda sample, shape: load_target(sample, "udd5", shape), UDD5_CLASSES


def load_voc21(root: Path):
    from dinotool.natural_evaluation import discover_samples, load_rgb, load_target

    samples = discover_samples("voc21", root)
    return samples, load_rgb, lambda sample, shape: load_target(sample, "voc21", shape), VOC21_CLASSES


def resize_upstream_natural(rgb: torch.Tensor) -> torch.Tensor:
    """Match mmseg Resize(scale=(2048,336), keep_ratio=True) in cfg_voc21."""
    height, width = rgb.shape[-2:]
    ratio = min(2048.0 / float(width), 336.0 / float(height))
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    resized = cv2.resize(array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255.0)


@torch.inference_mode()
def predict_voc21(model: VIPOfficialAdapter, rgb: torch.Tensor, queries) -> np.ndarray:
    """Run the upstream cfg_voc21 sliding decoder and low-confidence rule."""
    original_height, original_width = rgb.shape[-2:]
    resized = resize_upstream_natural(rgb)
    height, width = resized.shape[-2:]
    h_crop = w_crop = 336
    h_stride = w_stride = 112
    h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
    logits_sum = torch.zeros((len(VOC21_CLASSES), height, width), device=model.device)
    count = torch.zeros((1, height, width), device=model.device)
    settings = VIPSettings(tau=3.0, tem=0.3, prob_thd=0.28, background=True, bg_idx=0,
                           resize_long_edge=448, slide_crop=336, slide_stride=112, logit_scale=40.0)
    for iy in range(h_grids):
        for ix in range(w_grids):
            bottom = min(iy * h_stride + h_crop, height)
            right = min(ix * w_stride + w_crop, width)
            top = max(bottom - h_crop, 0)
            left = max(right - w_crop, 0)
            patch = resized[:, top:bottom, left:right]
            patch_height, patch_width = patch.shape[-2:]
            # VOC images are resized to height <=336, and the upstream crop
            # is 336x336. Keep the same centered pad for rare short edges.
            if patch_height < h_crop or patch_width < w_crop:
                patch = F.pad(patch, (0, max(0, w_crop - patch_width),
                                      0, max(0, h_crop - patch_height)))
            # Rounding in the upstream Resize can produce a one-pixel-long
            # edge. The official head is fixed at 21x21, so clamp the padded
            # window to its exact 336x336 input shape.
            patch = patch[:, :h_crop, :w_crop]
            if tuple(patch.shape) != (3, h_crop, w_crop):
                raise RuntimeError(f"VOC21 crop shape invariant failed: resized={tuple(resized.shape)} "
                                   f"window={(top, bottom, left, right)} patch={tuple(patch.shape)}")
            patch_logits = model.crop_logits(patch, queries, settings)[:, :patch_height, :patch_width]
            logits_sum[:, top:bottom, left:right] += patch_logits
            count[:, top:bottom, left:right] += 1
    logits = logits_sum / count
    logits = F.interpolate(logits[None], size=(original_height, original_width), mode="bilinear", align_corners=False)[0]
    probabilities = logits.softmax(0)
    prediction = probabilities.argmax(0)
    prediction = prediction.masked_fill(probabilities.amax(0) < settings.prob_thd, settings.bg_idx)
    return prediction.to(torch.uint8).cpu().numpy()


def main(args: argparse.Namespace) -> None:
    upstream = Path(args.upstream_root).resolve()
    import subprocess

    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"VIP source revision changed: {commit}")
    root = Path(args.data_root).resolve()
    loader = load_udd5 if args.dataset == "udd5" else load_voc21
    all_samples, load_image, load_target, class_names = loader(root)
    samples = sorted(all_samples, key=lambda sample: digest(sample.key, args.dataset))[:args.limit]

    if args.dataset == "udd5":
        vocab = Path(args.vocabulary_config).resolve()
        specs = load_class_specs(vocab)
        if tuple(spec.name for spec in specs) != class_names:
            raise ValueError("UDD5 vocabulary class order does not match the locked taxonomy")
        aliases = tuple(spec.synonyms for spec in specs)
        vocabulary_source = str(vocab)
    else:
        vocab = upstream / "configs" / "cls_voc21.txt"
        aliases = upstream_aliases(vocab)
        if len(aliases) != len(class_names):
            raise ValueError("Pinned VOC21 alias file has the wrong class count")
        vocabulary_source = str(vocab)

    model = VIPOfficialAdapter(DINOTextSegmenter(checkpoints(args), device=args.device, amp=True), upstream)
    queries = model.encode_queries(tuple(class_names), aliases)
    settings = upstream_settings(args.dataset)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, sample in enumerate(samples, 1):
        rgb = load_image(sample)
        target = load_target(sample, tuple(rgb.shape[-2:]))
        if args.dataset == "voc21":
            prediction = predict_voc21(model, rgb, queries)
        else:
            prediction, _ = model.predict(rgb, queries, settings)
        stem = f"{number:02d}_{sample.key.replace('/', '_').replace('\\\\', '_').replace('.', '-') }".strip()
        # The Geov2 directory already owns the paired RGB/target; save copies
        # here so this result is independently inspectable and transferable.
        array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(output / f"{stem}_rgb.png")
        Image.fromarray(target.astype(np.uint8)).save(output / f"{stem}_target.png")
        Image.fromarray(prediction).save(output / f"{stem}_pred.png")
        np.save(output / f"{stem}_official_vip.npy", prediction)
        rows.append({"key": sample.key, "stem": stem, "miou": miou(prediction, target, len(class_names))})
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(samples),
                          "key": sample.key, "miou": rows[-1]["miou"]}), flush=True)
    signature = {
        "implementation": "vip-official-5bd25ee-qualitative",
        "upstream_commit": commit,
        "dataset": args.dataset,
        "classes": list(class_names),
        "alias_counts": [len(group) for group in aliases],
        "vocabulary_path": vocabulary_source,
        "sample_keys": [sample.key for sample in samples],
        "settings": vars(settings) if args.dataset == "udd5" else {
            "tau": 3.0, "tem": 0.3, "prob_thd": 0.28, "background": True,
            "bg_idx": 0, "resize": "scale=(2048,336), keep_ratio=True",
            "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0,
        },
        "checkpoint_manifest": checkpoint_manifest(model.backbone.checkpoints),
        "rows": rows,
    }
    (output / "signature.json").write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
