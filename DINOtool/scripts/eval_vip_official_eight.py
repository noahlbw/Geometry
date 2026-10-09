#!/usr/bin/env python3
"""Evaluate pinned upstream VIP inference on the locked eight-dataset masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from dinotool.config import CheckpointConfig
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.vip_official_adapter import VIPOfficialAdapter, upstream_aliases, upstream_settings

from eval_stride_ov_loveda_e1 import load_rgb as load_loveda_rgb, target_ids


DATASETS = ("loveda", "udd5", "oem", "vdd", "potsdam", "vaihingen", "landcoverai", "flair1")
OFFICIAL_CLASSES = {"potsdam": "cls_potsdam.txt", "vdd": "cls_vdd.txt",
                    "vaihingen": "cls_vaihingen.txt"}
PINNED_COMMIT = "5bd25ee03ec25c1538622cf7da661e8c0461e769"


class Confusion:
    def __init__(self, names: tuple[str, ...], predicted_classes: int):
        self.names = names
        self.predicted_classes = predicted_classes
        self.matrix = np.zeros((len(names), predicted_classes), dtype=np.int64)
        self.ignored = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError("VIP prediction and locked target shapes differ.")
        valid = (target >= 0) & (target < len(self.names))
        self.ignored += int((~valid).sum())
        if np.any(prediction[valid] >= self.predicted_classes):
            raise ValueError("VIP prediction exceeds its query vocabulary.")
        encoded = target[valid].astype(np.int64) * self.predicted_classes + prediction[valid]
        self.matrix += np.bincount(encoded, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, object]:
        classes = len(self.names)
        intersection = self.matrix[np.arange(classes), np.arange(classes)]
        target_count = self.matrix.sum(1)
        predicted_count = self.matrix.sum(0)[:classes]
        union = target_count + predicted_count - intersection
        iou = np.divide(intersection, union, out=np.full(classes, np.nan), where=union > 0)
        result = {
            "mean_iou_percent": round(float(np.nanmean(iou) * 100), 4),
            "pixel_accuracy_percent": round(float(intersection.sum() / max(target_count.sum(), 1) * 100), 4),
            "per_class": [{"name": name, "iou_percent": None if np.isnan(iou[index])
                           else round(float(iou[index] * 100), 4)}
                          for index, name in enumerate(self.names)],
            "confusion_matrix": self.matrix.tolist(),
            "ignored_pixels": self.ignored,
            "unscored_prediction_classes": self.predicted_classes - classes,
        }
        if self.names[0] == "background":
            result["foreground_mean_iou_percent"] = round(float(np.nanmean(iou[1:]) * 100), 4)
        if "other" in self.names:
            other_index = self.names.index("other")
            result["non_residual_mean_iou_percent"] = round(
                float(np.nanmean(np.delete(iou, other_index)) * 100), 4
            )
        return result


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "upstream-root", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--vocabulary-config", default="")
    parser.add_argument("--vocabulary-source", choices=("auto", "official", "json"), default="auto")
    parser.add_argument("--background-ablation", action="store_true",
                        help="Score both native confidence-to-background and plain argmax.")
    parser.add_argument("--finite-empty-rows", action="store_true",
                        help="Explicitly repair only undefined all-masked proxy rows with self-Value.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def protocol(args: argparse.Namespace):
    root = Path(args.data_root)
    if args.dataset == "loveda":
        samples = discover_loveda_samples(root)
        random.Random(args.sample_seed).shuffle(samples)
        load_image = lambda sample: load_loveda_rgb(sample.image_path)
        load_mask = lambda sample, key, shape: target_ids(sample.mask_path, key)
        names = {"P": ("building", "road", "water", "barren", "tree", "farm"),
                 "D": ("background", "building", "road", "water", "barren", "tree", "farm")}
    else:
        if args.dataset in ("udd5", "oem"):
            from eval_cver_external import class_names, discover_samples, load_rgb, load_target
        elif args.dataset in ("flair1", "landcoverai"):
            from dinotool.gear_datasets import class_names, discover_samples, load_rgb, load_target
        else:
            from dinotool.rs_external import class_names, discover_samples, load_rgb, load_target
        samples = discover_samples(args.dataset, root)
        names = {args.dataset: (("other", "wall", "road", "vegetation", "vehicle", "roof", "water")
                                if args.dataset == "vdd" else class_names(args.dataset))}
        load_image = lambda sample: load_rgb(sample, args.dataset)
        load_mask = lambda sample, key, shape: load_target(sample, args.dataset, shape)
    return samples, names, load_image, load_mask


def candidates(args: argparse.Namespace, scored: dict[str, tuple[str, ...]]):
    official = args.vocabulary_source == "official" or (
        args.vocabulary_source == "auto" and args.dataset in OFFICIAL_CLASSES
    )
    if official:
        if args.dataset not in OFFICIAL_CLASSES:
            raise ValueError("Official VIP query groups are unavailable for this dataset.")
        path = Path(args.upstream_root) / "configs" / OFFICIAL_CLASSES[args.dataset]
        groups = upstream_aliases(path)
        query_names = {key: (*names, "clutter") if args.dataset == "vaihingen" else names
                       for key, names in scored.items()}
        if any(len(groups) != len(names) for names in query_names.values()):
            raise ValueError("Upstream VIP alias groups do not match the scored taxonomy.")
        return {key: groups for key in scored}, query_names, path, "upstream_official"
    path = Path(args.vocabulary_config)
    if not path.is_file():
        raise FileNotFoundError("Unsupported-by-VIP dataset needs a fixed external alias JSON.")
    specs = load_class_specs(path)
    available = {spec.name: spec.synonyms for spec in specs}
    if any(set(names) - available.keys() for names in scored.values()):
        raise ValueError("External VIP candidates do not match locked class names.")
    aliases = {key: tuple(available[name] for name in names) for key, names in scored.items()}
    source = ("matched_json_vocabulary" if args.dataset in OFFICIAL_CLASSES
              else "cross_dataset_vip_method_adaptation")
    return aliases, scored, path, source


def checkpoint_config(args: argparse.Namespace) -> CheckpointConfig:
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(
        dinov3_repo=Path(args.dinov3_repo).resolve(), checkpoint_dir=root,
        dinotxt_weights=root / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        lvd_weights=root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        sat_weights=root / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        bpe_path=root / "bpe_simple_vocab_16e6.txt.gz",
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.shard_index < args.num_shards or not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid shard or CUDA memory fraction.")
    if args.progress_every < 1 or args.max_images < 0:
        raise ValueError("Invalid progress or image limit.")
    upstream = Path(args.upstream_root).resolve()
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"VIP source revision changed: {commit}")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing VIP evaluation: {output}")
    all_samples, scored, load_image, load_mask = protocol(args)
    if args.max_images:
        if args.dataset != "loveda":
            random.Random(args.sample_seed).shuffle(all_samples)
        all_samples = all_samples[:args.max_images]
    global_keys = [sample.key for sample in all_samples]
    samples = all_samples[args.shard_index::args.num_shards]
    if not samples or len(set(global_keys)) != len(global_keys):
        raise ValueError("Empty shard or duplicate global sample keys.")
    alias_groups, query_names, vocab_path, source_type = candidates(args, scored)
    output.mkdir(parents=True, exist_ok=True)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = checkpoint_config(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=True)
    if getattr(args, "finite_empty_rows", False):
        from dinotool.finite_vip_observer import FiniteVIPObserver
        model = FiniteVIPObserver(backbone, upstream)
    else:
        model = VIPOfficialAdapter(backbone, upstream)
    banks = {key: model.encode_queries(query_names[key], alias_groups[key]) for key in scored}
    settings = upstream_settings(args.dataset)
    if args.background_ablation and not settings.background:
        raise ValueError("This dataset has no official confidence-to-background rule.")
    signature = {
        "implementation": "vip-official-5bd25ee-eight-20260929",
        "upstream_commit": commit, "dataset": args.dataset, "source_type": source_type,
        "scored_classes": scored, "query_classes": query_names,
        "alias_counts": {key: [len(group) for group in alias_groups[key]] for key in scored},
        "sample_keys": [sample.key for sample in samples], "sample_count": len(samples),
        "sample_keys_sha256": digest([sample.key for sample in samples]),
        "global_sample_count": len(global_keys), "global_sample_keys_sha256": digest(global_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "settings": vars(settings), "template_count": len(model.templates),
        "background_ablation": args.background_ablation,
        "vocabulary_path": str(vocab_path.resolve()),
        "vocabulary_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "data_root": str(Path(args.data_root).resolve()),
        "short_edge_policy": "Pad crop to 336 if keep-ratio resize yields a shorter edge; report count.",
        "note": "Upstream VIP visual proxy attention and scoring; original masks and class scoring locked to GEAR run.",
    }
    if getattr(args, "finite_empty_rows", False):
        signature.update(implementation="vip-official-5bd25ee-finite-rows-20261003",
                         numeric_repair="Self-Value only on all-masked proxy rows; no threshold/settings changes.")
    write_json(output / "signature.json", signature)
    matrices = ({key: {arm: Confusion(names, len(query_names[key]))
                       for arm in ("threshold_off", "threshold_on")}
                for key, names in scored.items()} if args.background_ablation else
                {key: Confusion(names, len(query_names[key])) for key, names in scored.items()})
    padding_counts = {key: 0 for key in scored}
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    started = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            if args.background_ablation:
                probabilities, padded = model.predict_probabilities(image, bank, settings)
                confidence, prediction = probabilities.max(0)
                plain = prediction.to(torch.uint8).cpu().numpy()
                gated = prediction.masked_fill(confidence < settings.prob_thd,
                                               settings.bg_idx).to(torch.uint8).cpu().numpy()
                matrices[key]["threshold_off"].update(plain, target)
                matrices[key]["threshold_on"].update(gated, target)
            else:
                prediction, padded = model.predict(image, bank, settings)
                matrices[key].update(prediction, target)
            padding_counts[key] += padded
        if number % args.progress_every == 0 or number == len(samples):
            metrics = ({key: {arm: matrix.summary() for arm, matrix in arms.items()}
                        for key, arms in matrices.items()} if args.background_ablation else
                       {key: matrix.summary() for key, matrix in matrices.items()})
            result = {
                "status": "complete" if number == len(samples) else "running",
                "processed_images": number, "total_images": len(samples),
                "metrics": metrics,
                "short_edge_padded_crops": padding_counts,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(backbone.device) / 1048576
                                        if backbone.device.type == "cuda" else 0.0),
                "signature": signature,
            }
            if getattr(args, "finite_empty_rows", False):
                result.update(empty_proxy_rows=model.empty_rows, observed_proxy_rows=model.observed_rows)
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(samples),
                              "miou": {key: ({arm: metric["mean_iou_percent"]
                                              for arm, metric in metrics[key].items()}
                                             if args.background_ablation else
                                             metrics[key]["mean_iou_percent"]) for key in banks}}),
                  flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
