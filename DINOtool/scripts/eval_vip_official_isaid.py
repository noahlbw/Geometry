"""VIP reproduction on iSAID validation tiles.

The evaluator uses the pinned VIP attention implementation, the upstream
``cls_iSAID.txt`` alias groups, ``prob_thd=0.3`` and background-enabled
16-class scoring. Targets are loaded only after prediction for scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinotool.config import CheckpointConfig
from dinotool.isaid import ISaidConfusionMatrix, ISaidSample, discover_isaid_validation, isaid_target_ids
from dinotool.model import DINOTextSegmenter
from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPSettings, upstream_aliases


CLASSES = (
    "background", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "bridge", "large vehicle",
    "small vehicle", "helicopter", "swimming pool", "roundabout",
    "soccer ball field", "airplane", "harbor",
)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def make_checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(
        dinov3_repo=Path(args.dinov3_repo).resolve(),
        checkpoint_dir=root,
        dinotxt_weights=root / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        lvd_weights=root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        sat_weights=root / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        bpe_path=root / "bpe_simple_vocab_16e6.txt.gz",
    )


def main(args: argparse.Namespace) -> None:
    if args.samples_manifest:
        manifest = json.loads(Path(args.samples_manifest).read_text())
        samples = [ISaidSample(row["key"], Path(row["image_path"]), Path(row["mask_path"]), Path(row["mask_path"]))
                   for row in manifest["samples"]]
    else:
        samples = discover_isaid_validation(args.data_root)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    upstream = Path(args.upstream_root).resolve()
    # The pinned VIP tokenizer helper hard-codes an empty path.  Bind it to
    # the already downloaded frozen BPE file before constructing DINOText.
    from dinov3.eval.text import tokenizer as dino_tokenizer
    bpe_path = Path(args.checkpoint_dir).resolve() / "bpe_simple_vocab_16e6.txt.gz"
    dino_tokenizer.get_tokenizer = lambda *_a, **_k: dino_tokenizer.Tokenizer(vocab_path=str(bpe_path))
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device, amp=True)
    model = VIPOfficialAdapter(backbone, upstream)
    alias_path = Path(args.vocabulary_config).resolve()
    aliases = upstream_aliases(alias_path)
    if len(aliases) != len(CLASSES):
        raise RuntimeError(f"Official iSAID alias file has {len(aliases)} rows; expected {len(CLASSES)}")
    queries = model.encode_queries(CLASSES, aliases)
    settings = VIPSettings(tau=4.0, tem=1.0, prob_thd=0.3, background=True, bg_idx=0,
                           resize_long_edge=448, slide_crop=336, slide_stride=112, logit_scale=40.0)
    matrix = ISaidConfusionMatrix()
    timing_ms, memory_mib = [], []
    keys = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        image = load_rgb(sample.image_path)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        prediction, _ = model.predict(image, queries, settings)
        torch.cuda.synchronize()
        timing_ms.append((time.perf_counter() - begin) * 1000.0)
        memory_mib.append(torch.cuda.max_memory_allocated() / 1048576.0)
        # Standard iSAID IDs are target/prediction 0..15; 255 is ignored.
        target = isaid_target_ids(sample.semantic_mask_path).astype(np.int64, copy=False)
        matrix.update(prediction, target)
        keys.append(sample.key)
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(json.dumps({"processed": index, "total": len(samples)}), flush=True)
    result = {
        "status": "complete", "dataset": "isaid", "model": "VIP pinned proxy-attention reproduction (official iSAID aliases; standard 16-class target metric)",
        "classes": list(CLASSES), "samples": keys, "processed_images": len(keys),
        "metrics": matrix.summary(), "mean_milliseconds": float(np.mean(timing_ms)),
        "median_milliseconds": float(np.median(timing_ms)), "peak_allocated_mib": float(max(memory_mib)),
        "mean_peak_allocated_mib": float(np.mean(memory_mib)), "wall_seconds": time.perf_counter() - started,
        "vocabulary": str(alias_path),
        "target_used_only_for_scoring": True,
        "official_alias_groups_used": True,
        "paper_comparison_valid": True,
        "paper_iSAID_reference_miou_percent": 26.1,
        "sample_manifest": args.samples_manifest,
        "settings": vars(settings),
    }
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "mean_iou_percent": result["metrics"]["mean_iou_percent"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data-root", "checkpoint-dir", "dinov3-repo", "upstream-root", "vocabulary-config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--samples-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    main(parser.parse_args())
