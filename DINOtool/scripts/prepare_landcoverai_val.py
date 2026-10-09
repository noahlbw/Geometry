#!/usr/bin/env python3
"""Produce only the official LandCover.ai v1 validation crops with split.py rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2


def prepare(root: Path, output: Path) -> dict:
    split = root / "val.txt"
    keys = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    if len(keys) != len(set(keys)):
        raise ValueError("Official validation IDs contain duplicates.")
    groups = {}
    for key in keys:
        stem, index = key.rsplit("_", 1)
        groups.setdefault(stem, []).append((key, int(index)))
    image_dir, mask_dir = output / "val" / "images", output / "val" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stem, items in sorted(groups.items()):
        image = cv2.imread(str(root / "images" / f"{stem}.tif"))
        mask = cv2.imread(str(root / "masks" / f"{stem}.tif"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Missing or mismatched original TIFF: {stem}")
        columns = (image.shape[1] + 511) // 512
        for key, index in items:
            row, column = divmod(index, columns)
            top, left = row * 512, column * 512
            rgb = image[top:top + 512, left:left + 512]
            target = mask[top:top + 512, left:left + 512]
            if rgb.shape[:2] != (512, 512) or target.shape != (512, 512):
                raise ValueError(f"Official ID references a partial edge: {key}")
            rgb_path, target_path = image_dir / f"{key}.jpg", mask_dir / f"{key}_m.png"
            if rgb_path.exists() or target_path.exists():
                raise FileExistsError(f"Use a fresh output directory: {key}")
            # cv2's default JPEG quality is the original split.py export setting.
            if not cv2.imwrite(str(rgb_path), rgb) or not cv2.imwrite(str(target_path), target):
                raise OSError(f"Failed to write official validation crop: {key}")
        print(f"Prepared {stem}: {len(items)} validation patches", flush=True)
    shutil.copyfile(split, output / "val.txt")
    result = {"dataset": "landcoverai-v1", "split": "official-val",
              "images": len(keys), "original_images": len(groups), "tile_size": 512,
              "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
              "export": "cv2 BGR read + default JPEG export, scalar mask PNG",
              "source": "https://landcover.ai.linuxpolska.com/"}
    (output / "preparation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
