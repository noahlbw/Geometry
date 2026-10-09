#!/usr/bin/env python3
"""Extract, audit, and remap COCO-Stuff 2017 for CAFe's fixed 41-class source vocabulary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dinotool.coco_stuff import (
    COCO_CAFE41_NAMES,
    COCO_CAFE41_RAW_IDS,
    COCO_CAFE41_REMAP,
    CocoStuffSample,
    validate_cafe41_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="COCOStuff2017 root containing archives/.")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def extract_members(archive_path: Path, destination: Path, *, split: str, suffix: str) -> int:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing archive: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        infos = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(suffix) and f"/{split}/" in f"/{info.filename}"
        ]
        if not infos:
            raise ValueError(f"Archive {archive_path.name} has no {split} {suffix} members.")
        for info in infos:
            name = Path(info.filename).name
            if name != f"{Path(name).stem}{suffix}":
                raise ValueError(f"Unsafe archive member name: {info.filename}")
            target = destination / name
            if target.exists():
                if target.stat().st_size != info.file_size:
                    raise FileExistsError(f"Refusing to overwrite incomplete extraction: {target}")
                continue
            temporary = target.with_name(f".{target.name}.tmp")
            with archive.open(info) as source, temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                temporary.unlink(missing_ok=True)
                raise IOError(f"Incomplete extraction for {info.filename}")
            os.replace(temporary, target)
            extracted += 1
    return extracted


def discover_pairs(root: Path, split: str) -> list[CocoStuffSample]:
    image_dir = root / "images" / split
    label_dir = root / "labels_raw" / split
    images = {path.stem: path for path in image_dir.glob("*.jpg")}
    labels = {path.stem: path for path in label_dir.glob("*.png")}
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_images or missing_labels:
        raise ValueError(
            f"{split} RGB/mask mismatch: {len(missing_labels)} images without labels and "
            f"{len(missing_images)} labels without images."
        )
    if not images:
        raise ValueError(f"No COCO pairs found for {split}.")
    return [CocoStuffSample(key=key, image_path=images[key], label_path=labels[key]) for key in sorted(images)]


def create_derived_labels(root: Path, split: str, pairs: list[CocoStuffSample]) -> tuple[list[CocoStuffSample], list[CocoStuffSample], np.ndarray, int]:
    destination = root / "labels_cafe41" / split
    destination.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(len(COCO_CAFE41_NAMES), dtype=np.int64)
    all_derived_pairs: list[CocoStuffSample] = []
    valid_pairs: list[CocoStuffSample] = []
    ignored_only = 0
    for index, sample in enumerate(pairs, start=1):
        with Image.open(sample.image_path) as handle:
            image_size = handle.size
        with Image.open(sample.label_path) as handle:
            raw = np.asarray(handle, dtype=np.uint8)
        if image_size != (raw.shape[1], raw.shape[0]):
            raise ValueError(f"RGB/mask resolution mismatch for {sample.key}")
        mapped = COCO_CAFE41_REMAP[raw]
        target = destination / sample.label_path.name
        if target.exists():
            with Image.open(target) as handle:
                existing = np.asarray(handle, dtype=np.uint8)
            if existing.shape != mapped.shape or not np.array_equal(existing, mapped):
                raise ValueError(f"Existing derived label differs from locked mapping: {target}")
        else:
            temporary = target.with_name(f".{target.name}.tmp")
            Image.fromarray(mapped).save(temporary, format="PNG")
            os.replace(temporary, target)
        derived = CocoStuffSample(sample.key, sample.image_path, target)
        all_derived_pairs.append(derived)
        valid = mapped != 255
        if valid.any():
            histogram += np.bincount(mapped[valid], minlength=len(histogram))[:len(histogram)]
            valid_pairs.append(derived)
        else:
            ignored_only += 1
        if index % 1000 == 0:
            print(json.dumps({"event": "remapped", "split": split, "images": index, "total": len(pairs)}), flush=True)
    return all_derived_pairs, valid_pairs, histogram, ignored_only


def manifest_record(root: Path, split: str, samples: list[CocoStuffSample], *, kind: str, seed: int | None = None) -> dict:
    return {
        "format": "coco_stuff_cafe41_v1",
        "kind": kind,
        "split": split,
        "seed": seed,
        "root": str(root),
        "class_names": list(COCO_CAFE41_NAMES),
        "samples": [sample.record() for sample in samples],
    }


def prepare_split(root: Path, split: str, seed: int) -> None:
    image_archive = root / "archives" / f"{split}.zip"
    label_archive = root / "archives" / "stuffthingmaps_trainval2017.zip"
    extract_members(image_archive, root / "images" / split, split=split, suffix=".jpg")
    extract_members(label_archive, root / "labels_raw" / split, split=split, suffix=".png")
    pairs = discover_pairs(root, split)
    all_derived_pairs, valid_pairs, histogram, ignored_only = create_derived_labels(root, split, pairs)
    manifests = root / "manifests"
    if split == "train2017":
        write_json(manifests / "train2017_cafe41.json", manifest_record(root, split, valid_pairs, kind="train_selected"))
    else:
        if len(pairs) != 5000:
            raise ValueError(f"Expected the official 5,000 COCO val images, found {len(pairs)}")
        ordered = list(all_derived_pairs)
        random.Random(seed).shuffle(ordered)
        dev, audit = ordered[:2500], ordered[2500:]
        write_json(manifests / "val2017_source_dev.json", manifest_record(root, split, dev, kind="source_dev", seed=seed))
        write_json(manifests / "val2017_source_audit.json", manifest_record(root, split, audit, kind="source_audit", seed=seed))
    write_json(
        root / "metadata" / f"{split}_cafe41_audit.json",
        {
            "format": "coco_stuff_cafe41_audit_v1",
            "split": split,
            "all_paired_images": len(pairs),
            "selected_images": len(valid_pairs),
            "all_ignore_images": ignored_only,
            "selected_pixel_counts": {name: int(count) for name, count in zip(COCO_CAFE41_NAMES, histogram)},
            "raw_ids": list(COCO_CAFE41_RAW_IDS),
        },
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if root.name.casefold() != "cocostuff2017":
        raise ValueError("Use the locked COCOStuff2017 dataset root.")
    validate_cafe41_mapping()
    write_json(
        root / "metadata" / "cafe41_mapping.json",
        {
            "format": "coco_stuff_cafe41_mapping_v1",
            "raw_ids": list(COCO_CAFE41_RAW_IDS),
            "class_names": list(COCO_CAFE41_NAMES),
            "png_value_0": "person and ignored by the CAFe 41-class source view",
            "png_value_255": "void and ignored",
        },
    )
    splits = ("val2017", "train2017") if args.split == "all" else (f"{args.split}2017",)
    for split in splits:
        prepare_split(root, split, args.seed)
    print(json.dumps({"status": "complete", "root": str(root), "splits": list(splits)}), flush=True)


if __name__ == "__main__":
    main()
