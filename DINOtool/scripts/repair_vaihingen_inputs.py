#!/usr/bin/env python3
"""Rebuild the existing fixed Vaihingen validation tiles from source bands."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PALETTE = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2,
           (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 255,
           (0, 0, 0): 255}


def offsets(length, size=1000):
    if length < size:
        raise ValueError("Source is smaller than the historical tile size.")
    result = list(range(0, length-size+1, size))
    if result[-1]+size < length:
        result.append(length-size)
    return result


def remap_label(rgb):
    target = np.full(rgb.shape[:2], 255, np.uint8)
    known = np.zeros(rgb.shape[:2], bool)
    for color, label in PALETTE.items():
        mask = (rgb == color).all(-1)
        target[mask] = label
        known |= mask
    if not known.all():
        raise ValueError("Unknown source label colors; do not infer mapping from metrics.")
    return target


def source_crop(source, tile_name):
    match = re.fullmatch(r"(.+)_y(\d+)_x(\d+)\.tif", tile_name)
    if match is None or source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("Expected named tile and uint8 three-band source.")
    y, x = int(match[2]), int(match[3])
    yy, xx = offsets(source.shape[0]), offsets(source.shape[1])
    return source[yy[y]:yy[y]+1000, xx[x]:xx[x]+1000].copy()


def main(args):
    import tifffile

    root, old, output = map(Path, (args.source_root, args.old_prepared, args.output))
    if output.exists():
        raise ValueError("Preserve old and existing repaired outputs.")
    files = sorted((old/"val/images").glob("*.tif"))
    if len(files) != 113:
        raise ValueError(f"Expected 113 historical validation tiles, found {len(files)}.")
    (output/"val/images").mkdir(parents=True)
    (output/"val/labels").mkdir()
    cache, rows = {}, []
    for image in files:
        stem = re.fullmatch(r"(.+)_y\d+_x\d+\.tif", image.name)[1]
        if stem not in cache:
            cache = {stem: (tifffile.imread(root/"top"/(stem+".tif")),
                            tifffile.imread(root/"labels_raw"/(stem+".tif")))}
        source_image, source_label = cache[stem]
        rgb = source_crop(source_image, image.name)
        label = remap_label(source_crop(source_label, image.name))
        previous_label = tifffile.imread(old/"val/labels"/image.name)
        if not np.array_equal(label, previous_label):
            raise ValueError(f"Historical mask differs from declared palette: {image.name}")
        tifffile.imwrite(output/"val/images"/image.name, rgb, photometric="rgb")
        tifffile.imwrite(output/"val/labels"/image.name, label, photometric="minisblack")
        if not np.array_equal(tifffile.imread(output/"val/images"/image.name), rgb):
            raise ValueError("Saved imagery does not reproduce source crop.")
        previous_image = tifffile.imread(image)
        rows.append({"key": image.name, "old_shape": list(previous_image.shape),
                     "old_black": bool(previous_image.max() == 0), "new_shape": list(rgb.shape),
                     "source_crop_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                     "label_unchanged": True, "source_bands_unchanged": True})
    report = {"status": "complete", "images": len(rows), "old_black_images": sum(r["old_black"] for r in rows),
              "source_root": str(root), "old_prepared": str(old), "output": str(output),
              "bands": "Original three bands, unchanged (Vaihingen IRRG; not natural RGB).",
              "protocol": "Same 113 keys, same 1000px edge-overlap offsets, same five-class labels; clutter ignored.",
              "defect": "Upstream split_tiff remapped image colors to labels regardless of is_label.",
              "samples": rows}
    (output/"input_repair_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "old-prepared", "output"):
        parser.add_argument("--"+name, required=True)
    main(parser.parse_args())
