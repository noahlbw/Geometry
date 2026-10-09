#!/usr/bin/env python3
"""Render selected LoveDA P-protocol comparisons for parallel DINO readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dinotool.loveda import discover_loveda_samples

METHODS = ("N_native", "G_geometry", "Joint_log")
COLORS = np.asarray(((228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),(255,255,51)), dtype=np.uint8)
EVIDENCE = {"other":(232,232,232),"keep":(245,158,11),"take":(16,185,129),"miss":(220,38,38),"lose":(126,34,206)}


def target(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path)).copy()
    if raw.ndim == 3:
        raw = raw[..., 0]
    result = np.full(raw.shape, 255, dtype=np.uint8)
    valid = (raw >= 2) & (raw <= 7)
    result[valid] = raw[valid] - 2
    return result


def prediction(root: Path, method: str, key: str, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(root / "predictions" / "P" / method / key)).copy()
    if value.shape != shape or value.ndim != 2:
        raise ValueError(f"Invalid prediction: {method}/{key}")
    return value


def miou(value: np.ndarray, truth: np.ndarray) -> float:
    valid, scores = truth != 255, []
    for label in range(6):
        inter = np.count_nonzero(valid & (truth == label) & (value == label))
        union = np.count_nonzero(valid & ((truth == label) | (value == label)))
        scores.append(inter / union if union else np.nan)
    return float(np.nanmean(scores))


def metrics(preds: dict[str, np.ndarray], truth: np.ndarray) -> dict[str, float | int]:
    native, geo, joint = (preds[item] for item in METHODS)
    valid = truth != 255
    n_ok, g_ok, j_ok = native == truth, geo == truth, joint == truth
    n, g, j = (miou(item, truth) for item in (native, geo, joint))
    return {"n":round(n*100,3),"g":round(g*100,3),"j":round(j*100,3),"adv":g-max(n,j),"valid":int(valid.sum()),
            "keep":int((valid & n_ok & ~g_ok & j_ok).sum()),"take":int((valid & ~n_ok & g_ok & j_ok).sum()),
            "miss":int((valid & ~n_ok & g_ok & ~j_ok).sum()),"lose":int((valid & n_ok & ~g_ok & ~j_ok).sum())}


def labels(value: np.ndarray) -> Image.Image:
    output = np.full((*value.shape,3), 32, dtype=np.uint8)
    valid = value < 6
    output[valid] = COLORS[value[valid]]
    return Image.fromarray(output, "RGB")


def evidence(preds: dict[str, np.ndarray], truth: np.ndarray) -> Image.Image:
    native, geo, joint = (preds[item] for item in METHODS)
    valid = truth != 255
    output = np.full((*truth.shape,3), EVIDENCE["other"], dtype=np.uint8)
    output[~valid] = (42,42,42)
    n_ok, g_ok, j_ok = native == truth, geo == truth, joint == truth
    output[valid & n_ok & ~g_ok & j_ok] = EVIDENCE["keep"]
    output[valid & ~n_ok & g_ok & j_ok] = EVIDENCE["take"]
    output[valid & ~n_ok & g_ok & ~j_ok] = EVIDENCE["miss"]
    output[valid & n_ok & ~g_ok & ~j_ok] = EVIDENCE["lose"]
    return Image.fromarray(output, "RGB")


def panel(image: Image.Image, side: int, nearest: bool = False) -> Image.Image:
    return image.resize((side, side), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)


def choose(records: list[dict[str, object]]) -> list[dict[str, object]]:
    first = max(records, key=lambda item: item["stats"]["adv"])
    rest = [item for item in records if item["key"] != first["key"]]
    second = max(rest, key=lambda item: item["stats"]["keep"])
    rest = [item for item in rest if item["key"] != second["key"]]
    third = max(rest, key=lambda item: item["stats"]["miss"])
    for item, reason in zip((first, second, third), ("geometry wins", "joint keeps native", "joint misses geometry")):
        item["reason"] = reason
    return [first, second, third]


def render(selected: list[dict[str, object]], path: Path, side: int) -> None:
    titles = ("RGB", "GT P", "Native", "G geometry", "Joint log", "Joint evidence")
    header, caption = 30, 36
    sheet = Image.new("RGB", (side * len(titles), header + len(selected) * (side + caption)), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for column, title in enumerate(titles):
        draw.text((column * side + 5, 10), title, fill="black", font=font)
    for row, item in enumerate(selected):
        top = header + row * (side + caption)
        predictions, truth, result = item["predictions"], item["target"], item["stats"]
        panels = (
            panel(Image.open(item["image_path"]).convert("RGB"), side),
            panel(labels(truth), side, True),
            panel(labels(predictions["N_native"]), side, True),
            panel(labels(predictions["G_geometry"]), side, True),
            panel(labels(predictions["Joint_log"]), side, True),
            panel(evidence(predictions, truth), side, True),
        )
        for column, value in enumerate(panels):
            sheet.paste(value, (column * side, top))
        text = (f"{item['reason']}: {item['key']}  N/G/J {result['n']:.1f}/{result['g']:.1f}/{result['j']:.1f}  "
                f"amber keep-N {result['keep']:,}; green take-G {result['take']:,}; red miss-G {result['miss']:,}; purple lose-N {result['lose']:,}")
        draw.text((5, top + side + 10), text, fill="black", font=font)
    sheet.save(path, "JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=208)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    args = parser.parse_args()
    root = Path(args.prediction_root)
    records = []
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    for sample in samples[: args.max_images]:
        truth = target(sample.mask_path)
        preds = {method: prediction(root, method, sample.key, truth.shape) for method in METHODS}
        records.append({"key":sample.key,"image_path":sample.image_path,"target":truth,"predictions":preds,"stats":metrics(preds, truth)})
    selected = choose(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    render(selected, output / "parallel_readout_examples.jpg", args.tile_size)
    aggregate = {name:sum(item["stats"][name] for item in records) for name in ("valid","keep","take","miss","lose")}
    payload = {"protocol":"LoveDA P: ground-truth background pixels excluded",
               "evidence_colors":{"amber":"native correct / geometry wrong / joint correct","green":"native wrong / geometry correct / joint correct","red":"native wrong / geometry correct / joint wrong","purple":"native correct / geometry wrong / joint wrong","gray":"other valid pixels"},
               "aggregate":aggregate,
               "selected":[{"reason":item["reason"],"key":item["key"],"stats":item["stats"]} for item in selected]}
    (output / "parallel_readout_examples.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
