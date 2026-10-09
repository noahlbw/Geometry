#!/usr/bin/env python3
"""Verify sample pairing, the fixed taxonomy and one real image/mask shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dinotool.gear_datasets import class_names, discover_samples, load_rgb, load_target
from dinotool.prompts import load_class_specs


def audit(dataset: str, root: Path, vocabulary: Path) -> dict:
    samples = discover_samples(dataset, root)
    specs = load_class_specs(vocabulary)
    if tuple(spec.name for spec in specs) != class_names(dataset):
        raise ValueError("Vocabulary order differs from official taxonomy.")
    if any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("Expected exactly 20 aliases per class.")
    image = load_rgb(samples[0], dataset)
    mask = load_target(samples[0], dataset, tuple(image.shape[-2:]))
    result = {"dataset": dataset, "paired_images": len(samples),
              "first_sample": samples[0].key, "image_shape": list(image.shape),
              "rgb_range": [float(image.min()), float(image.max())],
              "first_mask_mapped_ids": np.unique(mask).tolist(),
              "class_names": list(class_names(dataset)), "aliases_per_class": 20}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("flair1", "landcoverai"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vocabulary-config", type=Path, required=True)
    args = parser.parse_args()
    audit(args.dataset, args.data_root, args.vocabulary_config)
