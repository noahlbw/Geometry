"""COCO-Stuff's 171 active classes, separate from the locked CAFe-41 protocol."""
from pathlib import Path

import torch
import yaml

from .coco_stuff import CocoStuffCafe41Dataset, CocoStuffSample, samples_from_manifest

# COCO's 91 thing-ID slots contain eleven categories without annotations.
# Raw PNG indices are one less than the corresponding COCO category IDs.
UNUSED_THING_IDS = (11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90)
ACTIVE_IDS = tuple(i for i in range(182) if i not in UNUSED_THING_IDS)


def class_names(official_root):
    config = Path(official_root) / "CAFe_DINO/configs/config_cocostuff.yaml"
    names = yaml.safe_load(config.read_text())["class_names"][0]
    if len(names) != 182:
        raise ValueError("Expected the pinned 182-slot COCO class dictionary")
    active = tuple(names[i] for i in ACTIVE_IDS)
    if len(active) != 171 or len(set(active)) != 171:
        raise ValueError("The 171 active COCO names must be unique")
    if names[0] != "person" or names[156] != "sky" or names[170] != "brick wall":
        raise ValueError("COCO raw-label/name indexing changed")
    return active


def raw_samples(root, manifest):
    root = Path(root)
    records = samples_from_manifest(root / "manifests" / manifest)
    samples = []
    for sample in records:
        split = "train2017" if "train2017" in manifest else "val2017"
        image = root / "images" / split / sample.image_path.name
        mask = root / "labels_raw" / split / sample.label_path.name
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing raw COCO pair: {sample.key}")
        samples.append(CocoStuffSample(sample.key, image, mask))
    return samples


def remap_labels(target):
    lut = torch.full((256,), 255, dtype=torch.long, device=target.device)
    lut[list(ACTIVE_IDS)] = torch.arange(171, device=target.device)
    if ((target < 0) | (target > 255)).any():
        raise ValueError("COCO masks must be byte IDs")
    result = lut[target.long()]
    if ((target != 255) & (result == 255)).any():
        raise ValueError("COCO mask contains inactive or unknown raw IDs")
    return result


class CocoStuffFullDataset(CocoStuffCafe41Dataset):
    def __getitem__(self, index):
        rgb, raw = super().__getitem__(index)
        return rgb, remap_labels(raw)
