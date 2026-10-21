#!/usr/bin/env python3
"""Merge VIP's official aliases with the existing Qwen-20 candidate pools.

The resulting pool is intended as the input to the VIP VG/SC selector.  VIP
aliases are always kept.  For foreground classes, Qwen aliases are appended
until the pool contains ``target_size`` distinct phrases; classes whose VIP
bank already contains at least that many phrases are kept intact.  VOC and
COCO background aliases are a special case in VIP: the complete official
background bank is retained and Qwen background phrases are not appended.

This script only constructs candidate pools; it never calls a model or uses
annotations.  It also writes a manifest with per-class provenance so that a
selector run can be audited later.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATASETS = {
    "voc21": {
        "candidate": "voc21.json",
        "official": "cls_voc21.txt",
        "background_name": "background",
        "preserve_background": True,
    },
    "coco_object81": {
        "candidate": "coco_object81.json",
        "official": "cls_coco_object.txt",
        "background_name": "background",
        "preserve_background": True,
    },
    "ade150": {
        "candidate": "ade150.json",
        "official": "cls_ade20k.txt",
        # ADE's 150-class protocol has no explicit background class.
        "background_name": None,
        "preserve_background": False,
    },
}


def _key(value: str) -> str:
    """Comparison key used for duplicate aliases.

    Keep the original spelling in the output, but remove trivial differences
    in case and whitespace when deciding whether two phrases are the same.
    """

    return re.sub(r"\s+", " ", value.strip()).casefold()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("classes"), list):
        raise ValueError(f"Expected a JSON bank with a classes list: {path}")
    return value


def _read_official_lines(path: Path, class_names: list[str]) -> dict[str, list[str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if len(lines) != len(class_names):
        raise ValueError(
            f"Official file {path} has {len(lines)} non-empty rows, "
            f"but the candidate pool has {len(class_names)} classes"
        )
    groups: dict[str, list[str]] = {}
    for name, line in zip(class_names, lines):
        # VIP uses comma-separated aliases; preserve their order exactly.
        aliases = [part.strip() for part in line.split(",") if part.strip()]
        groups[name] = aliases
    return groups


def _merge_class(
    *,
    name: str,
    qwen_aliases: list[str],
    official_aliases: list[str],
    target_size: int,
    preserve_background: bool,
) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    seen: set[str] = set()
    official_kept: list[str] = []
    qwen_added: list[str] = []

    def append(alias: str, destination: list[str]) -> None:
        alias = alias.strip()
        if not alias:
            return
        key = _key(alias)
        if key in seen:
            return
        seen.add(key)
        selected.append(alias)
        destination.append(alias)

    for alias in official_aliases:
        append(alias, official_kept)

    # The official VOC/COCO background is deliberately complete.  It is a
    # hand-built negative vocabulary in VIP, not a 20-candidate foreground
    # group, so truncating or replacing it would change the baseline.
    if not preserve_background and len(selected) < target_size:
        for alias in qwen_aliases:
            if len(selected) >= target_size:
                break
            append(alias, qwen_added)

    meta = {
        "name": name,
        "official_count": len(official_kept),
        "qwen_candidate_count": len(qwen_aliases),
        "qwen_added_count": len(qwen_added),
        "total_count": len(selected),
        "official_aliases": official_kept,
        "qwen_added": qwen_added,
        "target_size": target_size,
        "background_preserved": preserve_background,
    }
    return selected, meta


def merge_dataset(
    dataset: str,
    candidate_path: Path,
    official_path: Path,
    output_path: Path,
    target_size: int,
) -> dict[str, Any]:
    qwen_bank = _read_json(candidate_path)
    qwen_classes = qwen_bank["classes"]
    names: list[str] = []
    qwen_by_name: dict[str, list[str]] = {}
    for entry in qwen_classes:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"Malformed class entry in {candidate_path}: {entry!r}")
        name = entry["name"]
        aliases = entry.get("synonyms", entry.get("aliases"))
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"Malformed synonyms for {name!r} in {candidate_path}")
        names.append(name)
        qwen_by_name[name] = aliases

    official_by_name = _read_official_lines(official_path, names)
    preserve_background = DATASETS[dataset]["preserve_background"]
    background_name = DATASETS[dataset]["background_name"]

    classes: list[dict[str, Any]] = []
    class_manifest: list[dict[str, Any]] = []
    for name in names:
        selected, meta = _merge_class(
            name=name,
            qwen_aliases=qwen_by_name[name],
            official_aliases=official_by_name[name],
            target_size=target_size,
            preserve_background=(preserve_background and name == background_name),
        )
        classes.append({"name": name, "synonyms": selected})
        class_manifest.append(meta)

    payload = {
        "dataset": dataset,
        "source": "VIP official aliases + Qwen20 candidate pool",
        "selector_input": "VIP VG/SC candidate pool",
        "target_candidate_count": target_size,
        "background_rule": (
            "retain_complete official background for VOC21 and COCO-Object81"
            if preserve_background
            else "no explicit background class; apply foreground rule to all 150 ADE classes"
        ),
        "official_source": str(official_path),
        "qwen_source": str(candidate_path),
        "classes": classes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset": dataset,
        "candidate_pool": str(output_path),
        "official_source": str(official_path),
        "qwen_source": str(candidate_path),
        "class_count": len(classes),
        "official_alias_count": sum(m["official_count"] for m in class_manifest),
        "qwen_added_alias_count": sum(m["qwen_added_count"] for m in class_manifest),
        "total_alias_count": sum(m["total_count"] for m in class_manifest),
        "classes_with_qwen_additions": sum(m["qwen_added_count"] > 0 for m in class_manifest),
        "class_stats": class_manifest,
    }


def _default_official_dir(repo: Path) -> Path:
    # In this workspace the checked-out Geometry repo is under OVSS/artifacts,
    # while the VIP checkout is at OVSS/third_party/VIP_official.
    workspace_vip = repo.parent.parent / "third_party" / "VIP_official" / "configs"
    if workspace_vip.is_dir():
        return workspace_vip
    # The three source files are also tracked with the experiment, making the
    # script reproducible in a clean clone without the VIP checkout.
    return repo / "research" / "alias_selector_cross_model_natural_20261021" / "sources" / "vip_configs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--official-dir", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    candidate_dir = repo / "research" / "alias_selector_cross_model_natural_20261021" / "candidate_pools"
    output_dir = args.output_dir or (
        repo / "research" / "alias_selector_cross_model_natural_20261021" / "candidate_pools_official_plus_qwen20"
    )
    official_dir = (args.official_dir or _default_official_dir(repo)).resolve()
    output_dir = output_dir.resolve()

    manifest: dict[str, Any] = {
        "protocol": "official_plus_qwen20_candidate_pool_v1",
        "target_candidate_count": args.target_size,
        "inputs": {
            "candidate_dir": str(candidate_dir),
            "official_dir": str(official_dir),
        },
        "rule": {
            "foreground": "keep all official aliases, append distinct Qwen aliases until target size",
            "official_over_target": "keep all official aliases",
            "voc21_coco_object81_background": "retain complete official background and append no Qwen aliases",
            "ade150": "no explicit background; apply foreground rule to all classes",
        },
        "datasets": {},
    }
    for dataset, spec in DATASETS.items():
        stats = merge_dataset(
            dataset,
            candidate_dir / spec["candidate"],
            official_dir / spec["official"],
            output_dir / f"{dataset}.json",
            args.target_size,
        )
        manifest["datasets"][dataset] = stats

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "datasets": manifest["datasets"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
