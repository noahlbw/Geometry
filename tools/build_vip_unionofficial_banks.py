#!/usr/bin/env python3
"""Build the official-preserving VIP screening banks.

The VIP VG/SC selector is allowed to reject an official phrase when its
response is weaker than the canonical phrase.  For this experiment the
official VIP vocabulary is an immutable anchor: only aliases introduced by
the Qwen expansion may be admitted or rejected.  This script unions each
screened group with the corresponding official group and writes an auditable
manifest.  It never reads images or masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DATASETS = ("voc21", "coco_object81", "ade150")


def key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def union_bank(dataset: str, official_path: Path, screened_path: Path, output: Path) -> dict:
    official = load(official_path)
    screened = load(screened_path)
    if len(official["classes"]) != len(screened["classes"]):
        raise ValueError(f"class-count mismatch for {dataset}")
    classes = []
    for off, scr in zip(official["classes"], screened["classes"]):
        if off["name"] != scr["name"]:
            raise ValueError(f"class-order mismatch: {off['name']} != {scr['name']}")
        aliases = []
        seen = set()
        for phrase in list(off["synonyms"]) + list(scr["synonyms"]):
            norm = key(phrase)
            if norm and norm not in seen:
                seen.add(norm)
                aliases.append(phrase)
        classes.append({"name": off["name"], "synonyms": aliases})
    bank = {
        "dataset": dataset,
        "classes": classes,
        "template": screened.get("template", official.get("template", "seg_template")),
        "background_rule": "official_fixed_for_voc_coco" if dataset in {"voc21", "coco_object81"} else "retained",
        "selector_family": "vip_vg_sc_official_union_qwen_screened",
        "retention_rule": "union_all_official_aliases_with_vip_screened_qwen_aliases",
        "sources": [str(official_path.resolve()), str(screened_path.resolve())],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset": dataset,
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "official_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
        "screened_sha256": hashlib.sha256(screened_path.read_bytes()).hexdigest(),
        "total_aliases": sum(len(c["synonyms"]) for c in classes),
        "per_class_counts": {c["name"]: len(c["synonyms"]) for c in classes},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research/alias_selector_cross_model_natural_20261021"))
    parser.add_argument("--official-dir", default="banks/six_controls")
    parser.add_argument("--screened-dir", default="vip_official_plus_qwen20_fixedbg")
    parser.add_argument("--output-dir", default="vip_official_plus_qwen20_unionofficial")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = {
        "protocol": "union_all_official_aliases_with_vip_screened_qwen_aliases_v1",
        "selection_uses_miou": False,
        "datasets": {},
    }
    for dataset in DATASETS:
        official = root / args.official_dir / dataset / "vip_official.json"
        screened = root / args.screened_dir / dataset / "vip_bank.json"
        output = root / args.output_dir / dataset / "vip_bank.json"
        manifest["datasets"][dataset] = union_bank(dataset, official, screened, output)
    out = root / args.output_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
