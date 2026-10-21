"""Materialize the six natural-image vocabulary controls used in the alias study.

The script only rearranges frozen local JSON files.  It never runs a model or
uses masks.  All banks use the candidate-pool class order so that the evaluator
can compare them with one class index mapping.  For VOC21/COCO-Object81 the
background group is fixed to the complete official VIP background line in every
bank that advertises ``fixed_official_background``; ADE150 has no scored
background class, so its first class remains a foreground class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# The checked-out GitHub repository keeps research/ at its root.  The local
# experiment workspace historically wrapped that repository in an
# ``artifacts/geometry_github`` directory, so accept both layouts.
if (ROOT / "research/alias_selector_cross_model_natural_20261021").exists():
    SOURCE = ROOT / "research/alias_selector_cross_model_natural_20261021"
else:
    SOURCE = ROOT / "artifacts/geometry_github/research/alias_selector_cross_model_natural_20261021"
official_candidates = [
    ROOT / "third_party/VIP_official/configs",
    SOURCE / "sources/vip_configs",
    ROOT.parent.parent / "third_party/VIP_official/configs",
]
OFFICIAL = next((p for p in official_candidates if p.exists()), official_candidates[0])
OUT_DEFAULT = SOURCE / "banks/six_controls"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def official_groups(dataset: str) -> list[list[str]]:
    fn = {"voc21": "cls_voc21.txt", "coco_object81": "cls_coco_object.txt", "ade150": "cls_ade20k.txt"}[dataset]
    return [[x.strip() for x in line.split(",") if x.strip()]
            for line in (OFFICIAL / fn).read_text(encoding="utf-8").splitlines()]


def official_template(dataset: str) -> str:
    return "ImageNet80" if dataset == "coco_object81" else "seg_template"


def normalize_classes(rows: list[dict[str, Any]], groups: list[list[str]], *,
                      fixed_background: bool, dataset: str,
                      first_only: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        aliases = list(dict.fromkeys(str(x) for x in row.get("synonyms", [])))
        if not aliases:
            raise ValueError(f"empty aliases for class {row.get('name')!r}")
        if first_only:
            # Candidate pools contain twenty Qwen expressions per class.  The
            # canonical-only and class-fixed-background controls must contain
            # exactly one foreground expression per class; otherwise they are
            # indistinguishable from the full expansion control.
            aliases = aliases[:1]
        if fixed_background and dataset in {"voc21", "coco_object81"} and i == 0:
            aliases = list(dict.fromkeys(groups[0]))
        out.append({"name": row["name"], "synonyms": aliases})
    return out


def write_bank(out: Path, dataset: str, name: str, classes: list[dict[str, Any]], *,
               template: str, background_rule: str, source_files: list[Path],
               description: str) -> dict[str, Any]:
    bank = {
        "dataset": dataset,
        "classes": classes,
        "template": template,
        "background_rule": background_rule,
        "selector_family": name,
        "description": description,
        "sources": [str(p.resolve()) for p in source_files],
    }
    path = out / f"{name}.json"
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "bank": name,
        "path": str(path),
        "template": template,
        "background_rule": background_rule,
        "classes": len(classes),
        "total_aliases": sum(len(c["synonyms"]) for c in classes),
        "per_class_counts": {c["name"]: len(c["synonyms"]) for c in classes},
        "description": description,
    }


def build(dataset: str, root: Path, out_root: Path) -> dict[str, Any]:
    pool_path = root / "candidate_pools" / f"{dataset}.json"
    pool = load(pool_path)
    pool_classes = pool["classes"]
    official = official_groups(dataset)
    if len(pool_classes) != len(official):
        raise ValueError((dataset, len(pool_classes), len(official)))
    rows = [{"name": c["name"], "synonyms": list(c["synonyms"])} for c in pool_classes]
    fixed_bg = dataset in {"voc21", "coco_object81"}
    template = official_template(dataset)
    out = out_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []

    # (1) Candidate-pool canonical only.  This intentionally leaves the
    # background as the literal one-word candidate, making it the strict
    # 1-expression-per-class control.
    result.append(write_bank(
        out, dataset, "canonical_only", normalize_classes(rows, official, fixed_background=False, dataset=dataset, first_only=True),
        template=template, background_rule="canonical_only", source_files=[pool_path],
        description="One candidate-pool canonical expression per class; strict 1-expression control."))

    # (2) Same class canonicals, with the official fixed background group.
    result.append(write_bank(
        out, dataset, "class_fixed_background", normalize_classes(rows, official, fixed_background=True, dataset=dataset, first_only=True),
        template=template, background_rule="fixed_official_background" if fixed_bg else "retained",
        source_files=[pool_path, OFFICIAL / {"voc21": "cls_voc21.txt", "coco_object81": "cls_coco_object.txt", "ade150": "cls_ade20k.txt"}[dataset]],
        description="One candidate-pool canonical per foreground class plus the official fixed background group."))

    # (3) Official VIP vocabulary (all expressions from the official file).
    official_rows = [{"name": rows[i]["name"], "synonyms": official[i]} for i in range(len(rows))]
    result.append(write_bank(
        out, dataset, "vip_official", normalize_classes(official_rows, official, fixed_background=False, dataset=dataset),
        template=template, background_rule="fixed_official_background" if fixed_bg else "retained",
        source_files=[OFFICIAL / {"voc21": "cls_voc21.txt", "coco_object81": "cls_coco_object.txt", "ade150": "cls_ade20k.txt"}[dataset]],
        description="Complete official VIP class groups, including its official background group."))

    # (4) Full 20-expression Qwen expansion.  Background is kept fixed for
    # VOC/COCO to avoid letting generic background candidates change the test.
    qwen20 = normalize_classes(rows, official, fixed_background=fixed_bg, dataset=dataset)
    result.append(write_bank(
        out, dataset, "vip20_expansion", qwen20,
        template=template, background_rule="fixed_official_background" if fixed_bg else "retained",
        source_files=[pool_path, OFFICIAL / {"voc21": "cls_voc21.txt", "coco_object81": "cls_coco_object.txt", "ade150": "cls_ade20k.txt"}[dataset]],
        description="All 20 Qwen candidates per class; fixed official background for VOC/COCO."))

    # (5,6) Frozen selector outputs.  They were built against the same pool;
    # replace only the background group for fair cross-selector comparison.
    for name, source_name, desc in [
        ("vip_screened", "vip_bank.json", "Frozen VIP VG/SC screened bank from the shared Qwen20 pool."),
        ("jev_screened", "jev_bank.json", "Frozen JEV semantic screened bank from the shared Qwen20 pool."),
    ]:
        selector_root = root / ("vip" if name.startswith("vip") else "jev") / dataset / source_name
        selected = load(selector_root)
        selected_rows = [{"name": c["name"], "synonyms": list(c["synonyms"])} for c in selected["classes"]]
        if len(selected_rows) != len(rows):
            raise ValueError((dataset, name, len(selected_rows), len(rows)))
        result.append(write_bank(
            out, dataset, name, normalize_classes(selected_rows, official, fixed_background=fixed_bg, dataset=dataset),
            template=template, background_rule="fixed_official_background" if fixed_bg else "retained",
            source_files=[selector_root, pool_path], description=desc + " Background standardized to official for VOC/COCO."))

    manifest = {
        "dataset": dataset,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "official_template": template,
        "official_background_fixed": fixed_bg,
        "banks": result,
        "protocol": {
            "canonical_only": "candidate pool first expression, including one-word background",
            "class_fixed_background": "candidate pool first foreground expression + official background",
            "vip_official": "complete VIP official groups",
            "vip20_expansion": "all 20 candidates/class + official background",
            "vip_screened": "frozen VIP VG/SC screen; official background",
            "jev_screened": "frozen JEV text screen; official background",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", default=["voc21", "coco_object81", "ade150"])
    p.add_argument("--root", type=Path, default=SOURCE)
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()
    rows = [build(ds, args.root, args.out_root) for ds in args.datasets]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
