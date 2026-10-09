"""Label-only audit of saved supports; never changes observations or predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dinotool.grounded_local_observer import fractional_box_support
from dinotool.grounded_mask_observer import IMPLEMENTATION
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from run_grounded_mask_likelihood_suite import SETTINGS, SOURCE, TOOL


def audit(root, setting):
    dataset, data, vocab, _, _ = setting
    path = root/dataset
    destination = path/"support_audit.json"
    if destination.exists():
        raise ValueError("Refusing existing support audit.")
    result = json.loads((path/"verified.json").read_text())
    if result["status"] != "complete" or result["implementation"] != IMPLEMENTATION:
        raise ValueError("Complete verified candidate required.")
    args = argparse.Namespace(dataset=dataset, data_root=str(Path("/data/test/datasets")/data),
        sample_seed=20260923, vdd_ontology="official")
    samples, _, _, load_mask = protocol(args, load_class_specs(TOOL/"configs"/vocab))
    lookup = {sample.key: sample for sample in samples}
    banks = result["signature"]["source_signature"]["vocabularies"]
    totals = {key: {"names": bank["classes"], "box_correct_mass": np.zeros(len(bank["classes"])),
        "box_total_mass": np.zeros(len(bank["classes"])), "mask_correct_mass": np.zeros(len(bank["classes"])),
        "mask_total_mass": np.zeros(len(bank["classes"])), "qualified_query_count": np.zeros(len(bank["classes"]), np.int64)}
        for key, bank in banks.items()}
    last_key, target = None, None
    windows = json.loads((path/"window_manifest.json").read_text())
    for row in windows:
        key, bank = row["protocol"], banks[row["protocol"]]
        classes = len(bank["classes"])
        if last_key != (row["sample_key"], key):
            target = load_mask(lookup[row["sample_key"]], key, tuple(row["image_shape"]))
            last_key = row["sample_key"], key
        top, left = row["top"], row["left"]
        ah, aw = min(512, target.shape[0]-top), min(512, target.shape[1]-left)
        crop = np.full((512, 512), -1, np.int64)
        crop[:ah, :aw] = target[top:top+ah, left:left+aw]
        truth = np.stack([(crop == c).reshape(32, 16, 32, 16).mean((1, 3)).reshape(1024)
                          for c in range(classes)], -1)
        valid = truth.sum(-1)
        parents = np.repeat(np.arange(classes), bank["counts"])
        with np.load(SOURCE/dataset/row["file"], allow_pickle=False) as raw, \
                np.load(path/row["mask_file"], allow_pickle=False) as observed:
            for chunk in range(int(raw["chunks"])):
                scores = raw[f"alias_scores_{chunk}"]
                scores = np.where(scores > .25, scores, 0.)
                selected_parents = parents[raw[f"alias_indices_{chunk}"]]
                weights = np.stack([scores[:, selected_parents == c].sum(-1)/20 for c in range(classes)], -1)
                totals[key]["qualified_query_count"] += (weights > 0).sum(0)
                box = fractional_box_support(torch.from_numpy(raw[f"boxes_{chunk}"])).numpy()
                mask = observed[f"supports_{chunk}"]
                if mask.shape != box.shape or not np.isfinite(mask).all() or np.any(mask > box+1e-7):
                    raise ValueError("Invalid saved observed support.")
                for name, support in (("box", box), ("mask", mask)):
                    totals[key][name+"_correct_mass"] += ((support @ truth)*weights).sum(0)
                    totals[key][name+"_total_mass"] += ((support @ valid)[:, None]*weights).sum(0)
    summary = {}
    for key, values in totals.items():
        summary[key] = [{"class": name, "qualified_query_count": int(values["qualified_query_count"][c]),
            **{f"{kind}_{field}": float(values[f"{kind}_{field}"][c])
               for kind in ("box", "mask") for field in ("correct_mass", "total_mass")},
            **{f"{kind}_precision_percent": (100*float(values[f"{kind}_correct_mass"][c])/float(values[f"{kind}_total_mass"][c])
               if values[f"{kind}_total_mass"][c] > 0 else None) for kind in ("box", "mask")}}
            for c, name in enumerate(values["names"])]
    report = {"status": "complete", "implementation": IMPLEMENTATION, "dataset": dataset,
        "window_count": len(windows), "sample_keys": result["signature"]["sample_keys"],
        "readout_or_selection_changed": False, "per_class": summary,
        "definition": "Score-weighted summed support mass aligned with patch-averaged labels; all20 aliases. "
            "Not object AP, exact subpatch precision, or full-image IoU; repeated queries/windows may overlap. "
            "Mask precision can rise merely by deleting support, so correct mass is reported too."}
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"dataset": dataset, "per_class": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.resolve().relative_to((TOOL/"results").resolve())
    for setting in SETTINGS:
        audit(args.root, setting)
