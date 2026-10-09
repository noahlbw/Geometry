#!/usr/bin/env python3
"""Build exact UDD5 vocabulary controls without inspecting target masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all20", type=Path, required=True)
    parser.add_argument("--fixed15", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text())["classes"]


def main(args):
    original, fixed, dynamic = (load(args.all20), load(args.fixed15),
                                load(args.dynamic))
    names = [entry["name"] for entry in original]
    if names != [entry["name"] for entry in fixed] or names != [entry["name"] for entry in dynamic]:
        raise ValueError("Class order differs across vocabulary inputs")
    for base, a, b in zip(original, fixed, dynamic):
        full = base["synonyms"]
        if len(full) != 20 or len(set(full)) != 20:
            raise ValueError("Expected unique all-20 candidates")
        for selected in (a["synonyms"], b["synonyms"]):
            if len(selected) != len(set(selected)) or not set(selected).issubset(full):
                raise ValueError("Selected vocabulary is not an exact subset")
        if len(a["synonyms"]) != 15:
            raise ValueError("Expected fixed-15 aliases in every class")

    controls = {
        "road20_others15": [base if base["name"] == "road" else a
                             for base, a in zip(original, fixed)],
        "road15_others20": [a if base["name"] == "road" else base
                             for base, a in zip(original, fixed)],
        "dynamic": dynamic,
    }
    for seed in range(3):
        fixed_random = []
        matched_random = []
        for class_index, (base, selected) in enumerate(zip(original, dynamic)):
            aliases = base["synonyms"]
            rng = random.Random(20260929 + 1009 * seed + class_index)
            chosen15 = set(rng.sample(aliases, 15))
            rng_matched = random.Random(20261029 + 1009 * seed + class_index)
            chosen_matched = set(rng_matched.sample(aliases,
                                                   len(selected["synonyms"])))
            fixed_random.append({"name": base["name"],
                                 "synonyms": [item for item in aliases
                                              if item in chosen15]})
            matched_random.append({"name": base["name"],
                                   "synonyms": [item for item in aliases
                                                if item in chosen_matched]})
        controls[f"random15_s{seed}"] = fixed_random
        controls[f"randommatch_s{seed}"] = matched_random

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, classes in controls.items():
        path = args.output_dir / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(json.dumps({"classes": classes}, indent=2) + "\n")
        print(json.dumps({"arm": name,
                          "counts": [len(entry["synonyms"]) for entry in classes]}))


if __name__ == "__main__":
    main(parse_args())
