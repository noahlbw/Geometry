#!/usr/bin/env python3
"""Read-only correlation audit of label-free alias proxies against GT marginals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def pearson(xs, ys):
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys))
    left = sum((x - xmean) ** 2 for x in xs)
    right = sum((y - ymean) ** 2 for y in ys)
    return numerator / (left * right) ** 0.5 if left and right else None


def auc(values, positive):
    yes = [value for value, label in zip(values, positive) if label]
    no = [value for value, label in zip(values, positive) if not label]
    if not yes or not no:
        return None
    wins = sum((x > y) + 0.5 * (x == y) for x in yes for y in no)
    return wins / (len(yes) * len(no))


def main(paths):
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        aliases = data["aliases"]
        objective = [item["mean_logprob_gain"] for item in aliases]
        positive = [value > 0 for value in objective]
        per_class = defaultdict(list)
        for index, item in enumerate(aliases):
            per_class[item["class"]].append(index)
        print(data["dataset"], "aliases", len(aliases), "positive", sum(positive))
        for field in ("native_counterfactual_gain", "mean_class_delta",
                      "unique_top_fraction", "old_selected"):
            raw = [float(item[field]) for item in aliases]
            centered = raw.copy()
            target = objective.copy()
            for indices in per_class.values():
                a = sum(raw[index] for index in indices) / len(indices)
                b = sum(objective[index] for index in indices) / len(indices)
                for index in indices:
                    centered[index] -= a
                    target[index] -= b
            print(field, "auc", round(auc(raw, positive), 4),
                  "pearson", round(pearson(raw, objective), 4),
                  "within_class_pearson", round(pearson(centered, target), 4))
        print("native_proxy_top", [
            (item["class"], item["alias"], round(item["mean_logprob_gain"] * 1000, 2))
            for item in sorted(aliases, key=lambda x: x["native_counterfactual_gain"],
                               reverse=True)[:12]
        ])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    main(parser.parse_args().results)
