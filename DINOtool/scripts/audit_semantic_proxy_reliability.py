#!/usr/bin/env python3
"""Audit saved alias diagnostics without image inference or vocabulary selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def auc(scores: list[float], labels: list[bool]) -> tuple[float | None, int]:
    positive = [s for s, y in zip(scores, labels) if y]
    negative = [s for s, y in zip(scores, labels) if not y]
    pairs = len(positive) * len(negative)
    if not pairs:
        return None, 0
    wins = sum(float(a > b) + 0.5 * float(a == b)
               for a in positive for b in negative)
    return wins / pairs, pairs


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def audit(payload: dict) -> dict:
    if payload.get("status") != "complete":
        raise ValueError("Only complete saved diagnostic files are supported.")
    aliases = payload["aliases"]
    if not aliases:
        raise ValueError("No alias diagnostics.")
    groups = defaultdict(list)
    for index, row in enumerate(aliases):
        groups[row["class"]].append(index)
        if not all(math.isfinite(float(row[key])) for key in (
                "mean_logprob_gain", "native_counterfactual_gain", "flip_balance")):
            raise ValueError("Nonfinite diagnostic value.")
        if row["beneficial_flips"] - row["harmful_flips"] != row["flip_balance"]:
            raise ValueError("Inclusion flip direction does not match the producer.")

    # The historical low15 rule ranks the INVERSE native expectation higher.
    proxy = [-float(row["native_counterfactual_gain"]) for row in aliases]
    likelihood = [float(row["mean_logprob_gain"]) for row in aliases]
    flips = [float(row["flip_balance"]) for row in aliases]

    def target_report(target: list[float]) -> dict:
        eligible = [i for i, value in enumerate(target) if value != 0]
        score = [proxy[i] for i in eligible]
        label = [target[i] > 0 for i in eligible]
        overall, pair_count = auc(score, label)
        classes, weighted_wins, within_pairs = {}, 0.0, 0
        for name, indices in groups.items():
            members = [i for i in indices if target[i] != 0]
            value, pairs = auc([proxy[i] for i in members],
                               [target[i] > 0 for i in members])
            classes[name] = {"positive": sum(target[i] > 0 for i in indices),
                             "negative": sum(target[i] < 0 for i in indices),
                             "neutral": sum(target[i] == 0 for i in indices),
                             "inverse_native_auc": value}
            if pairs:
                weighted_wins += value * pairs
                within_pairs += pairs
        comparable = [i for i in eligible if proxy[i] != 0]
        return {"positive": sum(value > 0 for value in target),
                "negative": sum(value < 0 for value in target),
                "neutral": sum(value == 0 for value in target),
                "inverse_native_auc": overall,
                "overall_comparable_pairs": pair_count,
                "within_class_pair_weighted_auc": (
                    weighted_wins / within_pairs if within_pairs else None),
                "within_class_comparable_pairs": within_pairs,
                "natural_zero_sign_agreement": (
                    sum(sign(proxy[i]) == sign(target[i]) for i in comparable)
                    / len(comparable) if comparable else None),
                "natural_zero_comparable_aliases": len(comparable),
                "classes": classes}

    conflict = [row for row in aliases
                if row["mean_logprob_gain"] > 0 and row["flip_balance"] < 0]
    comparable = [i for i in range(len(aliases))
                  if likelihood[i] != 0 and flips[i] != 0]
    return {"dataset": payload["dataset"], "images": payload["images"],
            "tiles": payload["tiles"], "alias_count": len(aliases),
            "likelihood_retention": target_report(likelihood),
            "correct_pixel_retention": target_report(flips),
            "likelihood_vs_flips_sign_agreement": (
                sum(sign(likelihood[i]) == sign(flips[i]) for i in comparable)
                / len(comparable) if comparable else None),
            "likelihood_vs_flips_comparable_aliases": len(comparable),
            "positive_likelihood_negative_flips_count": len(conflict),
            "positive_likelihood_negative_flips_examples": [
                {key: row[key] for key in ("class", "alias", "mean_logprob_gain",
                                           "flip_balance", "native_counterfactual_gain")}
                for row in sorted(conflict, key=lambda r: r["flip_balance"])[:8]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError(f"Refusing to overwrite audit output: {args.output}")
    result = {"status": "complete",
              "note": ("Saved labeled audits only; no image inference, parameter fitting, "
                       "threshold search, or vocabulary output. Retention likelihood and "
                       "flip signs follow diagnose_gear_alias_marginals.py; neither is mIoU. "
                       "The natural-zero sign test is not the historical rank15 selection."),
              "datasets": [{"source": path.as_posix(),
                            **audit(json.loads(path.read_text(encoding="utf-8-sig")))}
                           for path in args.inputs]}
    encoded = json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
