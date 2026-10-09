#!/usr/bin/env python3
"""Reject Q-Lift source checkpoints that do not satisfy the frozen v3 gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_ARGS: dict[str, Any] = {
    "updates": 20896,
    "validate_every": 2612,
    "checkpoint_every": 500,
    "warmup_steps": 500,
    "stop_after_validation": 2612,
    "batch_size": 2,
    "accum_steps": 4,
    "workers": 4,
    "new_lr": 1e-4,
    "head_lr": 2e-5,
    "visual_lr": 2e-6,
    "weight_decay": 0.01,
    "kd_weight": 0.02,
    "label_smoothing": 0.1,
    "seed": 20260921,
    "amp": "bf16",
    "memory_fraction": 0.75,
    "qlift_levels": 2,
    "qlift_guide_dim": 128,
    "qlift_hidden_dim": 256,
    "qlift_kernel": 3,
    "qlift_query_chunk": 8,
    "qlift_tune_visual_blocks": 0,
}
EXPECTED_WORLD_SIZE = 4
EXPECTED_STEP = 2612
EXPECTED_TRAINABLE_PARAMETERS = 3_895_181


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("skip", "haar", "image_lift", "query_lift"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def last_training_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training = [record for record in records if record.get("status") == "training"]
    if not training:
        raise ValueError(f"No training LR record in {path}")
    return training[-1]


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def require_close(label: str, actual: Any, expected: Any, *, tolerance: float = 1e-10) -> None:
    """Require numerically identical source-selection evidence.

    The corrected gate stops at its first validation, so its recorded best score,
    the current validation score, and the checkpoint's score must all be the same.
    This check prevents a valid-looking checkpoint file from being paired with a
    different selection record.
    """

    try:
        difference = abs(float(actual) - float(expected))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {actual!r}, {expected!r}") from error
    if difference > tolerance:
        raise ValueError(f"{label} differs by {difference}: {actual!r} != {expected!r}")


def verify(run_root: Path, arm: str) -> dict[str, Any]:
    run_root = run_root.resolve()
    if (run_root / "failure.json").exists():
        raise RuntimeError(f"Source run failed: {load_json(run_root / 'failure.json')}")
    status = load_json(run_root / "status.json")
    training = load_json(run_root / "training_config.json")
    manifest = run_root / "source_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    require_equal("status.status", status.get("status"), "paused_at_validation")
    require_equal("status.step", status.get("step"), EXPECTED_STEP)
    require_equal("status.total_updates", status.get("total_updates"), EXPECTED_ARGS["updates"])
    require_equal("status.world_size", status.get("world_size"), EXPECTED_WORLD_SIZE)

    args = training.get("args")
    architecture = training.get("architecture")
    if not isinstance(args, dict) or not isinstance(architecture, dict):
        raise ValueError("training_config lacks args/architecture mappings")
    require_equal("training.world_size", training.get("world_size"), EXPECTED_WORLD_SIZE)
    # A run may keep the same apparent LR schedule while importing model,
    # optimizer, or RNG state from the obsolete compressed-schedule gate.
    # The corrected source comparison is deliberately fresh-start only.
    if "resume" not in args:
        raise ValueError("args.resume is missing; a fresh-start corrected run must record resume=null")
    require_equal("args.resume", args["resume"], None)
    for key, expected in EXPECTED_ARGS.items():
        require_equal(f"args.{key}", args.get(key), expected)
    require_equal("architecture.name", architecture.get("name"), "Q-Lift-DINO-v1")
    require_equal("architecture.arm", architecture.get("arm"), arm)
    require_equal("architecture.tune_visual_blocks", architecture.get("tune_visual_blocks"), 0)
    require_equal("architecture.tuned_visual_indices", architecture.get("tuned_visual_indices"), [])
    require_equal("architecture.trainable_parameters", architecture.get("trainable_parameters"), EXPECTED_TRAINABLE_PARAMETERS)

    validation = status.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("status lacks source validation")
    require_equal("validation.validation_step", validation.get("validation_step"), EXPECTED_STEP)
    coco, oem = validation.get("coco"), validation.get("oem")
    if not isinstance(coco, dict) or not isinstance(oem, dict):
        raise ValueError("source validation lacks COCO/OEM results")
    require_equal("validation.coco.images", coco.get("images"), 2500)
    require_equal("validation.oem.images", oem.get("images"), 384)
    expected_selection = 0.5 * (float(coco["mean_iou"]) + float(oem["mean_iou"]))
    if abs(float(validation["selection_score"]) - expected_selection) > 1e-10:
        raise ValueError("validation.selection_score is not mean(COCO, OEM)")
    if "best_source_score" not in status:
        raise ValueError("status lacks best_source_score")
    require_close(
        "status.best_source_score vs validation.selection_score",
        status["best_source_score"],
        validation["selection_score"],
    )

    checkpoint_path = run_root / "best_inference.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("selected checkpoint is not a mapping")
    require_equal("checkpoint.format", checkpoint.get("format"), "cafe_qlift_v1")
    checkpoint_architecture = checkpoint.get("architecture")
    checkpoint_validation = checkpoint.get("validation")
    if not isinstance(checkpoint_architecture, dict) or not isinstance(checkpoint_validation, dict):
        raise ValueError("selected checkpoint lacks architecture/validation")
    require_equal("checkpoint.architecture.arm", checkpoint_architecture.get("arm"), arm)
    require_equal("checkpoint.step", checkpoint.get("step"), EXPECTED_STEP)
    require_equal("checkpoint.validation.validation_step", checkpoint_validation.get("validation_step"), EXPECTED_STEP)
    require_equal("checkpoint.validation", checkpoint_validation, validation)
    if "best_source_score" not in checkpoint:
        raise ValueError("selected checkpoint lacks best_source_score")
    require_close(
        "checkpoint.best_source_score vs validation.selection_score",
        checkpoint["best_source_score"],
        validation["selection_score"],
    )
    require_close(
        "checkpoint.best_source_score vs status.best_source_score",
        checkpoint["best_source_score"],
        status["best_source_score"],
    )

    last_train = last_training_record(run_root / "steps.jsonl")
    require_equal("last_training.total_updates", last_train.get("total_updates"), EXPECTED_ARGS["updates"])
    if int(last_train.get("step", 0)) < 2600:
        raise ValueError(f"Last logged training step is unexpectedly early: {last_train.get('step')}")
    lrs = last_train.get("learning_rates")
    if not isinstance(lrs, dict):
        raise ValueError("Last training record lacks learning_rates")
    # At this gate the full 20,896-step cosine remains close to the initial LR.
    # This rejects the old 2,612-step run, whose LR had decayed to the floor.
    if not (9e-5 <= float(lrs.get("new", 0.0)) <= EXPECTED_ARGS["new_lr"]):
        raise ValueError(f"Unexpected new LR at gate: {lrs.get('new')}")
    if not (1.8e-5 <= float(lrs.get("head", 0.0)) <= EXPECTED_ARGS["head_lr"]):
        raise ValueError(f"Unexpected head LR at gate: {lrs.get('head')}")

    return {
        "status": "contract_valid",
        "arm": arm,
        "run_root": str(run_root),
        "selection_score": float(validation["selection_score"]),
        "status_best_source_score": float(status["best_source_score"]),
        "checkpoint_best_source_score": float(checkpoint["best_source_score"]),
        "last_training_step": int(last_train["step"]),
        "last_learning_rates": {"new": float(lrs["new"]), "head": float(lrs["head"])},
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(verify(args.run_root, args.arm), sort_keys=True))


if __name__ == "__main__":
    main()
