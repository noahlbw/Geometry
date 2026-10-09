#!/usr/bin/env python3
"""Strict post-hoc spatial audit for a completed native Q-Lift LoveDA P/D run.

This is intentionally a separate executable from ``eval_cafe_vc_protocols``.
It never selects a checkpoint or changes inference settings: all semantic
inference settings are read from a completed native reference evaluation and
then checked before a fixed selected Q-Lift checkpoint is re-run.  Spatial
diagnostics are published only when the reconstructed global confusion matrix
is exactly equal to the reference ``results.json`` confusion matrix.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from cafedino_locked_loveda import build_model, image_tensor, target_ids
from dinotool.cafe_qlift import CafeQLift
from dinotool.loveda import discover_loveda_samples
from dinotool.ov_train import _write_json_atomic
from dinotool.qlift_spatial_audit import (
    AUDIT_SCHEMA_VERSION,
    LOCKED_GATE_STEP,
    audit_image,
    file_sha256,
    load_native_argmax_digests,
    prediction_digest_mismatches,
    remove_temporary_tree,
    sample_stat_digest,
    spatial_summary,
    validate_native_reference_contract,
    validate_native_reference_bindings,
    validate_q_lift_checkpoint_contract,
)
from eval_cafe_vc_protocols import checkpoint_config, protocol_settings
from train_cafe_rc import sliding_logits


REFERENCE_CONFIG = "evaluation_config.json"
REFERENCE_RESULT = "results.json"
AUDIT_TEMP_SHARD_DIRNAME = ".spatial_audit_shards.tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-evaluation-dir", required=True,
                        help="Completed native P/<arm> or D/<arm> evaluator output.")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audit-output-dir", required=True,
                        help="Fresh output directory; existing outputs are never reused.")
    parser.add_argument("--memory-fraction", type=float, default=0.75,
                        help="Resource limit only; it does not alter the locked inference protocol.")
    parser.add_argument("--progress-every", type=int, default=24)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _source_file_hashes() -> dict[str, str]:
    paths = (
        ROOT / "scripts" / "eval_cafe_vc_protocols.py",
        ROOT / "scripts" / "cafedino_locked_loveda.py",
        ROOT / "scripts" / "train_cafe_rc.py",
        ROOT / "dinotool" / "cafe_qlift.py",
        ROOT / "dinotool" / "qlift.py",
        ROOT / "dinotool" / "qlift_spatial_audit.py",
        Path(__file__).resolve(),
    )
    return {str(path): _sha256(path) for path in paths}


def validate_reference_contract(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Reject anything other than a completed, full, native Q-Lift P/D evaluation."""

    protocol = config.get("protocol")
    if not isinstance(protocol, str):
        raise ValueError("Reference evaluation lacks a P/D protocol")
    settings = protocol_settings(protocol)
    validate_native_reference_contract(config, result, protocol=protocol, settings=settings)
    return settings


def validate_checkpoint_contract(
    payload: dict[str, Any],
    *,
    reference: dict[str, Any],
    base_checkpoint: Path,
) -> None:
    """Bind a source-selected 2,612-update Q-Lift checkpoint to its reference."""
    validate_q_lift_checkpoint_contract(payload, reference=reference, base_checkpoint=base_checkpoint)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"Reference contract mismatch for {label}")


def _cleanup_temporary_shards(output: Path) -> None:
    """Ensure an integrity failure never leaves raw DDP diagnostics published."""

    remove_temporary_tree(output / AUDIT_TEMP_SHARD_DIRNAME)


def _write_integrity_failure(output: Path, status: str, **details: Any) -> None:
    _cleanup_temporary_shards(output)
    _write_json_atomic(output / "integrity.json", {"status": status, **details})


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference_metric_mismatches(summary: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """A matrix match is the hard gate; these checks make the report self-auditing."""

    expected = summary["global"]
    mismatches: dict[str, Any] = {}
    for key in ("mean_iou", "pixel_accuracy"):
        actual, target = expected.get(key), result.get(key)
        if actual is None or target is None or not math.isclose(float(actual), float(target), rel_tol=0.0, abs_tol=1e-12):
            mismatches[key] = {"reconstructed": actual, "reference": target}
    if "foreground_mean_with_background_false_positives" in expected:
        key = "foreground_mean_with_background_false_positives"
        actual, target = expected.get(key), result.get(key)
        if actual is None or target is None or not math.isclose(float(actual), float(target), rel_tol=0.0, abs_tol=1e-12):
            mismatches[key] = {"reconstructed": actual, "reference": target}
    return mismatches


def _make_manifest(
    *,
    reference_dir: Path,
    config: dict[str, Any],
    weights: Path,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "reference_evaluation_dir": str(reference_dir),
        "reference_evaluation_config_sha256": _sha256(reference_dir / REFERENCE_CONFIG),
        "reference_results_sha256": _sha256(reference_dir / REFERENCE_RESULT),
        "reference_protocol": config["protocol"],
        "reference_model_variant": config["model_variant"],
        "reference_weights": str(weights),
        "reference_weights_sha256": _sha256(weights),
        "reference_world_size": config["world_size"],
        "sample_keys_sha256": hashlib.sha256("\n".join(config["sample_keys"]).encode("utf-8")).hexdigest(),
        "current_sample_file_stat_sha256": sample_stat_digest(samples),
        "reference_integrity_bindings": config["integrity_bindings"],
        "data_root": str(Path(args.data_root).resolve()),
        "definitions": {
            "confusion_axes": "rows are mapped ground-truth class IDs; columns are argmax prediction IDs",
            "boundary": "valid GT pixel with an 8-neighbour valid GT pixel of a different class",
            "interior": "valid GT pixel that is not in the GT semantic boundary stratum",
            "components": "8-connected GT semantic components at final scored resolution; not instance annotations",
            "small_object": "tiny + small: component area <= 0.5% of valid evaluated pixels in that image",
            "pairwise": "predeclared focal pairs plus exhaustive unordered class-pair diagnostics",
        },
        "source_file_sha256": _source_file_hashes(),
    }


@torch.inference_mode()
def run(args: argparse.Namespace, rank: int, local_rank: int, world: int) -> None:
    reference_dir = Path(args.reference_evaluation_dir).resolve()
    output = Path(args.audit_output_dir).resolve()
    config = _load_json(reference_dir / REFERENCE_CONFIG)
    result = _load_json(reference_dir / REFERENCE_RESULT)
    settings = validate_reference_contract(config, result)
    if config.get("world_size") != world:
        raise ValueError(
            f"Audit world_size={world} must match the completed reference world_size={config.get('world_size')}"
        )
    if not 0 < args.memory_fraction <= 1 or args.progress_every < 1:
        raise ValueError("Invalid resource settings")
    root = Path(args.data_root).resolve()
    if root.name.lower() not in {"val", "validation"} or "loveda" not in str(root).lower():
        raise ValueError("Only the locked external LoveDA validation root is allowed")
    samples = discover_loveda_samples(root)
    if len(samples) != 1669:
        raise ValueError(f"Locked LoveDA validation needs 1,669 images, found {len(samples)}")
    sample_keys = [sample.key for sample in samples]
    _require_equal("sample keys", sample_keys, config["sample_keys"])
    weights = Path(config["weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Selected checkpoint must deserialize to a mapping")
    validate_checkpoint_contract(payload, reference=config, base_checkpoint=Path(args.base_checkpoint))
    bindings = validate_native_reference_bindings(
        config,
        project_root=ROOT,
        official_root=Path(args.official_root),
        selected_checkpoint=weights,
        base_checkpoint=Path(args.base_checkpoint),
        bpe_path=Path(args.bpe_path),
        samples=samples,
        data_root=root,
    )
    reference_digests = load_native_argmax_digests(
        reference_dir, bindings["native_argmax_digests"], sample_keys,
    )
    model_config = checkpoint_config(payload)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Audit output must be fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / AUDIT_TEMP_SHARD_DIRNAME).mkdir()
        _write_json_atomic(output / "audit_manifest.json", _make_manifest(
            reference_dir=reference_dir, config=config, weights=weights, samples=samples, args=args,
        ))
        _write_json_atomic(output / "status.json", {
            "status": "evaluating", "protocol": config["protocol"], "model_variant": config["model_variant"],
            "reference_evaluation_dir": str(reference_dir), "world_size": world,
        })
    if world > 1:
        dist.barrier()

    # This exact construction mirrors the Q-Lift branch of eval_cafe_vc_protocols.py.
    cafe, _, _ = build_model(argparse.Namespace(
        official_root=args.official_root, checkpoint=args.base_checkpoint, bpe_path=args.bpe_path,
        device=str(device), window_size=settings["window_size"], model_mode="eval",
    ))
    model = CafeQLift(cafe, model_config).to(device).eval()
    model.load_adapted_state_dict(payload["adapted_state"])
    text = cafe.build_text_embeddings([settings["classes"]]).detach()

    classes = len(settings["classes"])
    local_matrix = np.zeros((classes, classes), dtype=np.int64)
    started = time.perf_counter()
    shard_root = output / AUDIT_TEMP_SHARD_DIRNAME
    image_shard = shard_root / f"images_rank{rank:02d}.jsonl"
    component_shard = shard_root / f"components_rank{rank:02d}.jsonl"
    with image_shard.open("w", encoding="utf-8") as image_stream, component_shard.open("w", encoding="utf-8") as component_stream:
        local_indices = range(rank, len(samples), world)
        for position, sample_index in enumerate(local_indices, 1):
            sample = samples[sample_index]
            image = image_tensor(sample.image_path, settings["evaluation_size"], device)
            logits = sliding_logits(model, image, text, settings["window_size"], settings["stride"], config["amp"])
            prediction = logits.argmax(0).cpu().numpy()
            # Deliberately after prediction: labels cannot influence inference.
            target = target_ids(sample.mask_path, settings["evaluation_size"], include_background=settings["include_background"])
            image_record, components = audit_image(
                prediction, target, settings["classes"], sample_index=sample_index, key=sample.key,
            )
            local_matrix += np.asarray(image_record["confusion_matrix"], dtype=np.int64)
            image_stream.write(json.dumps(image_record, ensure_ascii=True, separators=(",", ":")) + "\n")
            for component in components:
                component_stream.write(json.dumps(component, ensure_ascii=True, separators=(",", ":")) + "\n")
            if rank == 0 and (position % args.progress_every == 0 or sample_index + world >= len(samples)):
                _write_json_atomic(output / "status.json", {
                    "status": "evaluating", "protocol": config["protocol"], "model_variant": config["model_variant"],
                    "rank0_images": position, "rank0_total": len(range(0, len(samples), world)),
                    "elapsed_seconds": time.perf_counter() - started,
                })
    matrix_tensor = torch.as_tensor(local_matrix, dtype=torch.int64, device=device)
    if world > 1:
        dist.all_reduce(matrix_tensor)
        dist.barrier()

    if rank != 0:
        return
    image_records = [record for shard_rank in range(world)
                     for record in _read_jsonl(shard_root / f"images_rank{shard_rank:02d}.jsonl")]
    components = [record for shard_rank in range(world)
                  for record in _read_jsonl(shard_root / f"components_rank{shard_rank:02d}.jsonl")]
    image_records.sort(key=lambda item: item["sample_index"])
    components.sort(key=lambda item: (item["sample_index"], item["class_index"], item["component_id"]))
    observed_indices = [item["sample_index"] for item in image_records]
    if observed_indices != list(range(len(samples))):
        _write_integrity_failure(output, "audit_shard_index_mismatch", images=len(image_records))
        raise RuntimeError("DDP audit shard merge has missing, duplicate, or reordered images")
    try:
        digest_mismatches = prediction_digest_mismatches(reference_digests, image_records, sample_keys)
    except ValueError as error:
        _write_integrity_failure(output, "native_argmax_digest_contract_failure", error=repr(error))
        raise RuntimeError("Refusing spatial diagnostics: native per-image digest contract is invalid") from error
    if digest_mismatches:
        _write_integrity_failure(
            output,
            "native_argmax_digest_mismatch",
            mismatch_count=len(digest_mismatches),
            examples=digest_mismatches[:8],
        )
        raise RuntimeError("Refusing spatial diagnostics: per-image argmax digests differ from native evaluation")
    summary = spatial_summary(image_records, components, settings["classes"], include_background=settings["include_background"])
    reduced = matrix_tensor.cpu().numpy()
    reconstructed = np.asarray(summary["global"]["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(reconstructed, reduced):
        _write_integrity_failure(output, "per_image_ddp_confusion_mismatch")
        raise RuntimeError("Per-image audit confusion does not equal the DDP-reduced audit confusion")
    if summary["ignored_pixels"] != result["ignored_pixels"]:
        _write_integrity_failure(
            output,
            "ignored_pixel_mismatch",
            audit_ignored_pixels=summary["ignored_pixels"],
            reference_ignored_pixels=result["ignored_pixels"],
        )
        raise RuntimeError("Refusing spatial diagnostics: ignored-pixel support differs from reference results")
    reference_matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(reconstructed, reference_matrix):
        _write_integrity_failure(
            output,
            "reference_confusion_mismatch",
            reference_evaluation_dir=str(reference_dir),
            audit_confusion_matrix=reconstructed.tolist(),
            reference_confusion_matrix=reference_matrix.tolist(),
            difference=(reconstructed - reference_matrix).tolist(),
        )
        raise RuntimeError("Refusing spatial diagnostics: reconstructed global confusion differs from reference results")
    metric_mismatches = _reference_metric_mismatches(summary, result)
    if metric_mismatches:
        _write_integrity_failure(
            output,
            "reference_metric_mismatch_after_exact_confusion_match",
            mismatches=metric_mismatches,
        )
        raise RuntimeError("Refusing spatial diagnostics: metric reconstruction differs from reference results")
    _write_jsonl_atomic(output / "per_image.jsonl", image_records)
    _write_jsonl_atomic(output / "components.jsonl", components)
    _write_json_atomic(output / "spatial_summary.json", summary)
    _write_json_atomic(output / "integrity.json", {
        "status": "complete",
        "reference_integrity_bindings_exact_match": True,
        "native_per_image_argmax_digest_exact_match": True,
        "reference_confusion_exact_match": True,
        "per_image_confusion_equals_ddp_reduce": True,
        "ignored_pixels_exact_match": True,
        "reference_metric_mismatches": {},
        "images": len(image_records),
        "components": len(components),
    })
    _write_json_atomic(output / "status.json", {
        "status": "complete", "protocol": config["protocol"], "model_variant": config["model_variant"],
        "images": len(image_records), "components": len(components),
        "mean_iou_percent": summary["global"]["mean_iou_percent"],
        "reference_confusion_exact_match": True,
    })
    _cleanup_temporary_shards(output)


def main() -> None:
    args = parse_args()
    rank, local_rank, world = (
        int(os.getenv(key, default))
        for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    output = Path(args.audit_output_dir).resolve()
    try:
        run(args, rank, local_rank, world)
    except BaseException as error:
        if rank == 0 and output.is_dir() and not (output / "spatial_summary.json").exists():
            _cleanup_temporary_shards(output)
            _write_json_atomic(output / "failure.json", {"error": repr(error)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
