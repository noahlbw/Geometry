"""Pure CPU spatial diagnostics for the locked Q-Lift LoveDA audit.

This module deliberately contains no model loading, image preprocessing, or
checkpoint selection.  The executable audit imports the locked evaluator's
inference helpers and calls the functions here *after* each prediction is
fixed.  Keeping the spatial bookkeeping separate makes its definitions easy
to unit test without a GPU or a LoveDA checkout.
"""
from __future__ import annotations

from collections import deque
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Sequence

import numpy as np

try:  # OpenCV is already a pinned dependency of the locked LoveDA evaluator.
    import cv2
except ModuleNotFoundError:  # pragma: no cover - only useful for pure import diagnostics.
    cv2 = None


AUDIT_SCHEMA_VERSION = "qlift_loveda_spatial_audit_v2"
NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION = "qlift_loveda_native_argmax_digest_v1"
NATIVE_REFERENCE_BINDING_SCHEMA_VERSION = "qlift_loveda_native_reference_binding_v1"
NATIVE_ARGMAX_DIGEST_FILENAME = "native_argmax_digests.jsonl"
NATIVE_ARGMAX_DIGEST_TEMP_DIRNAME = ".native_argmax_digest_shards.tmp"
NATIVE_EVALUATOR_SOURCE_FILES: tuple[str, ...] = (
    "scripts/eval_cafe_vc_protocols.py",
    "scripts/cafedino_locked_loveda.py",
    "scripts/train_cafe_rc.py",
    "dinotool/cafe_qlift.py",
    "dinotool/qlift.py",
    "dinotool/loveda.py",
    "dinotool/inference.py",
)
OFFICIAL_EVALUATOR_SOURCE_FILES: tuple[str, ...] = (
    "CAFe_DINO/modeling/cafedino.py",
    "anyup/anyup/model.py",
    "dinov3/hub/dinotxt.py",
)
IGNORE_LABEL = 255
LOCKED_GATE_STEP = 2612
AREA_BINS: tuple[tuple[str, float, float], ...] = (
    ("tiny", 0.0, 0.0005),
    ("small", 0.0005, 0.005),
    ("medium", 0.005, 0.05),
    ("large", 0.05, float("inf")),
)


def file_sha256(path: Path) -> str:
    """Return the complete content digest of one immutable audit input file."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _same_resolved_path(first: Any, second: Path) -> bool:
    return isinstance(first, str) and Path(first).resolve() == Path(second).resolve()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def sample_key_digest(sample_keys: Sequence[str]) -> str:
    """Digest the locked sample order independently of the image/mask contents."""

    if any(not isinstance(key, str) or not key for key in sample_keys):
        raise ValueError("Sample keys must be nonempty strings")
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Sample keys must be unique")
    return sha256("\n".join(sample_keys).encode("utf-8")).hexdigest()


def native_evaluator_source_hashes(project_root: Path, official_root: Path) -> dict[str, str]:
    """Hash every direct source entrypoint used by the native Q-Lift P/D run.

    The list is deliberately explicit rather than recursively hashing a checkout:
    it binds the evaluator, fixed CAFe loader, sliding-window implementation,
    Q-Lift decoder, sample discovery, and the three official modules directly
    imported by the locked loader without turning unrelated checkout files into
    a hidden protocol dependency.
    """

    project_root = Path(project_root).resolve()
    official_root = Path(official_root).resolve()
    entries = {
        f"project/{relative}": project_root / relative
        for relative in NATIVE_EVALUATOR_SOURCE_FILES
    }
    entries.update({
        f"official/{relative}": official_root / relative
        for relative in OFFICIAL_EVALUATOR_SOURCE_FILES
    })
    return {name: file_sha256(path) for name, path in entries.items()}


def _content_file_record(path: Path, data_root: Path) -> dict[str, Any]:
    """Bind one LoveDA input file by root-relative path, bytes, and content."""

    path = Path(path).resolve()
    try:
        relative_path = path.relative_to(data_root).as_posix()
    except ValueError as error:
        raise ValueError(f"LoveDA input is outside the locked data root: {path}") from error
    stat = path.stat()
    return {
        "relative_path": relative_path,
        "bytes": int(stat.st_size),
        "sha256": file_sha256(path),
    }


def sample_manifest_binding(samples: Sequence[Any], data_root: Path) -> dict[str, Any]:
    """Record immutable per-image/per-mask content identity for native P/D.

    File size and mtime remain as a cheap provenance checksum, but the actual
    audit binding is a content SHA-256 record for every RGB image and mapped
    LoveDA mask.  This prevents a same-size, restored-mtime label edit from
    being silently accepted by boundary/component diagnostics.
    """

    data_root = Path(data_root).resolve()
    sample_keys = [str(sample.key) for sample in samples]
    files = [
        {
            "key": key,
            "image": _content_file_record(sample.image_path, data_root),
            "mask": _content_file_record(sample.mask_path, data_root),
        }
        for key, sample in zip(sample_keys, samples)
    ]
    binding = {
        "data_root": str(data_root),
        "images": int(len(samples)),
        "sample_keys_sha256": sample_key_digest(sample_keys),
        "sample_file_stat_sha256": sample_stat_digest(samples),
        "sample_content_sha256": _canonical_json_sha256(files),
        "files": files,
    }
    binding["sha256"] = _canonical_json_sha256(binding)
    return binding


def native_reference_bindings(
    *,
    project_root: Path,
    official_root: Path,
    selected_checkpoint: Path,
    base_checkpoint: Path,
    bpe_path: Path,
    samples: Sequence[Any],
    data_root: Path,
) -> dict[str, Any]:
    """Build strict config-level bindings for a native Q-Lift LoveDA run."""

    return {
        "schema_version": NATIVE_REFERENCE_BINDING_SCHEMA_VERSION,
        "selected_checkpoint": _file_binding(selected_checkpoint),
        "base_checkpoint": _file_binding(base_checkpoint),
        "bpe": _file_binding(bpe_path),
        "evaluator_source_sha256": native_evaluator_source_hashes(project_root, official_root),
        "data_manifest": sample_manifest_binding(samples, data_root),
    }


def _validate_file_binding(binding: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"Reference integrity binding lacks {label}")
    path, size = binding.get("path"), binding.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Reference integrity binding {label}.path is invalid")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"Reference integrity binding {label}.bytes is invalid")
    _require_sha256(binding.get("sha256"), f"Reference integrity binding {label}.sha256")
    return binding


def _validate_native_digest_binding(binding: Any, *, images: int) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError("Reference integrity binding lacks native_argmax_digests")
    if binding.get("schema_version") != NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION:
        raise ValueError("Reference native argmax digest schema is unsupported")
    path = binding.get("path")
    if not isinstance(path, str) or Path(path).name != path or path != NATIVE_ARGMAX_DIGEST_FILENAME:
        raise ValueError("Reference native argmax digest path must be the locked relative filename")
    if binding.get("entries") != images:
        raise ValueError("Reference native argmax digest count does not match the locked split")
    _require_sha256(binding.get("sha256"), "Reference native argmax digest sha256")
    return binding


def validate_native_reference_binding_structure(config: dict[str, Any]) -> dict[str, Any]:
    """Validate fields that must exist in a completed native evaluator config."""

    bindings = config.get("integrity_bindings")
    if not isinstance(bindings, dict) or bindings.get("schema_version") != NATIVE_REFERENCE_BINDING_SCHEMA_VERSION:
        raise ValueError("Reference lacks the required native inference integrity bindings")
    checkpoint = _validate_file_binding(bindings.get("selected_checkpoint"), label="selected_checkpoint")
    base = _validate_file_binding(bindings.get("base_checkpoint"), label="base_checkpoint")
    bpe = _validate_file_binding(bindings.get("bpe"), label="bpe")
    sources = bindings.get("evaluator_source_sha256")
    expected_source_keys = {
        *(f"project/{relative}" for relative in NATIVE_EVALUATOR_SOURCE_FILES),
        *(f"official/{relative}" for relative in OFFICIAL_EVALUATOR_SOURCE_FILES),
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_keys:
        raise ValueError("Reference evaluator source binding is incomplete")
    for name, digest in sources.items():
        _require_sha256(digest, f"Reference evaluator source hash {name}")
    manifest = bindings.get("data_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Reference integrity binding lacks data_manifest")
    if not isinstance(manifest.get("data_root"), str) or not manifest["data_root"]:
        raise ValueError("Reference data manifest path is invalid")
    if manifest.get("images") != config.get("images"):
        raise ValueError("Reference data manifest image count does not match config")
    _require_sha256(manifest.get("sample_keys_sha256"), "Reference data manifest sample key hash")
    if manifest["sample_keys_sha256"] != sample_key_digest(config.get("sample_keys", ())):
        raise ValueError("Reference data manifest sample key hash does not match config ordering")
    _require_sha256(manifest.get("sample_file_stat_sha256"), "Reference data manifest sample stat hash")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest["images"]:
        raise ValueError("Reference data manifest lacks one content record per image/mask pair")
    for index, (record, key) in enumerate(zip(files, config.get("sample_keys", ()) )):
        if not isinstance(record, dict) or record.get("key") != key:
            raise ValueError(f"Reference data manifest content record has an invalid key at index {index}")
        for side in ("image", "mask"):
            file_record = record.get(side)
            if not isinstance(file_record, dict):
                raise ValueError(f"Reference data manifest lacks {side} content at index {index}")
            relative_path, byte_count = file_record.get("relative_path"), file_record.get("bytes")
            if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
                raise ValueError(f"Reference data manifest {side} path is invalid at index {index}")
            if not isinstance(byte_count, int) or byte_count < 0:
                raise ValueError(f"Reference data manifest {side} bytes are invalid at index {index}")
            _require_sha256(file_record.get("sha256"), f"Reference data manifest {side} hash at index {index}")
    _require_sha256(manifest.get("sample_content_sha256"), "Reference data manifest content hash")
    if manifest["sample_content_sha256"] != _canonical_json_sha256(files):
        raise ValueError("Reference data manifest content hash does not reconstruct")
    expected_manifest = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != _canonical_json_sha256(expected_manifest):
        raise ValueError("Reference data manifest hash does not reconstruct")
    _validate_native_digest_binding(bindings.get("native_argmax_digests"), images=int(config["images"]))
    if config.get("weights") != checkpoint["path"]:
        raise ValueError("Reference weights path is not bound to its selected checkpoint")
    base_manifest = config.get("base_checkpoint")
    if not isinstance(base_manifest, dict):
        raise ValueError("Reference lacks the base CAFe checkpoint manifest")
    base_record = base_manifest.get("checkpoint")
    bpe_record = base_manifest.get("bpe")
    if not isinstance(base_record, dict) or not _same_resolved_path(base_record.get("path"), Path(base["path"])) or base_record.get("bytes") != base["bytes"]:
        raise ValueError("Reference base checkpoint manifest is not bound to its hash")
    if not isinstance(bpe_record, dict) or not _same_resolved_path(bpe_record.get("path"), Path(bpe["path"])) or bpe_record.get("bytes") != bpe["bytes"]:
        raise ValueError("Reference BPE manifest is not bound to its hash")
    return bindings


def validate_native_reference_bindings(
    config: dict[str, Any],
    *,
    project_root: Path,
    official_root: Path,
    selected_checkpoint: Path,
    base_checkpoint: Path,
    bpe_path: Path,
    samples: Sequence[Any],
    data_root: Path,
) -> dict[str, Any]:
    """Require a reference's hashes to bind exactly to this audit invocation."""

    bindings = validate_native_reference_binding_structure(config)
    expected = native_reference_bindings(
        project_root=project_root,
        official_root=official_root,
        selected_checkpoint=selected_checkpoint,
        base_checkpoint=base_checkpoint,
        bpe_path=bpe_path,
        samples=samples,
        data_root=data_root,
    )
    for name in ("selected_checkpoint", "base_checkpoint", "bpe", "evaluator_source_sha256", "data_manifest"):
        if bindings.get(name) != expected[name]:
            raise ValueError(f"Reference integrity binding mismatch for {name}")
    return bindings


def prediction_digest_record(prediction: np.ndarray, *, sample_index: int, key: str) -> dict[str, Any]:
    """Create a label-free, order-bound native argmax provenance record."""

    if not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a nonempty string")
    if prediction.ndim != 2:
        raise ValueError("Argmax digest requires an HW prediction map")
    return {
        "sample_index": sample_index,
        "key": key,
        "prediction_sha256": prediction_digest(prediction),
    }


def _validated_digest_records(records: Sequence[dict[str, Any]], sample_keys: Sequence[str], *, label: str) -> list[dict[str, Any]]:
    if len(records) != len(sample_keys):
        raise ValueError(f"{label} count {len(records)} does not equal locked sample count {len(sample_keys)}")
    ordered = sorted(records, key=lambda item: item.get("sample_index", -1))
    for index, (record, expected_key) in enumerate(zip(ordered, sample_keys)):
        if not isinstance(record, dict) or record.get("sample_index") != index or record.get("key") != expected_key:
            raise ValueError(f"{label} has a missing, duplicate, reordered, or key-mismatched sample at index {index}")
        _require_sha256(record.get("prediction_sha256"), f"{label} prediction digest at index {index}")
    return ordered


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected only JSON object rows in {path}")
    return rows


def remove_temporary_tree(path: Path) -> None:
    """Delete only a purpose-named temporary directory below a known output root."""

    path = Path(path)
    if not path.name.startswith(".") or not path.name.endswith(".tmp"):
        raise ValueError(f"Refusing to remove a non-temporary path: {path}")
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        raise ValueError(f"Temporary path is not a directory: {path}")


def merge_native_argmax_digest_shards(
    *,
    shard_dir: Path,
    output_path: Path,
    sample_keys: Sequence[str],
    world_size: int,
) -> dict[str, Any]:
    """Validate, atomically publish, then remove native argmax digest shards."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    shard_dir = Path(shard_dir)
    output_path = Path(output_path)
    try:
        records = [
            record
            for rank in range(world_size)
            for record in _read_jsonl(shard_dir / f"rank{rank:02d}.jsonl")
        ]
        records = _validated_digest_records(records, sample_keys, label="Native argmax digest shards")
        if output_path.name != NATIVE_ARGMAX_DIGEST_FILENAME:
            raise ValueError("Native argmax digest must use the locked output filename")
        _write_jsonl_atomic(output_path, records)
        return {
            "schema_version": NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION,
            "path": output_path.name,
            "entries": len(records),
            "sha256": file_sha256(output_path),
        }
    finally:
        remove_temporary_tree(shard_dir)


def load_native_argmax_digests(reference_dir: Path, binding: dict[str, Any], sample_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Load a complete published native digest manifest after verifying its hash."""

    binding = _validate_native_digest_binding(binding, images=len(sample_keys))
    path = Path(reference_dir).resolve() / binding["path"]
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("Reference native argmax digest file hash does not match evaluation config")
    return _validated_digest_records(_read_jsonl(path), sample_keys, label="Reference native argmax digests")


def prediction_digest_mismatches(
    reference_records: Sequence[dict[str, Any]],
    observed_records: Sequence[dict[str, Any]],
    sample_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Return exact native-vs-audit argmax mismatches after order/key validation."""

    reference = _validated_digest_records(reference_records, sample_keys, label="Reference native argmax digests")
    observed = _validated_digest_records(observed_records, sample_keys, label="Spatial-audit argmax digests")
    return [
        {
            "sample_index": expected["sample_index"],
            "key": expected["key"],
            "reference_prediction_sha256": expected["prediction_sha256"],
            "audit_prediction_sha256": actual["prediction_sha256"],
        }
        for expected, actual in zip(reference, observed)
        if expected["prediction_sha256"] != actual["prediction_sha256"]
    ]


def valid_target_mask(target: np.ndarray, classes: int) -> np.ndarray:
    """Return the exact scored-pixel support for a mapped LoveDA target."""

    if target.ndim != 2:
        raise ValueError(f"Expected an HW target map, got {target.shape}")
    return (target >= 0) & (target < classes)


def validate_prediction(prediction: np.ndarray, target: np.ndarray, classes: int) -> np.ndarray:
    """Validate an argmax map and return the scored-pixel support."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Prediction {prediction.shape} must match HW target {target.shape}")
    valid = valid_target_mask(target, classes)
    if valid.any() and ((prediction[valid] < 0) | (prediction[valid] >= classes)).any():
        values = np.unique(prediction[valid][(prediction[valid] < 0) | (prediction[valid] >= classes)]).tolist()
        raise ValueError(f"Prediction contains invalid class IDs on scored pixels: {values}")
    return valid


def confusion_for_mask(prediction: np.ndarray, target: np.ndarray, classes: int, mask: np.ndarray) -> np.ndarray:
    """Return a GT-row / prediction-column integer confusion matrix."""

    valid = validate_prediction(prediction, target, classes)
    if mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError("The diagnostic mask must be a bool HW array matching target")
    selected = valid & mask
    matrix = np.zeros((classes, classes), dtype=np.int64)
    if not selected.any():
        return matrix
    bins = target[selected].astype(np.int64) * classes + prediction[selected].astype(np.int64)
    return np.bincount(bins, minlength=classes * classes).reshape(classes, classes).astype(np.int64, copy=False)


def semantic_boundary_mask(target: np.ndarray, classes: int) -> np.ndarray:
    """Mark scored pixels adjacent (8-neighbour) to a different scored class.

    Ignored pixels deliberately do not create a boundary.  This makes P a
    foreground-class-interface diagnostic, not an object-versus-background
    boundary metric, because P removes background from its scored support.
    """

    valid = valid_target_mask(target, classes)
    values = target.astype(np.int64, copy=False)
    padded_values = np.pad(values, 1, mode="constant", constant_values=-1)
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    height, width = target.shape
    boundary = np.zeros_like(valid)
    center = padded_values[1:height + 1, 1:width + 1]
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbours = padded_values[dy:dy + height, dx:dx + width]
            neighbour_valid = padded_valid[dy:dy + height, dx:dx + width]
            boundary |= valid & neighbour_valid & (center != neighbours)
    return boundary


def per_class_summary(matrix: np.ndarray, names: Sequence[str]) -> list[dict[str, Any]]:
    """Compute the same IoU bookkeeping used by the locked evaluator."""

    if matrix.shape != (len(names), len(names)):
        raise ValueError("Confusion matrix shape does not match class names")
    intersection = np.diag(matrix)
    target_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    union = target_pixels + predicted_pixels - intersection
    output: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        iou = None if union[index] == 0 else float(intersection[index] / union[index])
        output.append({
            "id": index,
            "name": str(name),
            "iou": iou,
            "iou_percent": None if iou is None else round(100.0 * iou, 4),
            "intersection_pixels": int(intersection[index]),
            "union_pixels": int(union[index]),
            "target_pixels": int(target_pixels[index]),
            "predicted_pixels": int(predicted_pixels[index]),
            "predicted_to_gt": None if target_pixels[index] == 0 else float(predicted_pixels[index] / target_pixels[index]),
        })
    return output


def confusion_summary(matrix: np.ndarray, names: Sequence[str], *, include_background: bool) -> dict[str, Any]:
    """Summarize an integer confusion matrix without hiding zero denominators."""

    per_class = per_class_summary(matrix, names)
    ious = np.asarray([np.nan if item["iou"] is None else item["iou"] for item in per_class], dtype=np.float64)
    target_pixels = matrix.sum(axis=1)
    labeled = int(target_pixels.sum())
    correct = int(np.diag(matrix).sum())
    mean_iou = None if np.isnan(ious).all() else float(np.nanmean(ious))
    payload: dict[str, Any] = {
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "labeled_pixels": labeled,
        "correct_pixels": correct,
        "pixel_accuracy": None if not labeled else float(correct / labeled),
        "per_class": per_class,
        "mean_iou": mean_iou,
        "mean_iou_percent": None if mean_iou is None else round(100.0 * mean_iou, 4),
    }
    if include_background:
        foreground = ious[1:]
        value = None if not len(foreground) or np.isnan(foreground).all() else float(np.nanmean(foreground))
        payload["foreground_mean_with_background_false_positives"] = value
        payload["foreground_mean_with_background_false_positives_percent"] = (
            None if value is None else round(100.0 * value, 4)
        )
    return payload


def stratum_summary(matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """GT-conditioned boundary/interior correctness, not a misleading stratum IoU."""

    if matrix.shape != (len(names), len(names)):
        raise ValueError("Confusion matrix shape does not match class names")
    targets = matrix.sum(axis=1)
    correct = np.diag(matrix)
    pixels = int(targets.sum())
    return {
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "pixels": pixels,
        "correct_pixels": int(correct.sum()),
        "accuracy": None if not pixels else float(correct.sum() / pixels),
        "per_class_recall": {
            str(name): None if targets[index] == 0 else float(correct[index] / targets[index])
            for index, name in enumerate(names)
        },
    }


def area_bin(area_fraction: float) -> str:
    """Map a GT component's valid-image-normalized area to its predeclared bin."""

    if not 0 < area_fraction <= 1:
        raise ValueError(f"Component area fraction must be in (0, 1], got {area_fraction}")
    for name, lower, upper in AREA_BINS:
        if lower < area_fraction <= upper:
            return name
    raise AssertionError("Area bins must cover (0, 1]")


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV 8-connected labels and stats for a binary mask."""

    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Expected an HW bool component mask")
    if cv2 is not None:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(mask.astype(np.uint8)), connectivity=8,
        )
        return labels, stats

    # The normal audit runtime has OpenCV because the locked evaluator imports
    # it.  This compact fallback keeps the metric definitions independently
    # testable in a CPU-only environment without adding a new dependency.
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    stats: list[list[int]] = [[0, 0, width, height, int((~mask).sum())]]
    label = 0
    for top in range(height):
        for left in range(width):
            if not mask[top, left] or labels[top, left]:
                continue
            label += 1
            labels[top, left] = label
            queue: deque[tuple[int, int]] = deque([(top, left)])
            area = 0
            min_y = max_y = top
            min_x = max_x = left
            while queue:
                y, x = queue.popleft()
                area += 1
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            queue.append((ny, nx))
            stats.append([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area])
    return labels, np.asarray(stats, dtype=np.int32)


def component_records(
    prediction: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    sample_index: int,
    key: str,
) -> list[dict[str, Any]]:
    """Describe 8-connected GT semantic components without instance claims."""

    classes = len(names)
    valid = validate_prediction(prediction, target, classes)
    valid_pixels = int(valid.sum())
    if not valid_pixels:
        return []
    records: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(names):
        labels, stats = _connected_components(valid & (target == class_index))
        for component_id in range(1, len(stats)):
            component = labels == component_id
            area = int(stats[component_id, 4])
            if area <= 0:
                continue
            histogram = np.bincount(prediction[component].astype(np.int64), minlength=classes)
            maximum = int(histogram.max())
            dominant_ids = np.flatnonzero(histogram == maximum).astype(np.int64).tolist()
            same_class = int(histogram[class_index])
            fraction = float(area / valid_pixels)
            records.append({
                "sample_index": int(sample_index),
                "key": str(key),
                "class_index": int(class_index),
                "class": str(class_name),
                "component_id": int(component_id),
                "area_px": area,
                "area_fraction_of_valid": fraction,
                "area_bin": area_bin(fraction),
                "bbox_xywh": [int(value) for value in stats[component_id, :4]],
                "same_class_pixels": same_class,
                "same_class_fraction": float(same_class / area),
                "hit_at_25": bool(same_class / area >= 0.25),
                "hit_at_50": bool(same_class / area >= 0.50),
                "prediction_histogram": [int(value) for value in histogram],
                "dominant_prediction": str(names[dominant_ids[0]]),
                "dominant_prediction_ties": [str(names[index]) for index in dominant_ids],
            })
    return records


def _aggregate_component_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "components": 0,
            "area_pixels": 0,
            "macro_same_class_coverage": None,
            "pixel_weighted_same_class_coverage": None,
            "hit_at_25_rate": None,
            "hit_at_50_rate": None,
            "dominant_error_counts": {},
        }
    areas = np.asarray([item["area_px"] for item in records], dtype=np.int64)
    same = np.asarray([item["same_class_pixels"] for item in records], dtype=np.int64)
    coverage = np.asarray([item["same_class_fraction"] for item in records], dtype=np.float64)
    dominant_errors: dict[str, int] = {}
    for item in records:
        if item["dominant_prediction"] != item["class"]:
            dominant_errors[item["dominant_prediction"]] = dominant_errors.get(item["dominant_prediction"], 0) + 1
    return {
        "components": int(len(records)),
        "area_pixels": int(areas.sum()),
        "macro_same_class_coverage": float(coverage.mean()),
        "pixel_weighted_same_class_coverage": float(same.sum() / areas.sum()),
        "hit_at_25_rate": float(np.mean([item["hit_at_25"] for item in records])),
        "hit_at_50_rate": float(np.mean([item["hit_at_50"] for item in records])),
        "dominant_error_counts": dominant_errors,
    }


def component_summary(records: Sequence[dict[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    """Aggregate component retention by semantic class and predeclared scale bin."""

    return {
        "definition": (
            "8-connected components of each GT semantic class at the final scored resolution; "
            "these are not instance annotations"
        ),
        "small_object_cohort": "tiny + small = components at most 0.5% of valid pixels in their image",
        "all": _aggregate_component_group(records),
        "by_class": {
            str(name): _aggregate_component_group([item for item in records if item["class"] == name])
            for name in names
        },
        "by_area_bin": {
            name: _aggregate_component_group([item for item in records if item["area_bin"] == name])
            for name, _, _ in AREA_BINS
        },
    }


def _pair_stat(matrix: np.ndarray, names: Sequence[str], first: str, second: str) -> dict[str, Any]:
    i, j = names.index(first), names.index(second)
    targets = matrix.sum(axis=1)
    first_to_second = int(matrix[i, j])
    second_to_first = int(matrix[j, i])
    denominator = int(targets[i] + targets[j])
    return {
        "first": first,
        "second": second,
        "first_to_second_pixels": first_to_second,
        "second_to_first_pixels": second_to_first,
        "first_to_second_rate": None if targets[i] == 0 else float(first_to_second / targets[i]),
        "second_to_first_rate": None if targets[j] == 0 else float(second_to_first / targets[j]),
        "mutual_cross_rate": None if denominator == 0 else float((first_to_second + second_to_first) / denominator),
    }


def pairwise_diagnostics(matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """All pairs plus the focal remote-sensing pairs registered before results."""

    all_pairs = {
        f"{first}<->{second}": _pair_stat(matrix, names, first, second)
        for first, second in combinations(names, 2)
    }
    if {"tree", "farm"}.issubset(names):
        focal = (
            ("vegetation<->agriculture", "tree", "farm"),
            ("barren<->vegetation", "barren", "tree"),
            ("barren<->road", "barren", "road"),
            ("building<->road", "building", "road"),
        )
    elif {"forest", "agricultural"}.issubset(names):
        focal = (
            ("vegetation<->agriculture", "forest", "agricultural"),
            ("barren<->vegetation", "barren", "forest"),
            ("barren<->road", "barren", "road"),
            ("building<->road", "building", "road"),
        )
    else:
        focal = ()
    return {
        "definition": "row-conditioned directional confusion and symmetric cross-pair rate",
        "focal_pairs": {canonical: _pair_stat(matrix, names, first, second) for canonical, first, second in focal},
        "all_pairs": all_pairs,
    }


def prediction_digest(prediction: np.ndarray) -> str:
    """Stable digest of a stored-free argmax map for future reproducibility checks."""

    return sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()


def audit_image(
    prediction: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    sample_index: int,
    key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate all post-prediction spatial diagnostics for one image."""

    classes = len(names)
    valid = validate_prediction(prediction, target, classes)
    boundary = semantic_boundary_mask(target, classes)
    interior = valid & ~boundary
    full = confusion_for_mask(prediction, target, classes, valid)
    boundary_matrix = confusion_for_mask(prediction, target, classes, boundary)
    interior_matrix = confusion_for_mask(prediction, target, classes, interior)
    if not np.array_equal(full, boundary_matrix + interior_matrix):
        raise RuntimeError("Boundary and interior must partition every scored pixel")
    record = {
        "sample_index": int(sample_index),
        "key": str(key),
        "evaluated_shape": [int(target.shape[0]), int(target.shape[1])],
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int((~valid).sum()),
        "prediction_sha256": prediction_digest(prediction),
        "confusion_matrix": full.tolist(),
        "per_class": per_class_summary(full, names),
        "gt_boundary": stratum_summary(boundary_matrix, names),
        "gt_interior": stratum_summary(interior_matrix, names),
    }
    return record, component_records(prediction, target, names, sample_index=sample_index, key=key)


def spatial_summary(
    image_records: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    include_background: bool,
) -> dict[str, Any]:
    """Merge independently written image diagnostics into the final audit report."""

    classes = len(names)
    full = np.zeros((classes, classes), dtype=np.int64)
    boundary = np.zeros_like(full)
    interior = np.zeros_like(full)
    ignored = 0
    for item in image_records:
        full += np.asarray(item["confusion_matrix"], dtype=np.int64)
        boundary += np.asarray(item["gt_boundary"]["confusion_matrix"], dtype=np.int64)
        interior += np.asarray(item["gt_interior"]["confusion_matrix"], dtype=np.int64)
        ignored += int(item["ignored_pixels"])
    if not np.array_equal(full, boundary + interior):
        raise ValueError("Merged boundary/interior confusion does not reconstruct full confusion")
    full_summary = confusion_summary(full, names, include_background=include_background)
    boundary_summary = stratum_summary(boundary, names)
    interior_summary = stratum_summary(interior, names)
    errors = int(full.sum() - np.diag(full).sum())
    boundary_errors = int(boundary.sum() - np.diag(boundary).sum())
    boundary_share = None if full.sum() == 0 else float(boundary.sum() / full.sum())
    error_share = None if errors == 0 else float(boundary_errors / errors)
    return {
        "global": full_summary,
        "ignored_pixels": ignored,
        "gt_boundary": boundary_summary,
        "gt_interior": interior_summary,
        "error_decomposition": {
            "all_errors": errors,
            "boundary_errors": boundary_errors,
            "interior_errors": errors - boundary_errors,
            "boundary_pixel_share": boundary_share,
            "boundary_error_share": error_share,
            "boundary_error_enrichment": (
                None if boundary_share in (None, 0.0) or error_share is None else float(error_share / boundary_share)
            ),
        },
        "components": component_summary(records, names),
        "pairwise_confusion": {
            "global": pairwise_diagnostics(full, names),
            "gt_boundary": pairwise_diagnostics(boundary, names),
            "gt_interior": pairwise_diagnostics(interior, names),
        },
    }


def sample_stat_digest(samples: Iterable[Any]) -> str:
    """Digest current sample key/path/size/mtime provenance without loading labels."""

    lines = []
    for sample in samples:
        image = sample.image_path.stat()
        mask = sample.mask_path.stat()
        lines.append(
            f"{sample.key}\t{sample.image_path}\t{image.st_size}\t{image.st_mtime_ns}\t"
            f"{sample.mask_path}\t{mask.st_size}\t{mask.st_mtime_ns}"
        )
    return sha256("\n".join(lines).encode("utf-8")).hexdigest()


def validate_native_reference_contract(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    protocol: str,
    settings: dict[str, Any],
) -> None:
    """Validate immutable P/D evaluator metadata without importing GPU code."""

    if protocol not in {"P", "D"}:
        raise ValueError("Spatial audit accepts only locked LoveDA P or D references")
    if result.get("status") != "complete":
        raise ValueError("Reference evaluation is not complete")
    if config.get("qlift_audit") != "native" or result.get("qlift_audit") != "native":
        raise ValueError("Spatial audit requires a native Q-Lift reference, not a mechanism counterfactual")
    if config.get("max_images") is not None:
        raise ValueError("Spatial audit refuses bounded reference evaluations")
    if config.get("target_used_for_training_or_selection") is not False:
        raise ValueError("Reference does not attest that LoveDA was excluded from training/selection")
    if config.get("model_mode") != "eval":
        raise ValueError("Only deterministic eval-mode references are allowed")
    if config.get("images") != 1669 or result.get("images") != 1669:
        raise ValueError("Spatial audit requires the locked complete 1,669-image LoveDA validation split")
    if not isinstance(config.get("world_size"), int) or isinstance(config.get("world_size"), bool) or config["world_size"] < 1:
        raise ValueError("Reference config world_size is invalid")
    if result.get("world_size") != config["world_size"]:
        raise ValueError("Reference result/config world_size mismatch")
    if not isinstance(result.get("ignored_pixels"), int) or isinstance(result.get("ignored_pixels"), bool) or result["ignored_pixels"] < 0:
        raise ValueError("Reference result ignored_pixels is invalid")
    if not isinstance(config.get("sample_keys"), list) or len(config["sample_keys"]) != 1669:
        raise ValueError("Reference lacks the complete locked sample key ordering")
    if not isinstance(config.get("architecture"), dict) or config["architecture"].get("name") != "Q-Lift-DINO-v1":
        raise ValueError("Spatial audit requires a Q-Lift-DINO reference architecture")
    arm = config["architecture"].get("arm")
    if arm not in {"skip", "haar", "image_lift", "query_lift"}:
        raise ValueError(f"Unknown Q-Lift arm in reference: {arm!r}")
    checks = {
        "result.protocol": (result.get("protocol"), protocol),
        "result.model_variant": (result.get("model_variant"), arm),
        "config.model_variant": (config.get("model_variant"), arm),
        "result.weights": (result.get("weights"), config.get("weights")),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Reference contract mismatch for {label}: {actual!r} != {expected!r}")
    if not config.get("weights"):
        raise ValueError("Frozen published CAFe references are outside the Q-Lift spatial audit")
    for name, expected in settings.items():
        for side, record in (("config", config), ("result", result)):
            actual = record.get(name)
            if actual != expected:
                raise ValueError(f"Reference contract mismatch for {side}.{name}: {actual!r} != {expected!r}")
    matrix = np.asarray(result.get("confusion_matrix"), dtype=np.int64)
    classes = len(settings["classes"])
    if matrix.shape != (classes, classes) or np.any(matrix < 0):
        raise ValueError("Reference confusion matrix is invalid for its locked class list")
    validate_native_reference_binding_structure(config)


def validate_q_lift_checkpoint_contract(
    payload: dict[str, Any],
    *,
    reference: dict[str, Any],
    base_checkpoint: Path,
) -> None:
    """Bind a source-selected 2,612-update Q-Lift checkpoint to its reference."""

    if payload.get("format") != "cafe_qlift_v1":
        raise ValueError("Selected checkpoint is not cafe_qlift_v1")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("name") != "Q-Lift-DINO-v1":
        raise ValueError("Selected checkpoint is not Q-Lift-DINO-v1")
    if architecture.get("arm") != reference["model_variant"]:
        raise ValueError("Reference contract mismatch for checkpoint arm")
    if architecture != reference["architecture"]:
        raise ValueError("Reference contract mismatch for checkpoint architecture")
    if payload.get("step") != LOCKED_GATE_STEP:
        raise ValueError("Reference contract mismatch for checkpoint step")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("validation_step") != LOCKED_GATE_STEP:
        raise ValueError("Reference contract mismatch for checkpoint validation step")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("target_data_used") is not False:
        raise ValueError("Selected checkpoint does not attest source-only checkpoint selection")
    recorded = payload.get("base_checkpoint", {}).get("checkpoint")
    if not isinstance(recorded, dict):
        raise ValueError("Selected checkpoint lacks its base checkpoint manifest")
    if base_checkpoint.resolve() != Path(recorded.get("path", "")).resolve() or base_checkpoint.stat().st_size != recorded.get("bytes"):
        raise ValueError("Base CAFe checkpoint differs from the source-selected Q-Lift checkpoint")
