"""CPU-only contracts for strict Q-Lift post-hoc spatial diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from dinotool.qlift_spatial_audit import (
    NATIVE_ARGMAX_DIGEST_FILENAME,
    NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION,
    NATIVE_EVALUATOR_SOURCE_FILES,
    OFFICIAL_EVALUATOR_SOURCE_FILES,
    area_bin,
    audit_image,
    component_records,
    load_native_argmax_digests,
    merge_native_argmax_digest_shards,
    pairwise_diagnostics,
    native_reference_bindings,
    prediction_digest_mismatches,
    prediction_digest_record,
    sample_key_digest,
    semantic_boundary_mask,
    spatial_summary,
    validate_native_reference_contract,
    validate_native_reference_bindings,
    validate_q_lift_checkpoint_contract,
)


P_NAMES = ("building", "road", "water", "barren", "tree", "farm")


def _sha(value: str = "binding") -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reference(protocol: str = "P") -> tuple[dict, dict]:
    settings = (
        dict(evaluation_size=512, window_size=224, stride=112,
             classes=list(P_NAMES), include_background=False)
        if protocol == "P" else
        dict(evaluation_size=0, window_size=448, stride=224,
             classes=["background", "building", "road", "water", "barren", "forest", "agricultural"],
             include_background=True)
    )
    names = settings["classes"]
    matrix = np.eye(len(names), dtype=np.int64).tolist()
    config = {
        "protocol": protocol,
        **settings,
        "amp": "bf16",
        "model_mode": "eval",
        "world_size": 4,
        "images": 1669,
        "max_images": None,
        "weights": "/tmp/source-selected.pt",
        "model_variant": "query_lift",
        "qlift_audit": "native",
        "target_used_for_training_or_selection": False,
        "sample_keys": [f"sample-{index}" for index in range(1669)],
        "architecture": {"name": "Q-Lift-DINO-v1", "arm": "query_lift"},
        "base_checkpoint": {
            "checkpoint": {"path": "/tmp/base.pth", "bytes": 0},
            "bpe": {"path": "/tmp/bpe.txt", "bytes": 0},
        },
    }
    data_manifest = {
        "data_root": "/tmp/loveda/val",
        "images": 1669,
        "sample_keys_sha256": sample_key_digest(config["sample_keys"]),
        "sample_file_stat_sha256": _sha("sample-stat"),
    }
    data_manifest["files"] = [
        {
            "key": key,
            "image": {"relative_path": f"images/{key}", "bytes": 0, "sha256": _sha(f"image/{key}")},
            "mask": {"relative_path": f"masks/{key}", "bytes": 0, "sha256": _sha(f"mask/{key}")},
        }
        for key in config["sample_keys"]
    ]
    data_manifest["sample_content_sha256"] = hashlib.sha256(
        json.dumps(data_manifest["files"], sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_manifest["sha256"] = hashlib.sha256(
        json.dumps(data_manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config["integrity_bindings"] = {
        "schema_version": "qlift_loveda_native_reference_binding_v1",
        "selected_checkpoint": {"path": config["weights"], "bytes": 0, "sha256": _sha("selected")},
        "base_checkpoint": {"path": "/tmp/base.pth", "bytes": 0, "sha256": _sha("base")},
        "bpe": {"path": "/tmp/bpe.txt", "bytes": 0, "sha256": _sha("bpe")},
        "evaluator_source_sha256": {
            **{f"project/{name}": _sha(f"project/{name}") for name in NATIVE_EVALUATOR_SOURCE_FILES},
            **{f"official/{name}": _sha(f"official/{name}") for name in OFFICIAL_EVALUATOR_SOURCE_FILES},
        },
        "data_manifest": data_manifest,
        "native_argmax_digests": {
            "schema_version": NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION,
            "path": NATIVE_ARGMAX_DIGEST_FILENAME,
            "entries": 1669,
            "sha256": _sha("native-digests"),
        },
    }
    result = {
        "status": "complete",
        "protocol": protocol,
        **settings,
        "images": 1669,
        "weights": config["weights"],
        "world_size": 4,
        "model_variant": "query_lift",
        "qlift_audit": "native",
        "confusion_matrix": matrix,
        "ignored_pixels": 0,
    }
    return config, result


def test_reference_contract_accepts_only_complete_native_full_pd() -> None:
    config, result = _reference("P")
    validate_native_reference_contract(config, result, protocol="P", settings={
        "evaluation_size": 512, "window_size": 224, "stride": 112,
        "classes": list(P_NAMES), "include_background": False,
    })
    result["qlift_audit"] = "details-zero"
    with pytest.raises(ValueError, match="native"):
        validate_native_reference_contract(config, result, protocol="P", settings={
            "evaluation_size": 512, "window_size": 224, "stride": 112,
            "classes": list(P_NAMES), "include_background": False,
        })


def test_reference_contract_requires_result_world_and_ignored_support_equality() -> None:
    config, result = _reference("P")
    result["world_size"] = 1
    with pytest.raises(ValueError, match="world_size"):
        validate_native_reference_contract(config, result, protocol="P", settings={
            "evaluation_size": 512, "window_size": 224, "stride": 112,
            "classes": list(P_NAMES), "include_background": False,
        })
    config, result = _reference("P")
    result.pop("ignored_pixels")
    with pytest.raises(ValueError, match="ignored_pixels"):
        validate_native_reference_contract(config, result, protocol="P", settings={
            "evaluation_size": 512, "window_size": 224, "stride": 112,
            "classes": list(P_NAMES), "include_background": False,
        })


def test_checkpoint_contract_binds_arm_step_source_and_base(tmp_path) -> None:
    base = tmp_path / "base.pth"
    base.write_bytes(b"base")
    config, _ = _reference()
    payload = {
        "format": "cafe_qlift_v1",
        "architecture": config["architecture"],
        "step": 2612,
        "validation": {"validation_step": 2612},
        "source": {"target_data_used": False},
        "base_checkpoint": {"checkpoint": {"path": str(base.resolve()), "bytes": base.stat().st_size}},
    }
    validate_q_lift_checkpoint_contract(payload, reference=config, base_checkpoint=base)
    payload["architecture"] = {"name": "Q-Lift-DINO-v1", "arm": "image_lift"}
    with pytest.raises(ValueError, match="checkpoint arm"):
        validate_q_lift_checkpoint_contract(payload, reference=config, base_checkpoint=base)


def test_boundary_ignores_void_neighbours_and_partitions_scored_pixels() -> None:
    target = np.array([
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 1, 1],
        [255, 255, 1, 1],
    ], dtype=np.uint8)
    boundary = semantic_boundary_mask(target, classes=2)
    # The upper-right foreground-class-0 pixel touches only class 0 and void.
    assert not boundary[0, 1]
    # These pixels meet the real 0/1 semantic interface, including diagonals.
    assert boundary[1, 1]
    assert boundary[2, 2]
    prediction = np.array([
        [0, 1, 7, 7],
        [0, 0, 7, 7],
        [9, 9, 0, 1],
        [9, 9, 1, 1],
    ], dtype=np.int64)
    record, _ = audit_image(prediction, target, ("a", "b"), sample_index=0, key="x")
    full = np.asarray(record["confusion_matrix"])
    split = np.asarray(record["gt_boundary"]["confusion_matrix"]) + np.asarray(record["gt_interior"]["confusion_matrix"])
    np.testing.assert_array_equal(full, split)
    assert record["valid_pixels"] == 8


def test_components_use_eight_connectivity_and_report_coverage() -> None:
    target = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.uint8)
    prediction = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=np.int64)
    records = component_records(prediction, target, ("background", "tree"), sample_index=4, key="diagonal")
    tree = [item for item in records if item["class"] == "tree"]
    assert len(tree) == 1  # diagonal pixels are connected under the pre-registered 8-neighbour rule
    assert tree[0]["area_px"] == 3
    assert tree[0]["same_class_pixels"] == 2
    assert tree[0]["hit_at_50"]
    assert area_bin(0.0005) == "tiny"
    assert area_bin(0.0005001) == "small"
    assert area_bin(0.0051) == "medium"
    assert area_bin(0.051) == "large"


def test_pairwise_registry_has_protocol_specific_semantic_aliases() -> None:
    matrix = np.zeros((len(P_NAMES), len(P_NAMES)), dtype=np.int64)
    tree, farm = P_NAMES.index("tree"), P_NAMES.index("farm")
    matrix[tree, tree] = 90
    matrix[tree, farm] = 10
    matrix[farm, farm] = 80
    matrix[farm, tree] = 20
    report = pairwise_diagnostics(matrix, P_NAMES)
    vegetation = report["focal_pairs"]["vegetation<->agriculture"]
    assert vegetation["first"] == "tree"
    assert vegetation["second"] == "farm"
    assert vegetation["first_to_second_rate"] == pytest.approx(0.10)
    assert vegetation["second_to_first_rate"] == pytest.approx(0.20)
    assert vegetation["mutual_cross_rate"] == pytest.approx(0.15)


def test_spatial_summary_reconstructs_per_image_confusion_and_components() -> None:
    names = ("building", "road")
    target0 = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    prediction0 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    target1 = np.array([[0, 1], [1, 255]], dtype=np.uint8)
    prediction1 = np.array([[0, 1], [0, 7]], dtype=np.int64)
    first, first_components = audit_image(prediction0, target0, names, sample_index=0, key="a")
    second, second_components = audit_image(prediction1, target1, names, sample_index=1, key="b")
    summary = spatial_summary([first, second], first_components + second_components, names, include_background=False)
    expected = np.asarray(first["confusion_matrix"]) + np.asarray(second["confusion_matrix"])
    np.testing.assert_array_equal(np.asarray(summary["global"]["confusion_matrix"]), expected)
    assert summary["global"]["labeled_pixels"] == 7
    assert summary["components"]["all"]["components"] >= 2
    assert "building<->road" in summary["pairwise_confusion"]["global"]["all_pairs"]


def test_native_digest_merge_is_keyed_exact_and_cleans_temporary_shards(tmp_path) -> None:
    keys = ("a.png", "b.png", "c.png")
    predictions = [
        np.array([[0, 1]], dtype=np.int64),
        np.array([[1, 1]], dtype=np.int64),
        np.array([[1, 0]], dtype=np.int64),
    ]
    shard_dir = tmp_path / ".native_argmax_digest_shards.tmp"
    shard_dir.mkdir()
    rows = {
        0: [prediction_digest_record(predictions[0], sample_index=0, key=keys[0]),
            prediction_digest_record(predictions[2], sample_index=2, key=keys[2])],
        1: [prediction_digest_record(predictions[1], sample_index=1, key=keys[1])],
    }
    for rank, records in rows.items():
        (shard_dir / f"rank{rank:02d}.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records), encoding="utf-8",
        )
    published = tmp_path / NATIVE_ARGMAX_DIGEST_FILENAME
    binding = merge_native_argmax_digest_shards(
        shard_dir=shard_dir, output_path=published, sample_keys=keys, world_size=2,
    )
    assert not shard_dir.exists()
    reference = load_native_argmax_digests(tmp_path, binding, keys)
    assert prediction_digest_mismatches(reference, reference, keys) == []
    observed = [dict(record) for record in reference]
    observed[1]["prediction_sha256"] = _sha("different-prediction")
    mismatches = prediction_digest_mismatches(reference, observed, keys)
    assert [(item["sample_index"], item["key"]) for item in mismatches] == [(1, "b.png")]

    bad_dir = tmp_path / ".bad_native_argmax_shards.tmp"
    bad_dir.mkdir()
    (bad_dir / "rank00.jsonl").write_text(json.dumps(reference[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        merge_native_argmax_digest_shards(
            shard_dir=bad_dir, output_path=tmp_path / "bad.jsonl", sample_keys=keys, world_size=1,
        )
    assert not bad_dir.exists()


def test_runtime_reference_bindings_cover_checkpoint_base_bpe_sources_and_manifest(tmp_path) -> None:
    project_root = tmp_path / "project"
    official_root = tmp_path / "official"
    for relative in NATIVE_EVALUATOR_SOURCE_FILES:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative in OFFICIAL_EVALUATOR_SOURCE_FILES:
        path = official_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    data_root = tmp_path / "loveda" / "val"
    data_root.mkdir(parents=True)
    image, mask = data_root / "image.png", data_root / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    sample = SimpleNamespace(key="image.png", image_path=image, mask_path=mask)
    selected, base, bpe = tmp_path / "selected.pt", tmp_path / "base.pth", tmp_path / "bpe.txt"
    selected.write_bytes(b"selected")
    base.write_bytes(b"base")
    bpe.write_bytes(b"bpe")
    bindings = native_reference_bindings(
        project_root=project_root,
        official_root=official_root,
        selected_checkpoint=selected,
        base_checkpoint=base,
        bpe_path=bpe,
        samples=[sample],
        data_root=data_root,
    )
    bindings["native_argmax_digests"] = {
        "schema_version": NATIVE_ARGMAX_DIGEST_SCHEMA_VERSION,
        "path": NATIVE_ARGMAX_DIGEST_FILENAME,
        "entries": 1,
        "sha256": _sha("one-digest"),
    }
    config = {
        "images": 1,
        "sample_keys": [sample.key],
        "weights": str(selected.resolve()),
        "base_checkpoint": {
            "checkpoint": {"path": str(base.resolve()), "bytes": base.stat().st_size},
            "bpe": {"path": str(bpe.resolve()), "bytes": bpe.stat().st_size},
        },
        "integrity_bindings": bindings,
    }
    validate_native_reference_bindings(
        config,
        project_root=project_root,
        official_root=official_root,
        selected_checkpoint=selected,
        base_checkpoint=base,
        bpe_path=bpe,
        samples=[sample],
        data_root=data_root,
    )
    bpe.write_bytes(b"changed-bpe")
    with pytest.raises(ValueError, match="bpe"):
        validate_native_reference_bindings(
            config,
            project_root=project_root,
            official_root=official_root,
            selected_checkpoint=selected,
            base_checkpoint=base,
            bpe_path=bpe,
            samples=[sample],
            data_root=data_root,
        )

    # Same byte count and restored mtime still changes the per-mask SHA-256.
    bpe.write_bytes(b"bpe")
    original_stat = mask.stat()
    mask.write_bytes(b"other")
    os.utime(mask, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="data_manifest"):
        validate_native_reference_bindings(
            config,
            project_root=project_root,
            official_root=official_root,
            selected_checkpoint=selected,
            base_checkpoint=base,
            bpe_path=bpe,
            samples=[sample],
            data_root=data_root,
        )
