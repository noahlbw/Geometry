"""CPU-only provenance contracts for corrected Q-Lift target evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_module("cafe_qlift_gate", "scripts/analyze_cafe_qlift_gate.py")
evaluator = load_module("cafe_qlift_evaluator", "scripts/eval_cafe_vc_protocols.py")


def native_fixture(root: Path, *, arm: str = "query_lift") -> tuple[Path, Path, dict, list[SimpleNamespace]]:
    checkpoint = root / "corrected.pt"
    checkpoint.write_bytes(b"corrected-checkpoint")
    samples = [SimpleNamespace(key=f"sample-{index}") for index in range(1669)]
    sample_keys = [sample.key for sample in samples]
    architecture = {"name": "Q-Lift-DINO-v1", "arm": arm}
    selected = gate.file_binding(checkpoint)
    native = root / "native"
    native.mkdir()
    config = {
        "protocol": "P", "qlift_audit": "native", "model_variant": arm,
        "target_used_for_training_or_selection": False, "max_images": None,
        "images": 1669, "world_size": 4, "architecture": architecture,
        "sample_keys": sample_keys, "weights": str(checkpoint.resolve()),
        "integrity_bindings": {
            "selected_checkpoint": dict(selected),
            "native_argmax_digests": {"entries": 1669, "sha256": "a" * 64},
            "evaluator_source_sha256": {"project/eval": "b" * 64},
        },
        **gate.LOCKED_LOVEDA_SETTINGS["P"],
    }
    result = {
        "status": "complete", "protocol": "P", "qlift_audit": "native", "model_variant": arm,
        "images": 1669, "world_size": 4, "weights": str(checkpoint.resolve()),
        "mean_iou": 0.5, "ignored_pixels": 0, "confusion_matrix": [[1]],
        **gate.LOCKED_LOVEDA_SETTINGS["P"],
    }
    (native / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
    (native / "results.json").write_text(json.dumps(result), encoding="utf-8")
    return native, checkpoint, architecture, samples


def test_mechanism_reference_requires_exact_native_checkpoint_and_sample_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        native, checkpoint, architecture, samples = native_fixture(root)
        binding = evaluator._verified_mechanism_reference(
            native, protocol="P", samples=samples, weights=checkpoint, architecture=architecture,
        )
        assert binding["selected_checkpoint"]["sha256"] == gate.file_binding(checkpoint)["sha256"]
        config_path = native / "evaluation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["sample_keys"][100] = "wrong-key"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        try:
            evaluator._verified_mechanism_reference(
                native, protocol="P", samples=samples, weights=checkpoint, architecture=architecture,
            )
        except ValueError as error:
            assert "sample ordering" in str(error)
        else:
            raise AssertionError("counterfactual accepted a mismatched native sample ordering")


def test_spatial_audit_requires_hash_binding_to_the_exact_native_reference() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        native, checkpoint, architecture, _ = native_fixture(root, arm="skip")
        source = {"architecture": architecture, "selected_checkpoint": gate.file_binding(checkpoint)}
        native_config, native_result = gate.validate_native_target_evaluation(native, "P", "skip", source)
        spatial = root / "spatial" / "P" / "skip"
        spatial.mkdir(parents=True)
        integrity = {
            "status": "complete", "images": 1669, "components": 1,
            "reference_integrity_bindings_exact_match": True,
            "native_per_image_argmax_digest_exact_match": True,
            "reference_confusion_exact_match": True,
            "per_image_confusion_equals_ddp_reduce": True,
            "ignored_pixels_exact_match": True,
        }
        summary = {
            "global": {"confusion_matrix": [[1]], "mean_iou": 0.5}, "ignored_pixels": 0,
            "gt_boundary": {"pixels": 1, "accuracy": 1.0, "per_class_recall": {"building": 1.0}},
            "gt_interior": {"pixels": 0, "accuracy": None, "per_class_recall": {"building": None}},
            "error_decomposition": {},
            "components": {
                "definition": "test", "all": {},
                "by_area_bin": {"tiny": {}, "small": {}}, "by_class": {},
            },
            "pairwise_confusion": {"global": {"focal_pairs": {}}},
        }
        manifest = {
            "reference_protocol": "P", "reference_model_variant": "skip",
            "reference_evaluation_config_sha256": gate.fingerprint(native / "evaluation_config.json"),
            "reference_results_sha256": gate.fingerprint(native / "results.json"),
            "reference_weights": str(checkpoint.resolve()),
            "reference_weights_sha256": gate.fingerprint(checkpoint),
        }
        (spatial / "integrity.json").write_text(json.dumps(integrity), encoding="utf-8")
        (spatial / "status.json").write_text(json.dumps({"status": "complete", "protocol": "P", "model_variant": "skip"}), encoding="utf-8")
        (spatial / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (spatial / "spatial_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        gate.validate_spatial_audit(root / "spatial", "P", "skip", native, native_config, native_result, source)
        manifest["reference_results_sha256"] = hashlib.sha256(b"wrong-result").hexdigest()
        (spatial / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            gate.validate_spatial_audit(root / "spatial", "P", "skip", native, native_config, native_result, source)
        except ValueError as error:
            assert "not bound to this native result" in str(error)
        else:
            raise AssertionError("spatial audit accepted a mismatched native result hash")
