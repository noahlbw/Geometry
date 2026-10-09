#!/usr/bin/env python3
"""M0--M2 audit of frozen DINOv3 / CAFe-DINO region readouts.

This is a diagnostic runner, not an OVSS model or a segmentation trainer.  It
rebuilds the locked high-purity region protocol, caches X/Y from the same DINO
forward pass, and compares frozen text/CAFe readouts against source-only visual
comparators.  Target labels are used only for the pre-registered region
manifest and final scoring.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cafedino_locked_loveda import build_model  # noqa: E402
from dino_distribution_controls import (  # noqa: E402
    RegionWindow,
    _rng_seed,
    crop_rgb,
    discover_evaluation,
    discover_source,
    load_loveda_target,
    load_oem_target,
    normalized_tensor,
    sample_windows,
)
from dinotool.distribution_probe import descriptor, fit_token_projection, match_classifier_dimension  # noqa: E402
from dinotool.oem import OEM_CLASSES  # noqa: E402
from dinotool.readout_gap import (  # noqa: E402
    ProbeFit,
    classification_summary,
    error_decomposition,
    fit_logistic_probe,
    native_text_scores,
    nearest_centroid_scores,
    paired_confusions,
    paired_image_bootstrap_delta,
    prediction_from_scores,
    probe_scores,
    region_means,
    token_text_scores,
    top2_margin,
)


PROTOCOLS: dict[str, dict[str, Any]] = {
    "oem4": {
        "classes": ("rangeland", "developed space", "tree", "agriculture land"),
        "evaluation": "oem",
        "historical_output": "distribution_controls_oem4_hardregions_v4_sharedproj",
        "minimum_windows_per_class": 32,
        "expected": {
            "source": {"rangeland": 240, "developed space": 230, "tree": 240, "agriculture land": 240},
            "evaluation": {"rangeland": 72, "developed space": 43, "tree": 59, "agriculture land": 86},
        },
    },
    "transfer5": {
        "classes": ("bareland", "tree", "water", "building", "agriculture land"),
        "evaluation": "loveda",
        "historical_output": "distribution_controls_oem_to_loveda5_v4_sharedproj",
        "minimum_windows_per_class": 64,
        "expected": {
            "source": {"bareland": 76, "tree": 240, "water": 133, "building": 117, "agriculture land": 240},
            "evaluation": {"bareland": 176, "tree": 213, "water": 240, "building": 128, "agriculture land": 240},
        },
    },
}

COMPANION = {
    "name": "transfer5_oem_companion",
    "classes": PROTOCOLS["transfer5"]["classes"],
    "evaluation": "oem",
    "expected": {"bareland": 6, "tree": 59, "water": 17, "building": 24, "agriculture land": 86},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--source-data-root", required=True)
    parser.add_argument("--loveda-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--historical-output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("m0", "m1", "m2", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--allow-missing-source-images", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _csv_dump(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _class_indices(names: Sequence[str]) -> tuple[int, ...]:
    lookup = {spec.name: index for index, spec in enumerate(OEM_CLASSES)}
    return tuple(lookup[name] for name in names)


def _count_windows(windows: Sequence[RegionWindow]) -> dict[str, int]:
    result = {spec.name: 0 for spec in OEM_CLASSES}
    for window in windows:
        result[OEM_CLASSES[window.class_index].name] += 1
    return {name: value for name, value in result.items() if value}


def _manifest_row(protocol: str, cohort: str, index: int, window: RegionWindow) -> dict[str, Any]:
    return {
        "row_id": f"{protocol}:{cohort}:{index:05d}:{window.key}",
        "protocol": protocol,
        "cohort": cohort,
        "dataset_split": window.split,
        "image_id": window.sample.key,
        "image_path": str(window.sample.image_path),
        "mask_path": str(window.sample.mask_path),
        "region_id": window.key,
        "top": int(window.top), "left": int(window.left), "height": 224, "width": 224,
        "class_index": int(window.class_index), "class_name": OEM_CLASSES[window.class_index].name,
        "purity": float(window.purity), "original_image_group": window.sample.key,
    }


def _protocol_windows(args: argparse.Namespace, name: str) -> tuple[list[RegionWindow], list[RegionWindow], tuple[int, ...], tuple[str, ...]]:
    spec = PROTOCOLS[name]
    class_names = tuple(spec["classes"])
    indices = _class_indices(class_names)
    source_args = argparse.Namespace(source_data_root=args.source_data_root, source_train_split="train", max_train_images=None, allow_missing_source_images=args.allow_missing_source_images)
    train_samples = discover_source(source_args)
    train = sample_windows(
        train_samples, split="source_train", target_loader=load_oem_target, class_indices=indices,
        window_size=224, stride=112, minimum_purity=0.85, per_image_class=1, maximum_per_class=240,
        minimum_per_class=int(spec["minimum_windows_per_class"]), seed=args.seed,
    )
    if spec["evaluation"] == "oem":
        eval_samples_args = argparse.Namespace(
            eval_dataset="oem", eval_data_root=args.source_data_root, source_eval_split="val", max_eval_images=None,
            source_data_root=args.source_data_root, loveda_subset="val", allow_missing_source_images=args.allow_missing_source_images,
        )
    else:
        eval_samples_args = argparse.Namespace(
            eval_dataset="loveda", eval_data_root=args.loveda_root, source_eval_split="val", max_eval_images=None,
            source_data_root=args.source_data_root, loveda_subset="val", allow_missing_source_images=args.allow_missing_source_images,
        )
    samples, target_loader, _ = discover_evaluation(eval_samples_args)
    evaluation = sample_windows(
        samples, split="evaluation", target_loader=target_loader, class_indices=indices,
        window_size=224, stride=112, minimum_purity=0.85, per_image_class=1, maximum_per_class=240,
        minimum_per_class=int(spec["minimum_windows_per_class"]), seed=args.seed + 1,
    )
    return train, evaluation, indices, class_names


def _companion_windows(args: argparse.Namespace) -> tuple[list[RegionWindow], list[RegionWindow], tuple[int, ...], tuple[str, ...]]:
    # Source regions are deliberately the same transfer5 source fit, never selected on OEM val.
    train, _, indices, names = _protocol_windows(args, "transfer5")
    eval_args = argparse.Namespace(
        eval_dataset="oem", eval_data_root=args.source_data_root, source_eval_split="val", max_eval_images=None,
        source_data_root=args.source_data_root, loveda_subset="val", allow_missing_source_images=args.allow_missing_source_images,
    )
    samples, target_loader, _ = discover_evaluation(eval_args)
    evaluation = sample_windows(
        samples, split="evaluation", target_loader=target_loader, class_indices=indices,
        window_size=224, stride=112, minimum_purity=0.85, per_image_class=1, maximum_per_class=240,
        minimum_per_class=1, seed=args.seed + 1,
    )
    return train, evaluation, indices, names


def _historical_match(args: argparse.Namespace, protocol: str, evaluation: Sequence[RegionWindow]) -> dict[str, Any]:
    expected_dir = Path(args.historical_output_root) / PROTOCOLS[protocol]["historical_output"]
    archive = expected_dir / "multiclass_predictions.npz"
    if not archive.is_file():
        raise FileNotFoundError(f"Required authoritative historical archive is absent: {archive}")
    historic = np.load(archive)
    labels = np.asarray([window.class_index for window in evaluation], dtype=np.int64)
    groups = np.asarray([window.sample.key for window in evaluation])
    old_labels, old_groups = historic["target"], historic["groups"]
    if not np.array_equal(labels, old_labels) or not np.array_equal(groups, old_groups):
        raise RuntimeError(f"{protocol}: deterministic reconstructed evaluation rows do not match v4 target/group order.")
    return {
        "archive": str(archive), "ordered_target_group_match": True,
        "historical_methods": sorted(key for key in historic.files if key not in {"target", "groups"}),
    }


def stage_m0(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        existing = output / "protocol.json"
        if existing.is_file():
            return json.loads(existing.read_text(encoding="utf-8"))
        raise FileExistsError(f"Output exists but is not a valid audit run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cohorts: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_windows: dict[str, tuple[list[RegionWindow], list[RegionWindow], tuple[int, ...], tuple[str, ...]]] = {}
    for name in PROTOCOLS:
        train, evaluation, indices, class_names = _protocol_windows(args, name)
        all_windows[name] = (train, evaluation, indices, class_names)
        expected = PROTOCOLS[name]["expected"]
        if _count_windows(train) != expected["source"] or _count_windows(evaluation) != expected["evaluation"]:
            raise RuntimeError(f"{name}: reconstructed counts differ from locked v4 protocol.")
        overlap = set(window.sample.key for window in train) & set(window.sample.key for window in evaluation)
        if overlap:
            raise RuntimeError(f"{name}: source/evaluation images overlap ({len(overlap)}).")
        historical = _historical_match(args, name, evaluation)
        cohorts[name] = {
            "class_names": list(class_names), "class_indices": list(indices),
            "source_count_by_class": _count_windows(train), "evaluation_count_by_class": _count_windows(evaluation),
            "source_regions": len(train), "evaluation_regions": len(evaluation),
            "source_images": len({window.sample.key for window in train}),
            "evaluation_images": len({window.sample.key for window in evaluation}),
            "source_evaluation_image_overlap": 0, "historical": historical,
        }
        all_rows.extend(_manifest_row(name, "source", index, window) for index, window in enumerate(train))
        all_rows.extend(_manifest_row(name, "evaluation", index, window) for index, window in enumerate(evaluation))
    train, companion_eval, indices, names = _companion_windows(args)
    actual = _count_windows(companion_eval)
    if actual != COMPANION["expected"]:
        raise RuntimeError(f"Companion OEM support changed: {actual} != {COMPANION['expected']}")
    cohorts[COMPANION["name"]] = {
        "class_names": list(names), "class_indices": list(indices), "source_protocol": "transfer5",
        "source_regions": len(train), "evaluation_regions": len(companion_eval),
        "evaluation_count_by_class": actual, "evaluation_images": len({window.sample.key for window in companion_eval}),
        "support_limited": True,
    }
    all_rows.extend(_manifest_row(COMPANION["name"], "evaluation", index, window) for index, window in enumerate(companion_eval))
    with (output / "region_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    protocol = {
        "status": "M0 complete", "purpose": "Frozen region-level readout diagnosis; not segmentation mIoU or OVSS model training.",
        "locked": {"checkpoint": str(args.checkpoint), "crop": "224x224 RGB, no resize", "patch": "16 pixels / 14x14 tokens", "stride": 112, "purity": 0.85, "seed": args.seed, "source_only_fitting": True},
        "cohorts": cohorts, "region_manifest_rows": len(all_rows), "region_manifest_sha256": _digest(row["row_id"] for row in all_rows),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
    }
    _json_dump(output / "protocol.json", protocol)
    return protocol


def _windows_for_manifest(args: argparse.Namespace) -> dict[str, tuple[list[RegionWindow], list[RegionWindow], tuple[int, ...], tuple[str, ...]]]:
    result = {name: _protocol_windows(args, name) for name in PROTOCOLS}
    transfer = result["transfer5"]
    _, companion, indices, names = _companion_windows(args)
    result[COMPANION["name"]] = (transfer[0], companion, indices, names)
    return result


def _model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_args = argparse.Namespace(
        official_root=args.official_root, checkpoint=args.checkpoint, bpe_path=args.bpe_path,
        device=args.device, model_mode="eval", window_size=224,
    )
    model, _, checkpoint = build_model(model_args)
    return model, checkpoint


@torch.inference_mode()
def _extract_cohort(model: torch.nn.Module, windows: Sequence[RegionWindow], cache: Path, *, batch_size: int) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    x_path, y_path = cache / "X.npy", cache / "Y.npy"
    if x_path.is_file() or y_path.is_file():
        raise FileExistsError(f"Feature cache already exists and will not be overwritten: {cache}")
    device = next(model.parameters()).device
    x_mem: np.memmap | None = None
    y_mem: np.memmap | None = None
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        images = torch.stack([normalized_tensor(crop_rgb(window, 224)) for window in batch_windows]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, y, x = model.backbone.encode_image_with_patch_tokens(images, normalize=False)
        if x.shape[1] != 196 or y.shape[1] != 196:
            raise RuntimeError(f"Expected 14x14 token grids, got X={tuple(x.shape)}, Y={tuple(y.shape)}")
        if x_mem is None:
            x_mem = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(len(windows), *x.shape[1:]))
            y_mem = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(len(windows), *y.shape[1:]))
        x_mem[start : start + len(batch_windows)] = x.float().cpu().numpy()
        y_mem[start : start + len(batch_windows)] = y.float().cpu().numpy()
        if start == 0 or start + len(batch_windows) == len(windows) or (start // batch_size + 1) % 20 == 0:
            print(json.dumps({"feature_extract": str(cache), "completed": start + len(batch_windows), "total": len(windows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    assert x_mem is not None and y_mem is not None
    x_mem.flush(); y_mem.flush()
    return {"X": {"path": str(x_path), "shape": list(x_mem.shape), "dtype": str(x_mem.dtype)}, "Y": {"path": str(y_path), "shape": list(y_mem.shape), "dtype": str(y_mem.dtype)}, "seconds": time.perf_counter() - started}


@torch.inference_mode()
def _m1_smoke(model: torch.nn.Module, windows: Sequence[RegionWindow], class_names: Sequence[str]) -> dict[str, Any]:
    if len(windows) < 2:
        raise RuntimeError("M1 requires at least two source regions.")
    device = next(model.parameters()).device
    images = torch.stack([normalized_tensor(crop_rgb(window, 224)) for window in windows[:2]]).to(device)
    text = model.build_text_embeddings([list(class_names)]).to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        _, y, x = model.backbone.encode_image_with_patch_tokens(images, normalize=False)
        x_via_cafe = model.encode_patches(images)
        batch_logits = model(images, text, pre_text_emb=True)
        single = torch.cat([model(images[index : index + 1], text, pre_text_emb=True) for index in range(2)], dim=0)
    if x.shape[-1] != text.shape[-1] or y.shape[-1] != text.shape[-1]:
        raise RuntimeError(f"Text and X/Y spaces do not share expected dimensions: {x.shape}, {y.shape}, {text.shape}")
    x_error = float((x.float() - x_via_cafe.float()).abs().max().item())
    cafe_batch_error = float((batch_logits.float() - single.float()).abs().max().item())
    batch_probability = torch.softmax(batch_logits.float(), dim=1)
    single_probability = torch.softmax(single.float(), dim=1)
    probability_error = float((batch_probability - single_probability).abs().max().item())
    prediction_agreement = float((batch_probability.argmax(dim=1) == single_probability.argmax(dim=1)).float().mean().item())
    if x_error > 1e-4 or prediction_agreement != 1.0:
        raise RuntimeError(
            "M1 route check failed: CAFe input X must match and the tested batch route must preserve every pixel label; "
            f"X={x_error}, logit={cafe_batch_error}, probability={probability_error}, agreement={prediction_agreement}"
        )
    return {
        "X_shape": list(x.shape), "Y_shape": list(y.shape), "text_shape": list(text.shape),
        "cafe_input_max_abs_error": x_error, "cafe_batch_max_abs_logit_error": cafe_batch_error,
        "cafe_batch_max_abs_probability_error": probability_error, "cafe_batch_pixel_prediction_agreement": prediction_agreement,
        "batch_decision": "fixed batch route accepted only after exact pixel-label agreement; logit drift is retained as a numerical audit field",
    }


def stage_m1(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if not (output / "protocol.json").is_file():
        raise FileNotFoundError("M0 protocol.json is required before M1.")
    windows = _windows_for_manifest(args)
    model, checkpoint = _model(args)
    first = windows["oem4"]
    smoke = _m1_smoke(model, first[0], first[3])
    cache_root = output / "feature_cache"
    features: dict[str, Any] = {"checkpoint": checkpoint, "smoke": smoke, "cohorts": {}}
    for name, (train, evaluation, _, _) in windows.items():
        source_key = "transfer5_source" if name == COMPANION["name"] else f"{name}_source"
        if source_key not in features["cohorts"]:
            features["cohorts"][source_key] = _extract_cohort(model, train, cache_root / source_key, batch_size=args.batch_size)
        features["cohorts"][f"{name}_evaluation"] = _extract_cohort(model, evaluation, cache_root / f"{name}_evaluation", batch_size=args.batch_size)
    _json_dump(output / "feature_manifest.json", features)
    return features


def _load_tokens(feature_manifest: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    metadata = feature_manifest["cohorts"][key]
    return np.load(metadata["X"]["path"], mmap_mode="r"), np.load(metadata["Y"]["path"], mmap_mode="r")


@torch.inference_mode()
def _cafe_scores(model: torch.nn.Module, windows: Sequence[RegionWindow], class_names: Sequence[str], *, batch_size: int) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    text = model.build_text_embeddings([list(class_names)]).to(device)
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    votes: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        subset = windows[start : start + batch_size]
        images = torch.stack([normalized_tensor(crop_rgb(window, 224)) for window in subset]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            dense_logits = model(images, text, pre_text_emb=True)
        dense_float = dense_logits.float()
        dense_probability = torch.softmax(dense_float, dim=1)
        probabilities.append(dense_probability.mean(dim=(2, 3)).cpu().numpy())
        logits.append(dense_float.mean(dim=(2, 3)).cpu().numpy())
        vote = dense_probability.argmax(dim=1)
        votes.append(torch.stack([(vote == class_index).float().mean(dim=(1, 2)) for class_index in range(len(class_names))], dim=1).cpu().numpy())
        if start == 0 or start + len(subset) == len(windows) or (start // batch_size + 1) % 20 == 0:
            print(json.dumps({"cafe_forward": start + len(subset), "total": len(windows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    return {"mean_probability": np.concatenate(probabilities), "mean_logit": np.concatenate(logits), "vote_fraction": np.concatenate(votes)}


def _legacy_predict(source_x: np.ndarray, source_labels: np.ndarray, eval_x: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def restore_grid(tokens: np.ndarray) -> np.ndarray:
        array = np.asarray(tokens, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 196:
            raise ValueError(f"Legacy descriptor requires exactly 196 flattened tokens, got {array.shape}.")
        return array.reshape(14, 14, array.shape[1])

    source_grids = [restore_grid(tokens) for tokens in source_x]
    evaluation_grids = [restore_grid(tokens) for tokens in eval_x]
    projection = fit_token_projection(source_grids, dimension=32, maximum_tokens=100000, seed=seed)
    train_values = np.stack([descriptor("mean", projection.transform(grid), covariance_dimension=12, radial_bins=3) for grid in source_grids])
    eval_values = np.stack([descriptor("mean", projection.transform(grid), covariance_dimension=12, radial_bins=3) for grid in evaluation_grids])
    feature_seed = _rng_seed(seed, "classifier-feature-map", train_values.shape[1])
    train_matched = match_classifier_dimension(train_values, output_dimension=128, seed=feature_seed)
    eval_matched = match_classifier_dimension(eval_values, output_dimension=128, seed=feature_seed)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    return classifier.fit(StandardScaler().fit_transform(train_matched), source_labels).predict(StandardScaler().fit(train_matched).transform(eval_matched)).astype(np.int64)


def _save_fit(path: Path, fit: ProbeFit, *, fit_row_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, classes=fit.classes, mean=fit.mean, scale=fit.scale, coefficients=fit.coefficients, intercept=fit.intercept, fit_row_ids=np.asarray(fit_row_ids), c_value=fit.c_value, seed=fit.seed, effective_rank=fit.effective_rank)


def _source_bootstrap_rows(groups: Sequence[str], *, seed: int) -> np.ndarray:
    group_array = np.asarray(groups)
    unique = np.unique(group_array)
    picked = np.random.default_rng(seed).choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(group_array == item) for item in picked])


def _method_payload(scores: np.ndarray, class_indices: Sequence[int]) -> dict[str, Any]:
    return {"scores": scores, "prediction": prediction_from_scores(scores, class_indices), "margin": top2_margin(scores)}


def _evaluate_one(
    args: argparse.Namespace, output: Path, feature_manifest: dict[str, Any], name: str,
    train: Sequence[RegionWindow], evaluation: Sequence[RegionWindow], class_indices: Sequence[int], class_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    source_key = "transfer5_source" if name == COMPANION["name"] else f"{name}_source"
    source_x, source_y = _load_tokens(feature_manifest, source_key)
    eval_x, eval_y = _load_tokens(feature_manifest, f"{name}_evaluation")
    source_labels = np.asarray([window.class_index for window in train], dtype=np.int64)
    target = np.asarray([window.class_index for window in evaluation], dtype=np.int64)
    groups = np.asarray([window.sample.key for window in evaluation])
    source_ids = [f"{name}:source:{index:05d}:{window.key}" for index, window in enumerate(train)]
    class_indices_array = np.asarray(class_indices, dtype=np.int64)
    model, _ = _model(args)
    with torch.inference_mode():
        text = model.build_text_embeddings([list(class_names)]).float().cpu().numpy()
        cafe = _cafe_scores(model, evaluation, class_names, batch_size=args.batch_size)
    source_x_mean, source_y_mean = region_means(source_x), region_means(source_y)
    eval_x_mean, eval_y_mean = region_means(eval_x), region_means(eval_y)
    methods: dict[str, dict[str, Any]] = {
        "A-native": _method_payload(native_text_scores(eval_y_mean, text), class_indices),
        "A-cafe-input": _method_payload(token_text_scores(eval_x, text), class_indices),
        "A-pooling-control": _method_payload(native_text_scores(eval_x_mean, text), class_indices),
        "B-prob": _method_payload(cafe["mean_probability"], class_indices),
        "B-vote": _method_payload(cafe["vote_fraction"], class_indices),
        "B-mean-logit": _method_payload(cafe["mean_logit"], class_indices),
    }
    centroid_x, _ = nearest_centroid_scores(source_x_mean, source_labels, eval_x_mean, class_indices)
    centroid_y, _ = nearest_centroid_scores(source_y_mean, source_labels, eval_y_mean, class_indices)
    methods["C-X"] = _method_payload(centroid_x, class_indices)
    methods["C-Y"] = _method_payload(centroid_y, class_indices)
    fits: dict[str, ProbeFit] = {}
    for space, source_mean, eval_mean in (("X", source_x_mean, eval_x_mean), ("Y", source_y_mean, eval_y_mean)):
        for c_value in (1.0, 0.01, 0.1, 10.0):
            key = f"D-{space}" if c_value == 1.0 else f"E-{space}-C{c_value:g}"
            fit = fit_logistic_probe(source_mean, source_labels, c_value=c_value, seed=args.seed)
            fits[key] = fit
            methods[key] = _method_payload(probe_scores(fit, eval_mean), fit.classes)
            _save_fit(output / "source_fits" / f"{name}_{key.replace('-', '_')}.npz", fit, fit_row_ids=source_ids)
    # Predeclared source-image bootstrap refits diagnose fit stability but never select a target winner.
    refit_rows: list[dict[str, Any]] = []
    source_groups = [window.sample.key for window in train]
    for refit_index, refit_seed in enumerate((args.seed + 101, args.seed + 202, args.seed + 303), start=1):
        rows = _source_bootstrap_rows(source_groups, seed=refit_seed)
        for space, source_mean, eval_mean in (("X", source_x_mean, eval_x_mean), ("Y", source_y_mean, eval_y_mean)):
            fit = fit_logistic_probe(source_mean[rows], source_labels[rows], c_value=1.0, seed=refit_seed)
            prediction = prediction_from_scores(probe_scores(fit, eval_mean), fit.classes)
            refit_rows.append({"protocol": name, "method": f"D-{space}", "source_refit": refit_index, **classification_summary(target, prediction, class_indices)})
    legacy = _legacy_predict(source_x, source_labels, eval_x, seed=args.seed)
    methods["D-legacy"] = {"scores": np.empty((len(target), 0), dtype=np.float32), "prediction": legacy, "margin": np.full(len(target), np.nan, dtype=np.float32)}
    historical_reproduced: bool | None = None
    if name in PROTOCOLS:
        historic = np.load(Path(args.historical_output_root) / PROTOCOLS[name]["historical_output"] / "multiclass_predictions.npz")
        historical_reproduced = bool(np.array_equal(legacy, historic["dinov3_mean"]))
        if not historical_reproduced:
            raise RuntimeError(f"{name}: legacy DINO mean predictions did not reproduce v4 exactly.")
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for method, payload in methods.items():
        summary = classification_summary(target, payload["prediction"], class_indices)
        fit = fits.get(method)
        summary_rows.append({
            "protocol": name, "feature_space": "Y" if method in {"A-native", "C-Y", "D-Y"} or method.startswith("E-Y") else "X" if method not in {"B-prob", "B-vote", "B-mean-logit"} else "CAFe",
            "method": method, "reference_probe": "", "support": len(target), "num_images": len(set(groups)),
            "balanced_acc": summary["balanced_accuracy_percent"], "macro_f1": summary["macro_f1_percent"], "accuracy": summary["accuracy_percent"],
            "parameter_count": None if fit is None else fit.parameter_count, "effective_rank": None if fit is None else fit.effective_rank,
            "source_refit": 0, "candidate_set_id": "|".join(class_names), "legacy_reproduced": historical_reproduced if method == "D-legacy" else None,
        })
        for label, recall in summary["per_class_recall_percent"].items():
            per_class_rows.append({"protocol": name, "method": method, "class_index": int(label), "class_name": OEM_CLASSES[int(label)].name, "recall_percent": recall, "support": int(np.sum(target == int(label)))})
    summary_rows.extend(refit_rows)
    confusion_rows: list[dict[str, Any]] = []
    comparison_pairs = (("A-native", "D-Y"), ("A-cafe-input", "D-X"), ("B-prob", "D-X"), ("B-prob", "D-Y"))
    paired_details: dict[str, Any] = {}
    adjusted_alpha = 0.05 / 4.0
    for comparison_index, (open_method, visual_method) in enumerate(comparison_pairs):
        error_summary, error_per_class, kinds = error_decomposition(target, methods[open_method]["prediction"], methods[visual_method]["prediction"], class_indices)
        bootstrap = paired_image_bootstrap_delta(target, methods[open_method]["prediction"], methods[visual_method]["prediction"], groups, class_indices, samples=args.bootstrap_samples, seed=args.seed + comparison_index, alpha=adjusted_alpha)
        paired_details[f"{open_method}__{visual_method}"] = {"error": error_summary, "bootstrap_bonferroni_family4": bootstrap}
        for row in error_per_class:
            per_class_rows.append({"protocol": name, "method": open_method, "reference_probe": visual_method, "class_name": OEM_CLASSES[row["class_index"]].name, **row})
        for row in paired_confusions(target, methods[open_method]["prediction"], methods[visual_method]["prediction"], class_indices):
            confusion_rows.append({"protocol": name, "method": open_method, "reference_probe": visual_method, **row})
        for existing in summary_rows:
            if existing.get("method") == open_method and existing.get("source_refit", 0) == 0:
                existing[f"vs_{visual_method}_rer"] = error_summary["balanced_rer"]
                existing[f"vs_{visual_method}_reverse_loss"] = error_summary["balanced_reverse_loss"]
                existing[f"vs_{visual_method}_net_BA_gap"] = error_summary["balanced_net_accuracy_gap_percent"]
                existing[f"vs_{visual_method}_BA_delta_ci_low"] = bootstrap["ci95_low_points"]
                existing[f"vs_{visual_method}_BA_delta_ci_high"] = bootstrap["ci95_high_points"]
    per_region: dict[str, np.ndarray] = {"target": target, "groups": groups}
    for method, payload in methods.items():
        per_region[f"{method}__prediction"] = payload["prediction"]
        per_region[f"{method}__margin"] = payload["margin"]
        if payload["scores"].shape[1]:
            per_region[f"{method}__scores"] = payload["scores"]
    return summary_rows, per_class_rows, confusion_rows, per_region, {"paired": paired_details, "methods": methods}


def stage_m2(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    protocol_path, feature_path = output / "protocol.json", output / "feature_manifest.json"
    if not protocol_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError("M0 protocol and M1 feature manifests are required before M2.")
    feature_manifest = json.loads(feature_path.read_text(encoding="utf-8"))
    windows = _windows_for_manifest(args)
    all_summary: list[dict[str, Any]] = []
    all_per_class: list[dict[str, Any]] = []
    all_confusion: list[dict[str, Any]] = []
    all_scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, (train, evaluation, indices, class_names) in windows.items():
        summary, per_class, confusion, scores, details = _evaluate_one(args, output, feature_manifest, name, train, evaluation, indices, class_names)
        all_summary.extend(summary); all_per_class.extend(per_class); all_confusion.extend(confusion)
        all_scores.update({f"{name}__{key}": value for key, value in scores.items()})
        diagnostics[name] = details["paired"]
    _csv_dump(output / "summary.csv", all_summary)
    _csv_dump(output / "per_class.csv", all_per_class)
    _csv_dump(output / "confusion_pairs.csv", all_confusion)
    np.savez_compressed(output / "scores.npz", **all_scores)
    _json_dump(output / "m2_diagnostics.json", diagnostics)
    # Write one auditable row per region and preserve opaque score tensors in scores.npz.
    region_rows: list[dict[str, Any]] = []
    for name, (train, evaluation, indices, class_names) in windows.items():
        prefix = f"{name}__"
        for index, window in enumerate(evaluation):
            row: dict[str, Any] = {"protocol": name, "image_id": window.sample.key, "region_id": window.key, "GT_class": OEM_CLASSES[window.class_index].name, "purity": window.purity}
            for key, values in all_scores.items():
                if key.startswith(prefix) and key.endswith("__prediction"):
                    method = key[len(prefix) : -len("__prediction")]
                    row[f"{method}_pred"] = OEM_CLASSES[int(values[index])].name
                    margin = all_scores.get(f"{prefix}{method}__margin")
                    row[f"{method}_margin"] = None if margin is None or np.isnan(margin[index]) else float(margin[index])
            region_rows.append(row)
    _csv_dump(output / "per_region.csv", region_rows)
    primary: list[dict[str, Any]] = []
    for name in PROTOCOLS:
        for pair in ("A-native__D-Y", "A-cafe-input__D-X"):
            values = diagnostics[name][pair]
            boot = values["bootstrap_bonferroni_family4"]
            primary.append({"protocol": name, "contrast": pair, "BA_delta_points": boot["mean_delta_points"], "family_adjusted_ci_low": boot["ci95_low_points"], "family_adjusted_ci_high": boot["ci95_high_points"], "passes_3pt_positive_ci": bool(boot["mean_delta_points"] >= 3.0 and boot["ci95_low_points"] > 0.0)})
    _csv_dump(output / "gate_g1.csv", primary)
    report = ["# M0--M2 readout-gap audit", "", "All values are regional classification percentages, not segmentation mIoU.", "", "## G1 pre-registered primary contrasts", ""]
    for row in primary:
        report.append(f"- {row['protocol']} {row['contrast']}: ΔBA={row['BA_delta_points']:.3f}, family-adjusted CI [{row['family_adjusted_ci_low']:.3f}, {row['family_adjusted_ci_high']:.3f}], gate={row['passes_3pt_positive_ci']}")
    report.extend(["", "Geometry and class-holdout are not run by this M0--M2 command. Proceed only if the gate rule and stability panels support it."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"status": "M2 complete", "primary_gate": primary, "output": str(output)}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("batch-size and bootstrap-samples must be positive.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir).resolve()
    started = time.perf_counter()
    if args.stage in {"m0", "all"}:
        stage_m0(args, output)
    if args.stage in {"m1", "all"}:
        stage_m1(args, output)
    if args.stage in {"m2", "all"}:
        result = stage_m2(args, output)
    else:
        result = {"status": f"{args.stage} complete", "output": str(output)}
    result["wall_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
