"""Evaluate the fixed regional semantic candidate and equal-information controls."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.region_semantic_readout import (
    IMPLEMENTATION, METHODS, FrozenRegionObserver, RegionReadoutConfig, restricted_pool)
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import summary
from eval_stride_ov_loveda_e1 import make_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("loveda", "udd5", "oem", "vdd",
                        "potsdam", "vaihingen", "landcoverai", "flair1"))
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "semantic-source", "output-dir", "reference-full"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def predict(image, geometry, banks, observer, work):
    height, width = image.shape[-2:]
    coordinates = [(top, left) for top in tile_starts(height, 512, 128)
                   for left in tile_starts(width, 512, 128)]
    snapshot_indices = set(np.linspace(0, len(coordinates)-1, min(2, len(coordinates)), dtype=int))
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    totals, snapshots = {key: {"tiles": 0} for key in banks}, []
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        buffers = {(key, method): stack.enter_context(ProbabilityAccumulator(
            bank.class_count, height, width, 256, work)) for key, bank in banks.items() for method in METHODS}
        for number, (top, left) in enumerate(coordinates):
            rgb = _crop_at(image, top, left, 512).to(geometry.device)
            prepared = geometry.prepare_image(rgb)
            ah, aw = min(512, height-top), min(512, width-left)
            centers = (torch.arange(32, device=geometry.device)+.5)*16
            valid = ((centers[:, None] < ah) & (centers[None] < aw)).reshape(1, -1)
            scores, diagnostics = observer(geometry, prepared, banks, texts, valid, rgb)
            for key, bank in banks.items():
                for method in METHODS:
                    logits = F.interpolate(scores[key][method].transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                           (512, 512), mode="bilinear", align_corners=False)[0]/.07
                    buffers[key, method].add(logits[:, :ah, :aw].softmax(0).cpu().numpy(),
                                             blend[:ah, :aw], left, top)
                totals[key]["tiles"] += 1
                for field, value in diagnostics.items():
                    totals[key][field] = totals[key].get(field, 0.)+value
            if number in snapshot_indices:
                snapshots.append({"top": top, "left": left, **observer.last_audit})
        predictions = {key: {method: buffers[key, method].finalize(None)[0] for method in METHODS} for key in banks}
    return predictions, totals, snapshots


def audit_supports(snapshots, target, key, classes, state):
    for snapshot in snapshots:
        yy = snapshot["top"]+np.arange(32)*16+8
        xx = snapshot["left"]+np.arange(32)*16+8
        inside = (yy[:, None] < target.shape[0]) & (xx[None] < target.shape[1])
        truth = target[yy.clip(0, target.shape[0]-1)[:, None], xx.clip(0, target.shape[1]-1)[None]].reshape(-1)
        valid = inside.reshape(-1) & (truth >= 0) & (truth < classes)
        one_hot = np.eye(classes)[truth.clip(0, classes-1)]*valid[:, None]
        distribution = snapshot["operator"].numpy() @ one_hot
        distribution /= distribution.sum(-1, keepdims=True).clip(1e-12)
        usable = (snapshot["operator"].numpy() @ valid.astype(float)) > 0
        dominant = distribution.argmax(-1)
        observed = snapshot["observations"][key]
        baseline = observed["Geometry"].argmax(-1).numpy()
        for method, scores in observed.items():
            pred = scores.argmax(-1).numpy()
            current = state.setdefault(method, {"supports": 0, "dominant_matches": 0,
                "expected_target_mass_sum": 0., "purity_sum": 0., "beneficial": 0, "harmful": 0,
                "dominant_target_counts": [0]*classes, "dominant_confusion": np.zeros((classes, classes), np.int64)})
            current["supports"] += int(usable.sum())
            current["dominant_matches"] += int(((pred == dominant) & usable).sum())
            current["expected_target_mass_sum"] += float(distribution[np.arange(len(pred)), pred][usable].sum())
            current["purity_sum"] += float(distribution.max(-1)[usable].sum())
            current["beneficial"] += int(((baseline != dominant) & (pred == dominant) & usable).sum())
            current["harmful"] += int(((baseline == dominant) & (pred != dominant) & usable).sum())
            counts = np.bincount(dominant[usable], minlength=classes)
            current["dominant_target_counts"] = (np.asarray(current["dominant_target_counts"])+counts).tolist()
            current["dominant_confusion"] += np.bincount(
                dominant[usable]*classes+pred[usable], minlength=classes**2).reshape(classes, classes)


def serialize_audit(state):
    return {key: {method: {**{name: value.tolist() if isinstance(value, np.ndarray) else value
                              for name, value in values.items()},
                         "mean_target_mass": values["expected_target_mass_sum"]/max(values["supports"], 1),
                         "mean_support_purity": values["purity_sum"]/max(values["supports"], 1)}
                  for method, values in methods.items()} for key, methods in state.items()}


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing output.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    reference = json.loads(Path(args.reference_full).read_text())
    if not reference.get("coverage_verified"):
        raise ValueError("Reference must be coverage verified.")
    if digest([sample.key for sample in samples]) != reference["signature"]["global_sample_keys_sha256"]:
        raise ValueError("Full dataset sequence differs from the verified reference.")
    if args.sample_manifest:
        wanted = json.loads(args.sample_manifest.read_text())["sample_keys"]
        lookup = {sample.key: sample for sample in samples}
        samples = [lookup[key] for key in wanted]
    if args.max_images:
        samples = samples[:args.max_images]
    keys = [sample.key for sample in samples]
    selected = samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("Empty/duplicate sample set.")
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    geometry_config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, geometry_config)
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    counts = {key: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)] for key, bank in banks.items()}
    vocab_sha = hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest()
    if (vocab_sha != reference["signature"]["vocabulary"]["sha256"]
            or any(count != 20 for row in counts.values() for count in row)):
        raise ValueError("Vocabulary differs from the all20 reference.")
    config = RegionReadoutConfig()
    observer = FrozenRegionObserver(args.semantic_source, banks, geometry.device, config)
    # Full-support replay verifies that the trained head is preserved on real features.
    first_image = load_image(selected[0].image_path if args.dataset == "loveda" else selected[0])
    hidden, descriptor = observer.visual(_crop_at(first_image, 0, 0, 512).to(geometry.device))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        replay = restricted_pool(observer.model.vision_model.head, hidden, torch.ones(hidden.shape[:2], device=hidden.device))
    pool_error = float((F.normalize(replay.float(), dim=-1)-descriptor).abs().max())
    if pool_error > 3e-3:
        raise ValueError(f"Trained full-support pooling replay error: {pool_error}")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry_config), "region": asdict(config),
                 "semantic_source": observer.manifest, "primary": "RegionJoint",
                 "semantic_temperature": observer.semantic_temperature,
                 "pooling": "original trained probe/K/V/MLP with log Geometry-support prior",
                 "energy": "KL(p||g)+sum_r n_r KL(q_r||z_r)+sum_ri W_ri KL(p_i||z_r)",
                 "precision": "fp32 frozen weights, bf16 AMP",
                 "text": "same20 aliases/six RS templates; Geometry LME .07; SigLIP frozen trained temperature"},
        "competitive": None, "vocabulary": {"sha256": vocab_sha, "counts": counts,
            "aliases": {key: bank.alias_names for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(keys),
        "global_sample_keys_sha256": digest(keys), "sample_keys": [sample.key for sample in selected],
        "sample_keys_sha256": digest([sample.key for sample in selected]),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reference_full_sample_keys_sha256": reference["signature"]["global_sample_keys_sha256"],
        "note": "Frozen two-encoder exploratory study. All supports, no label-driven selection. "
                "Semantic q is regional class compatibility, not pixel occupancy. Labels enter after full image prediction."}
    output.mkdir(parents=True)
    matrices = {key: {method: np.zeros((bank.class_count,)*2, np.int64) for method in METHODS} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,)*3, np.int64) for method in METHODS[1:]} for key, bank in banks.items()}
    ignored, diagnostics, audit = dict.fromkeys(banks, 0), {key: {} for key in banks}, {key: {} for key in banks}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current, snapshots = predict(image, geometry, banks, observer, output)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count+predictions[key][method][valid]
                matrices[key][method] += np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                if method != "Geometry":
                    transitions[key][method] += transition_counts(predictions[key]["Geometry"], predictions[key][method], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0.)+value
            audit_supports(snapshots, target, key, bank.class_count, audit[key])
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in values.items()}
                        for key, values in matrices.items()},
            "transitions": {key: {method: {"counts": tensor.tolist(), **transition_summary(tensor)} for method, tensor in values.items()}
                            for key, values in transitions.items()},
            "diagnostics": {key: {field: value if field == "tiles" else value/max(values["tiles"], 1)
                                  for field, value in values.items()} for key, values in diagnostics.items()},
            "support_observation_audit": serialize_audit(audit), "trained_pool_replay_max_error": pool_error,
            "wall_seconds": time.perf_counter()-started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
        temporary = output/"results.json.tmp"
        temporary.write_text(json.dumps(result, indent=2)+"\n")
        temporary.replace(output/"results.json")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
            "miou": {key: {method: metric["mean_iou_percent"] for method, metric in methods.items()}
                     for key, methods in result["metrics"].items()}}), flush=True)


if __name__ == "__main__":
    main(parse_args())
