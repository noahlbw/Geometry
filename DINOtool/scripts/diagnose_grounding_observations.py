"""Collect frozen local observations before a separate label-only audit phase."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.grounded_local_observer import (
    IMPLEMENTATION, REVISION, GroundingConfig, FrozenGroundingObserver, box_observation)
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol, digest
from eval_geometry_vip_reliability import summary
from eval_stride_ov_loveda_e1 import make_checkpoints


def windows(height, width):
    choices = [(top, left) for top in tile_starts(height, 512, 128)
               for left in tile_starts(width, 512, 128)]
    indices = sorted(set(((len(choices)-1)//3, 2*(len(choices)-1)//3)))
    return [choices[index] for index in indices]


def write_json(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2)+"\n")
    temporary.replace(path)


def predictions_from_raw(raw, parents, classes):
    logits = torch.from_numpy(raw["geometry_logits"])
    geo = F.interpolate(logits.T.reshape(1, classes, 32, 32), (512, 512),
                        mode="bilinear", align_corners=False)[0].argmax(0)
    outputs, covered = {"Geometry": geo.numpy()}, {}
    for method, field in (("BoxMean", "alias_scores"), ("BoxNativeMax", "alias_max_scores")):
        evidence = torch.zeros(1024, classes)
        for index in range(int(raw["chunks"])):
            current = box_observation(torch.from_numpy(raw[f"boxes_{index}"]),
                torch.from_numpy(raw[f"{field}_{index}"]), raw[f"alias_indices_{index}"],
                parents, classes)
            evidence = torch.maximum(evidence, current)
        confidence, proposal = evidence.max(-1)
        has_box = (confidence > 0).reshape(32, 32).repeat_interleave(16, 0).repeat_interleave(16, 1)
        proposal = proposal.reshape(32, 32).repeat_interleave(16, 0).repeat_interleave(16, 1)
        outputs[method] = torch.where(has_box, proposal, geo).numpy()
        covered[method] = has_box.numpy()
    return outputs, covered


def audit(args, samples, specs, load_mask, manifest, output):
    banks = {key: [parent for c, spec in enumerate(classes) for parent in [c]*len(spec.synonyms)]
             for key, classes in specs.items()}
    totals = {key: {method: np.zeros((len(classes),)*2, np.int64)
                    for method in ("Geometry", "BoxMean", "BoxNativeMax")} for key, classes in specs.items()}
    transitions = {key: {method: np.zeros((len(classes),)*3, np.int64)
                         for method in ("BoxMean", "BoxNativeMax")} for key, classes in specs.items()}
    coverage = {key: {method: {"covered": 0, "valid": 0, "disagreements": 0,
                "proposal_correct_disagreements": 0, "geometry_correct_disagreements": 0}
                for method in ("BoxMean", "BoxNativeMax")} for key in specs}
    sample_lookup = {sample.key: sample for sample in samples}
    targets = {}
    for row in manifest:
        sample = sample_lookup[row["sample_key"]]
        key = row["protocol"]
        if (sample.key, key) not in targets:
            targets[sample.key, key] = load_mask(sample, key, tuple(row["image_shape"]))
        target = targets[sample.key, key]
        top, left = row["top"], row["left"]
        crop = np.full((512, 512), -1, np.int64)
        ah, aw = min(512, target.shape[0]-top), min(512, target.shape[1]-left)
        crop[:ah, :aw] = target[top:top+ah, left:left+aw]
        classes = len(specs[key])
        valid = (crop >= 0) & (crop < classes)
        with np.load(output/row["file"], allow_pickle=False) as raw:
            predictions, covered = predictions_from_raw(raw, banks[key], classes)
        for method, prediction in predictions.items():
            encoded = crop[valid]*classes+prediction[valid]
            totals[key][method] += np.bincount(encoded, minlength=classes**2).reshape(classes, classes)
            if method != "Geometry":
                transitions[key][method] += transition_counts(predictions["Geometry"], prediction, crop, classes)
                counts = coverage[key][method]
                disagreement = valid & (prediction != predictions["Geometry"])
                counts["covered"] += int((covered[method] & valid).sum())
                counts["valid"] += int(valid.sum())
                counts["disagreements"] += int(disagreement.sum())
                counts["proposal_correct_disagreements"] += int((disagreement & (prediction == crop)).sum())
                counts["geometry_correct_disagreements"] += int((disagreement & (predictions["Geometry"] == crop)).sum())
    metrics = {key: {method: summary(cm, tuple(spec.name for spec in specs[key]), 0)
                     for method, cm in group.items()} for key, group in totals.items()}
    for group in metrics.values():
        for metric in group.values():
            cm = np.asarray(metric["confusion_matrix"], np.int64)
            for c, entry in enumerate(metric["per_class"]):
                tp = int(cm[c, c])
                entry.update(precision_percent=100*tp/max(int(cm[:, c].sum()), 1),
                             recall_percent=100*tp/max(int(cm[c].sum()), 1))
    result = {"status": "complete", "implementation": IMPLEMENTATION,
        "processed_images": len(samples), "total_images": len(samples),
        "observation_window_count": len(manifest), "signature": json.loads((output/"signature.json").read_text()),
        "metrics": metrics, "coverage": coverage,
        "transitions": {key: {method: {"counts": value.tolist(), **transition_summary(value)}
                              for method, value in group.items()} for key, group in transitions.items()},
        "note": "Window-only observation diagnostic, not full-image dataset mIoU. "
                "Potential duplicate pixels across windows. BoxMean is raw rasterization plus baseline fallback, "
                "not a new coupled model. Detector absence never rejects a class. Labels loaded only after raw collection."}
    result["source_gate_passed"] = all(
        group["BoxMean"]["beneficial"] > group["BoxMean"]["harmful"]
        and coverage[key]["BoxMean"]["proposal_correct_disagreements"] >
            coverage[key]["BoxMean"]["geometry_correct_disagreements"]
        for key, group in result["transitions"].items())
    write_json(output/"results.json", result)
    print(json.dumps({"dataset": args.dataset, "source_gate_passed": result["source_gate_passed"],
                     "window_miou": {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                                     for key, group in metrics.items()}}), flush=True)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing diagnostic output.")
    vocabulary = load_class_specs(args.vocabulary_config)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    samples = [lookup[key] for key in fixed]
    if len(set(fixed)) != len(fixed):
        raise ValueError("Duplicate fixed samples.")
    if args.smoke:
        samples = samples[:1]
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, config)
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    detector_spec = specs["D"] if args.dataset == "loveda" else specs[args.dataset]
    aliases = tuple(alias for spec in detector_spec for alias in spec.synonyms)
    parents = tuple(c for c, spec in enumerate(detector_spec) for alias in spec.synonyms)
    if any(len(spec.synonyms) != 20 for classes in specs.values() for spec in classes):
        raise ValueError("Requires exactly20 unchanged aliases per class.")
    source = json.loads((args.grounding_source/"source_manifest.json").read_text())
    if source["revision"] != REVISION or source["status"] != "ready":
        raise ValueError("Pinned prepared detector required.")
    observer = FrozenGroundingObserver(args.grounding_source, aliases, parents, args.device)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset,
        "sample_keys": [sample.key for sample in samples], "sample_keys_sha256": digest([sample.key for sample in samples]),
        "geometry": asdict(config), "grounding": observer.config.signature(), "source": source,
        "vocabulary_sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "vocabularies": {key: {"classes": bank.class_names, "aliases": bank.alias_names,
            "counts": [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)]}
            for key, bank in banks.items()}, "detector_aliases": aliases, "detector_parents": parents,
        "captions": [{"indices": chunk["indices"], "caption": chunk["caption"],
                      "tokens": len(chunk["input_ids"])} for chunk in observer.chunks],
        "checkpoints": checkpoint_manifest(checkpoints), "target_masks_loaded_during_observation": False,
        "weights_frozen": all(not p.requires_grad for p in backbone.model.parameters()) and
                          all(not p.requires_grad for p in observer.model.parameters())}
    output.mkdir(parents=True)
    (output/"raw").mkdir()
    write_json(output/"signature.json", signature)
    manifests, observer_seconds, geometry_seconds = [], 0., 0.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        for window, (top, left) in enumerate(windows(*image.shape[-2:])[:1 if args.smoke else 2]):
            rgb = _crop_at(image, top, left, 512).to(args.device)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            prepared = geometry.prepare_image(rgb)
            geometry_scores = {key: alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                               bank.parent_indices, bank.class_count)[0].cpu().numpy() for key, bank in banks.items()}
            torch.cuda.synchronize()
            geometry_seconds += time.perf_counter()-tick
            tick = time.perf_counter()
            raw = observer.observe(rgb)
            torch.cuda.synchronize()
            observer_seconds += time.perf_counter()-tick
            for key, bank in banks.items():
                # LoveDA P omits the D background class without changing remaining alias strings.
                alias_lookup = {(detector_spec[parent].name, name): index
                                for index, (parent, name) in enumerate(zip(parents, aliases))}
                detector_to_bank = {alias_lookup[bank.class_names[int(bank.parent_indices[index])], name]: index
                                    for index, name in enumerate(bank.alias_names)}
                payload = {"geometry_logits": geometry_scores[key], "chunks": np.int64(len(raw)),
                           "geometry_relation": prepared.geometry_patch_conditional[0].float().cpu().numpy()}
                for index, chunk in enumerate(raw):
                    active = [i for i, alias in enumerate(chunk["alias_indices"]) if int(alias) in detector_to_bank]
                    payload.update({f"boxes_{index}": chunk["boxes"],
                        f"alias_scores_{index}": chunk["alias_scores"][:, active],
                        f"alias_max_scores_{index}": chunk["alias_max_scores"][:, active],
                        f"alias_indices_{index}": np.asarray([detector_to_bank[int(chunk["alias_indices"][i])]
                                                              for i in active], np.int64)})
                filename = f"raw/i{number-1:03d}_w{window}_{key}.npz"
                np.savez_compressed(output/filename, **payload)
                manifests.append({"sample_key": sample.key, "protocol": key, "top": top, "left": left,
                                  "image_shape": list(image.shape[-2:]), "file": filename})
            del prepared, raw
        write_json(output/"observation_status.json", {"status": "collected" if number == len(samples) else "collecting",
            "processed_images": number, "total_images": len(samples), "raw_windows": len(manifests),
            "target_masks_loaded": False, "geometry_seconds": geometry_seconds,
            "observer_seconds": observer_seconds, "wall_seconds": time.perf_counter()-started,
            "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576})
        print(f"Collected {args.dataset} {number}/{len(samples)}; labels still unloaded", flush=True)
    write_json(output/"raw_manifest.json", manifests)
    if not args.smoke:
        audit(args, samples, specs, load_mask, manifests, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    for field in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--grounding-source", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
