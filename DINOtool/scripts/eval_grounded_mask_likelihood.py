"""Collect image-only pixel supports and fixed readouts before a label audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.grounded_mask_observer import (
    IMPLEMENTATION, PRIMARY, METHODS, REVISION, MaskObservationConfig, FrozenMaskObserver)
from dinotool.geometry_localized_likelihood import class_localization_density, apply_localized_likelihood
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from eval_cached_localized_likelihood import audit, save_json


def predictions(raw, classes):
    probabilities, output = {}, {}
    for method in METHODS:
        if method == "MeanProb_MaskLikelihood":
            continue
        value = torch.from_numpy(raw[method])
        dense = F.interpolate(value.T.reshape(1, classes, 32, 32), (512, 512),
                              mode="bilinear", align_corners=False)[0]
        probabilities[method] = (dense/.07).softmax(0)
        output[method] = dense.argmax(0).numpy()
    blend = .5*(probabilities["Geometry"]+probabilities["MaskLocalLikelihood"])
    output["MeanProb_MaskLikelihood"] = blend.argmax(0).numpy()
    return output


def load_observation(path, row, device):
    with np.load(path/row["file"], allow_pickle=False) as raw:
        scores = torch.from_numpy(raw["geometry_logits"]).to(device)
        relation = torch.from_numpy(raw["geometry_relation"]).to(device)
        observations = [{name: raw[f"{name}_{chunk}"].copy()
                         for name in ("boxes", "alias_scores", "alias_indices")}
                        for chunk in range(int(raw["chunks"]))]
    return scores, relation, observations


def coordinate(row):
    return row["sample_key"], row["top"], row["left"]


@torch.inference_mode()
def main(args):
    output = args.output_dir
    if output.exists():
        raise ValueError("Refusing existing mask-likelihood output.")
    source = json.loads((args.source/"verified.json").read_text())
    old = source["signature"]
    box_source = json.loads((args.box_source/"verified.json").read_text())
    mask_source = json.loads((args.mask_source/"source_manifest.json").read_text())
    if (not source["coverage_verified"] or source["status"] != "complete"
            or not old["weights_frozen"] or old["target_masks_loaded_during_observation"]
            or old["dataset"] != args.dataset or mask_source["revision"] != REVISION
            or mask_source["status"] != "ready" or box_source["status"] != "complete"
            or not box_source["coverage_verified"] or not box_source["original_geometry_exact"]
            or box_source["signature"]["sample_keys"] != old["sample_keys"]
            or box_source["signature"]["candidate_scores_loaded_masks"]
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != old["vocabulary_sha256"]):
        raise ValueError("Verified unchanged image-only observations required.")
    keys = old["sample_keys"]
    manifests = json.loads((args.source/"raw_manifest.json").read_text())
    if len(keys) != len(set(keys)) or set(coordinate(row)[0] for row in manifests) != set(keys):
        raise ValueError("Incomplete/duplicate fixed sample coverage.")
    if any(count != 20 for bank in old["vocabularies"].values() for count in bank["counts"]):
        raise ValueError("All20 aliases must remain.")
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    if not set(keys) <= set(lookup):
        raise ValueError("Fixed source images unavailable.")
    original = {(coordinate(row), row["protocol"]): row for row in manifests}
    boxes = {(coordinate(row), row["protocol"]): row
             for row in json.loads((args.box_source/"window_manifest.json").read_text())}
    if set(boxes) != set(original):
        raise ValueError("Original box cache and raw window sequence differ.")
    if args.smoke:
        manifests, keys = manifests[:1], keys[:1]
    observer = FrozenMaskObserver(args.mask_source, args.device)
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "sample_keys": keys,
        "methods": METHODS, "mask_config": observer.config.signature(), "mask_source": mask_source,
        "source_signature": old, "source_results": source, "candidate_scores_loaded_masks": False,
        "weights_frozen": all(not p.requires_grad for p in observer.model.parameters()),
        "rule": "Unchanged signed likelihood, temperature .07, pseudocount1, phrase cutoff .25; mask replaces box support only"}
    output.mkdir(parents=True)
    (output/"scores").mkdir()
    (output/"masks").mkdir()
    save_json(output/"signature.json", {key: value for key, value in signature.items() if key != "source_results"})
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cached_coordinate, cached_masks, cached_rows = None, None, None
    image_key, image = None, None
    observation_seconds, readout_seconds, query_count, neutral_count = 0., 0., 0, 0
    for number, row in enumerate(manifests):
        geometry_scores, relation, observations = load_observation(args.source, row, args.device)
        height, width = row["image_shape"]
        centers = (torch.arange(32, device=args.device)+.5)*16
        valid = ((centers[:, None] < min(512, height-row["top"])) &
                 (centers[None] < min(512, width-row["left"]))).reshape(-1)
        if cached_coordinate != coordinate(row):
            if image_key != row["sample_key"]:
                sample = lookup[row["sample_key"]]
                image = load_image(sample.image_path if args.dataset == "loveda" else sample)
                image_key = row["sample_key"]
            rgb = _crop_at(image, row["top"], row["left"], 512).to(args.device)
            mask_row = original.get((coordinate(row), "D"), row)
            _, _, cached_rows = load_observation(args.source, mask_row, args.device)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            cached_masks, diagnostics = observer.observe(rgb, cached_rows, valid)
            torch.cuda.synchronize()
            observation_seconds += time.perf_counter()-tick
            query_count += diagnostics["mask_queries"]
            neutral_count += diagnostics["constant_box_queries"]
            cached_coordinate = coordinate(row)
            mask_file = f"masks/w{number:03d}.npz"
            np.savez_compressed(output/mask_file, **{f"supports_{c}": value for c, value in enumerate(cached_masks)})
        if len(observations) != len(cached_masks) or any(
                not np.array_equal(a["boxes"], b["boxes"]) for a, b in zip(observations, cached_rows)):
            raise ValueError("Protocol changed physical box observations.")
        mask_rows = [{**observation, "supports": support} for observation, support in zip(observations, cached_masks)]
        bank = old["vocabularies"][row["protocol"]]
        classes = len(bank["classes"])
        parents = np.repeat(np.arange(classes), bank["counts"]).tolist()
        permutation = torch.randperm(int(valid.sum()), generator=torch.Generator().manual_seed(20261002+number)).to(args.device)
        indices = valid.nonzero().flatten()
        shuffled = relation.clone()
        shuffled[indices[:, None], indices[None]] = relation[indices[permutation]][:, indices[permutation]]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        values = {"Geometry": geometry_scores}
        for method, rows, graph in (("BoxLocalLikelihood", observations, None),
                ("Geometry_LocalLikelihood", observations, relation), ("MaskLocalLikelihood", mask_rows, None),
                ("ShuffledMaskLikelihood", mask_rows, shuffled), (PRIMARY, mask_rows, relation)):
            density, _ = class_localization_density(rows, parents, classes, valid, graph)
            values[method] = apply_localized_likelihood(geometry_scores, density)
        identity, _ = class_localization_density(mask_rows, parents, classes, valid,
                                                torch.eye(len(valid), device=args.device))
        control, _ = class_localization_density(mask_rows, parents, classes, valid)
        if not torch.equal(identity, control):
            raise RuntimeError("Actual-mask identity Geometry failed to recover mask-only control.")
        previous_row = boxes[coordinate(row), row["protocol"]]
        with np.load(args.box_source/previous_row["scores_file"], allow_pickle=False) as previous:
            for method in ("Geometry", "BoxLocalLikelihood", "Geometry_LocalLikelihood"):
                if not np.array_equal(values[method].cpu().numpy(), previous[method]):
                    raise RuntimeError("Original source score changed: "+method)
        torch.cuda.synchronize()
        readout_seconds += time.perf_counter()-tick
        row["scores_file"], row["mask_file"] = f"scores/w{number:03d}.npz", mask_file
        np.savez_compressed(output/row["scores_file"], **{method: value.cpu().numpy() for method, value in values.items()})
        save_json(output/"collection_status.json", {"status": "collected" if number+1 == len(manifests) else "collecting",
            "processed_windows": number+1, "total_windows": len(manifests), "target_masks_loaded": False,
            "mask_queries": query_count, "constant_box_queries": neutral_count, "identity_control_exact": True,
            "original_source_scores_exact": True, "sam2_seconds": observation_seconds,
            "cached_readout_seconds": readout_seconds, "wall_seconds": time.perf_counter()-started,
            "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576})
        print(f"Collected {args.dataset} window{number+1}/{len(manifests)}, masks still unloaded", flush=True)
    save_json(output/"window_manifest.json", manifests)
    if args.smoke:
        return
    audit(args, signature, manifests, output, methods=METHODS, predictor=predictions, implementation=IMPLEMENTATION)
    result = json.loads((output/"results.json").read_text())
    for key, group in result["metrics"].items():
        for method in ("BoxLocalLikelihood", "Geometry_LocalLikelihood"):
            if not np.array_equal(group[method]["confusion_matrix"], box_source["metrics"][key][method]["confusion_matrix"]):
                raise RuntimeError("Source confusion changed: "+method)
    result.update(primary=PRIMARY, source_predictions_exact=True)
    save_json(output/"results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    for field in ("data-root", "vocabulary-config"):
        parser.add_argument("--"+field, required=True)
    for field in ("source", "box-source", "mask-source", "output-dir"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
