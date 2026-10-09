"""Save fixed native encoder-position readouts before auditing cached windows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.encoder_grounding_observer import (
    IMPLEMENTATION, PRIMARY, METHODS, REVISION, EncoderObservationConfig,
    FrozenEncoderGroundingObserver, native_class_scores, couple_encoder_scores)
from dinotool.gear_ov import _crop_at
from dinotool.prompts import load_class_specs
from eval_cached_localized_likelihood import audit, save_json
from eval_gear_ov import protocol


def predictions(raw, classes):
    probabilities, output = {}, {}
    for method in METHODS:
        if method == "MeanProb_Encoder":
            continue
        value = torch.from_numpy(raw[method])
        dense = F.interpolate(value.T.reshape(1, classes, 32, 32), (512, 512),
                              mode="bilinear", align_corners=False)[0]
        output[method] = dense.argmax(0).numpy()
        if method in ("Geometry", "EncoderGrounding"):
            probabilities[method] = (dense/.07).softmax(0)
    output["MeanProb_Encoder"] = (.5*(probabilities["Geometry"]+
                                      probabilities["EncoderGrounding"])).argmax(0).numpy()
    return output


def coordinate(row):
    return row["sample_key"], row["top"], row["left"]


def protocol_alias_indices(old, key):
    aliases, parents = old["detector_aliases"], old["detector_parents"]
    physical = old["vocabularies"]["D" if old["dataset"] == "loveda" else old["dataset"]]
    bank = old["vocabularies"][key]
    if aliases != physical["aliases"] or parents != np.repeat(
            np.arange(len(physical["classes"])), physical["counts"]).tolist():
        raise ValueError("Physical detector captions changed alias/class identities.")
    lookup = {(physical["classes"][parent], alias): i for i, (parent, alias) in enumerate(zip(parents, aliases))}
    if len(lookup) != len(aliases):
        raise ValueError("Duplicated physical class/alias pair.")
    bank_parents = np.repeat(np.arange(len(bank["classes"])), bank["counts"]).tolist()
    indices = [lookup[bank["classes"][parent], alias] for parent, alias in zip(bank_parents, bank["aliases"])]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicated protocol aliases.")
    return indices, bank_parents


@torch.inference_mode()
def main(args):
    output = args.output_dir
    if output.exists():
        raise ValueError("Refusing existing encoder-position output.")
    source = json.loads((args.source/"verified.json").read_text())
    old = source["signature"]
    detector_source = json.loads((args.encoder_source/"source_manifest.json").read_text())
    if (source["status"] != "complete" or not source["coverage_verified"]
            or old["dataset"] != args.dataset or not old["weights_frozen"]
            or old["target_masks_loaded_during_observation"]
            or detector_source != old["source"] or detector_source["revision"] != REVISION
            or detector_source["status"] != "ready"
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != old["vocabulary_sha256"]
            or any(count != 20 for bank in old["vocabularies"].values() for count in bank["counts"])):
        raise ValueError("Verified unchanged image-only Geometry/alias/source cache required.")
    keys = old["sample_keys"]
    manifests = json.loads((args.source/"raw_manifest.json").read_text())
    identities = [(coordinate(row), row["protocol"]) for row in manifests]
    if (len(keys) != len(set(keys)) or len(identities) != len(set(identities))
            or set(row["sample_key"] for row in manifests) != set(keys)):
        raise ValueError("Incomplete/duplicate original image/window coverage.")
    mappings = {key: protocol_alias_indices(old, key) for key in old["vocabularies"]}
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    if not set(keys) <= set(lookup):
        raise ValueError("Original images unavailable.")
    if args.smoke:
        manifests = manifests[:1]
        keys = [manifests[0]["sample_key"]]
    observer = FrozenEncoderGroundingObserver(args.encoder_source, old["detector_aliases"],
                                              old["detector_parents"], args.device)
    captions = [{"indices": chunk["indices"], "caption": chunk["caption"], "tokens": len(chunk["input_ids"])}
                for chunk in observer.chunks]
    if captions != old["captions"]:
        raise ValueError("Exact original round-robin captions changed.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "sample_keys": keys,
        "methods": METHODS, "encoder_config": EncoderObservationConfig().signature(),
        "encoder_source": detector_source, "source_signature": old, "source_results": source,
        "captions": captions, "candidate_scores_loaded_masks": False,
        "weights_frozen": all(not p.requires_grad for p in observer.model.parameters()),
        "geometry_source": "exact cached original512 geometry logits and relation"}
    output.mkdir(parents=True)
    (output/"scores").mkdir()
    (output/"observations").mkdir()
    save_json(output/"signature.json", {key: value for key, value in signature.items() if key != "source_results"})
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cached_coordinate, cached_probability, cached_support, observation_file = None, None, None, None
    image_key, image = None, None
    observation_seconds, readout_seconds = 0., 0.
    diagnostics = {key: {} for key in old["vocabularies"]}
    observations = []
    for number, row in enumerate(manifests):
        with np.load(args.source/row["file"], allow_pickle=False) as raw:
            g = torch.from_numpy(raw["geometry_logits"].copy()).to(args.device)
            relation = torch.from_numpy(raw["geometry_relation"].copy()).to(args.device)
        height, width = row["image_shape"]
        centers = (torch.arange(32, device=args.device)+.5)*16
        valid = ((centers[:, None] < min(512, height-row["top"])) &
                 (centers[None] < min(512, width-row["left"]))).flatten()
        if cached_coordinate != coordinate(row):
            if image_key != row["sample_key"]:
                sample = lookup[row["sample_key"]]
                image = load_image(sample.image_path if args.dataset == "loveda" else sample)
                image_key = sample.key
            if list(image.shape[-2:]) != row["image_shape"]:
                raise ValueError("Source image dimensions changed.")
            rgb = _crop_at(image, row["top"], row["left"], 512).to(args.device)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            cached_probability, cached_support, diag = observer.observe_encoder(rgb)
            torch.cuda.synchronize()
            observation_seconds += time.perf_counter()-tick
            observation_file = f"observations/w{number:03d}.npz"
            np.savez_compressed(output/observation_file, alias_probability=cached_probability.cpu().numpy(),
                                observed=cached_support.cpu().numpy())
            observations.append({"coordinate": coordinate(row), "file": observation_file, **diag})
            cached_coordinate = coordinate(row)
        indices, parents = mappings[row["protocol"]]
        probability = cached_probability[:, indices]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        b, informative = native_class_scores(probability, parents, g, cached_support)
        scores = {"Geometry": g, "EncoderGrounding": b, "MeanLogit_Encoder": .5*(g+b)}
        permutation = torch.randperm(int(valid.sum()), generator=torch.Generator().manual_seed(20261002+number)).to(args.device)
        active = valid.nonzero().flatten()
        shuffled = relation.clone()
        shuffled[active[:, None], active[None]] = relation[active[permutation]][:, active[permutation]]
        for method, graph in (("Shuffled_EncoderCoupling", shuffled), (PRIMARY, relation)):
            value, diag = couple_encoder_scores(g[None], b[None], graph[None], valid[None], informative[None])
            scores[method] = value[0]
            slot = diagnostics[row["protocol"]].setdefault(method, {})
            for field, entry in diag.items():
                slot[field] = slot.get(field, 0.)+entry
            if not torch.equal(value[0][~informative], g[~informative]):
                raise RuntimeError("Actual-cache no-information fallback changed Geometry.")
        identity, _ = couple_encoder_scores(g[None], g[None], relation[None], valid[None], torch.zeros_like(valid[None]))
        if not torch.equal(identity[0], g) or not np.array_equal(scores["Geometry"].cpu().numpy(), raw_geometry(args.source, row)):
            raise RuntimeError("Original Geometry score/identity changed.")
        torch.cuda.synchronize()
        readout_seconds += time.perf_counter()-tick
        row["scores_file"], row["observation_file"] = f"scores/w{number:03d}.npz", observation_file
        np.savez_compressed(output/row["scores_file"], **{method: value.cpu().numpy() for method, value in scores.items()})
        save_json(output/"collection_status.json", {"status": "collected" if number+1 == len(manifests) else "collecting",
            "processed_windows": number+1, "total_windows": len(manifests), "physical_windows": len(observations),
            "target_masks_loaded": False, "identity_control_exact": True, "original_geometry_scores_exact": True,
            "observation_seconds": observation_seconds, "cached_readout_seconds": readout_seconds,
            "wall_seconds": time.perf_counter()-started, "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576,
            "diagnostics_sum": diagnostics})
        print(f"Collected {args.dataset} window{number+1}/{len(manifests)}; labels still unloaded", flush=True)
    save_json(output/"window_manifest.json", manifests)
    save_json(output/"observation_manifest.json", observations)
    if args.smoke:
        return
    audit(args, signature, manifests, output, methods=METHODS, predictor=predictions,
          implementation=IMPLEMENTATION, primary=PRIMARY,
          note="Window-only development audit of frozen native encoder positions; not full-image metrics. "
               "Additional detection pretraining, no label-selected parameters. Saved all responses/scores before masks.")


def raw_geometry(source, row):
    with np.load(source/row["file"], allow_pickle=False) as raw:
        return raw["geometry_logits"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    for field in ("data-root", "vocabulary-config"):
        parser.add_argument("--"+field, required=True)
    for field in ("source", "encoder-source", "output-dir"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
