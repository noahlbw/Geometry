#!/usr/bin/env python3
"""Compare support-only native observations with saved deletion measurements."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_support_erasure import erase_supported_rgb, full_text_queries, support_operator
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.support_conditioned_geometry import SupportConfig
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


METHODS = ("GeometrySupport", "NativeGlobal", "NativeKeepMean", "NativeKeepZero",
           "MeanDeletion", "ZeroDeletion", "MeanShapley", "ZeroShapley")
IMPLEMENTATION = "geometry-support-content-observer-diagnostic-v1-20261001"


def alignment(records, names):
    result = {}
    for method in METHODS:
        confusion = np.zeros((len(names), len(names)), np.int64)
        beneficial, harmful, expected_mass = 0, 0, []
        for row in records:
            target = int(np.argmax(row["target_distribution"]))
            baseline = int(np.argmax(row["observations"]["GeometrySupport"]))
            prediction = int(np.argmax(row["observations"][method]))
            confusion[target, prediction] += 1
            beneficial += int(baseline != target and prediction == target)
            harmful += int(baseline == target and prediction != target)
            expected_mass.append(row["target_distribution"][prediction])
        result[method] = {"dominant_matches": int(np.trace(confusion)), "supports": len(records),
                          "mean_target_mass_of_selected_class": float(np.mean(expected_mass)),
                          "beneficial_vs_Geometry": beneficial, "harmful_vs_Geometry": harmful,
                          "dominant_confusion": confusion.tolist()}
    return result


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or args.dataset not in ("vdd", "potsdam") or args.source_diagnostic is None:
        raise ValueError("Requires a new output and the fixed VDD/Potsdam diagnostic.")
    source = args.source_diagnostic
    original = json.loads(source.read_text())
    prior_path = Path(f"results/geometry_support_erasure_snapshot_v2_{args.dataset}_20261001/results.json")
    prior = json.loads(prior_path.read_text())
    if original["status"] != "complete" or prior["status"] != "complete" or prior["snapshots"] != 16:
        raise ValueError("Incomplete original/intervention caches.")
    observations = {(row["sample_key"], row["top"], row["left"]): row for row in prior["records"]}
    samples, specs, load_rgb, _ = protocol(args, load_class_specs(args.vocabulary_config))
    samples = {sample.key: sample for sample in samples}
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    bank = geometry.encode_text(specs[args.dataset])
    if bank.alias_names != tuple(original["signature"]["alias_names"]):
        raise ValueError("The exact all20 alias sequence changed.")
    queries = full_text_queries(backbone, bank)
    text = F.normalize(bank.features.float(), dim=-1)

    def native_scores(rgb):
        return alias_class_scores(geometry.prepare_image(rgb).native_global.float() @ queries.T,
                                  bank.parent_indices, bank.class_count)[0]

    config, records, seen = SupportConfig(), [], set()
    maximum_geometry_error, maximum_native_error = 0., 0.
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for path in sorted((source.parent/"features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = cache["sample_key"], cache["top"], cache["left"]
        if key not in observations or key in seen:
            raise ValueError("Unexpected/duplicate window.")
        seen.add(key)
        image = load_rgb(samples[key[0]])
        rgb = _crop_at(image, key[1], key[2], 512).to(geometry.device)
        yy, xx = key[1]+torch.arange(32, device=geometry.device)*16+8, key[2]+torch.arange(32, device=geometry.device)*16+8
        valid = (yy[:, None] < image.shape[-2]) & (xx[None] < image.shape[-1])
        prepared = geometry.prepare_image(rgb)
        local = alias_class_scores(prepared.geometry_projected.float() @ text.T, bank.parent_indices, bank.class_count)[0]
        error = float((local.cpu()-cache["stage_scores"]["Geometry_original"][0]).abs().max())
        maximum_geometry_error = max(maximum_geometry_error, error)
        if error != 0.:
            raise ValueError(f"Original Geometry replay changed: {error}")
        operator = support_operator(prepared.raw_patch_tokens[0], prepared.geometry_patch_conditional[0], local, valid, config)
        reference = operator @ local
        original_score = alias_class_scores(prepared.native_global.float() @ queries.T, bank.parent_indices, bank.class_count)[0]
        cached = observations[key]["diagnostics"]
        if len(operator) != cached["regions"]:
            raise ValueError("The frozen support count changed.")
        error = float((original_score-torch.tensor(cached["semantic_observations"]["SupportGlobal"][0], device=geometry.device)).abs().max())
        maximum_native_error = max(maximum_native_error, error)
        if error != 0.:
            raise ValueError(f"Original native score replay changed: {error}")
        null_mean = native_scores(erase_supported_rgb(rgb, torch.ones_like(valid, dtype=torch.float32), valid))
        null_zero = native_scores(torch.zeros_like(rgb))
        current = []
        for index, row in enumerate(operator):
            strength = (row/row.max()).reshape(valid.shape)
            pixel_mask = F.interpolate((strength*valid)[None, None].float(), rgb.shape[-2:], mode="nearest")
            keep_mean = native_scores(erase_supported_rgb(rgb, 1-strength, valid))
            keep_zero = native_scores(rgb*pixel_mask)
            drop_zero = native_scores(rgb*(1-pixel_mask))
            drop_mean = torch.tensor(cached["semantic_observations"]["SupportErasedGlobal"][index], device=geometry.device)
            values = {"GeometrySupport": reference[index], "NativeGlobal": original_score,
                      "NativeKeepMean": keep_mean, "NativeKeepZero": keep_zero,
                      "MeanDeletion": original_score-drop_mean, "ZeroDeletion": original_score-drop_zero,
                      "MeanShapley": .5*(original_score-drop_mean+keep_mean-null_mean),
                      "ZeroShapley": .5*(original_score-drop_zero+keep_zero-null_zero)}
            current.append({"sample_key": key[0], "top": key[1], "left": key[2], "region": index,
                            "observations": {method: value.cpu().tolist() for method, value in values.items()}})
        # Targets are accessed only after every native observation is complete.
        truth = cache["target_patch_centers"][0].to(geometry.device)
        scored = valid.reshape(-1) & (truth >= 0) & (truth < bank.class_count)
        one_hot = F.one_hot(truth.clamp(0, bank.class_count-1).long(), bank.class_count).float()*scored[:, None]
        target_mass = operator @ one_hot
        target_mass = target_mass/target_mass.sum(-1, keepdim=True).clamp_min(1e-12)
        for index, row in enumerate(current):
            row["target_distribution"] = target_mass[index].cpu().tolist()
        records.extend(current)
        result = {"status": "complete" if seen == set(observations) else "running", "implementation": IMPLEMENTATION,
                  "dataset": args.dataset, "windows": len(seen), "total_windows": 16, "supports": len(records),
                  "classes": bank.class_names, "alignment": alignment(records, bank.class_names),
                  "maximum_geometry_replay_error": maximum_geometry_error, "maximum_native_replay_error": maximum_native_error,
                  "signature": {"primary_attribution_probe": "ZeroShapley", "config": vars(config),
                                "aliases": bank.alias_names, "source_samples": original["signature"]["samples"],
                                "vocabulary_sha256": original["signature"]["vocabulary_sha256"],
                                "rule": "two-coalition Shapley=.5*(original-drop+keep-null); unit weights, no area division",
                                "note": "All observers use the actual native full trained descriptor, same all20 aliases/six templates, and exactly the same Geometry support/FOV."},
                  "records": records, "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576,
                  "note": "Source diagnostic, not a new final segmentation model or full-image benchmark. No learned parameters, labels in observation, fitted thresholds or domain-specific routing. Influence attribution is not a class posterior; masked inputs can be out of distribution. Mean/zero fills test whether replacement semantics explain the saved failure."}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "windows": len(seen),
                          "dominant_matches": {method: row["dominant_matches"] for method, row in result["alignment"].items()}}), flush=True)
        del cache, prepared, image, rgb
    if seen != set(observations) or len(seen) != 16:
        raise ValueError("Incomplete fixed-window coverage.")


if __name__ == "__main__":
    main(parse_args())
