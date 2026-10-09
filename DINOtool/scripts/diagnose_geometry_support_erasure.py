#!/usr/bin/env python3
"""Fixed-window support-erasure observation; labels enter only the final audit."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_support_erasure import IMPLEMENTATION, METHODS, full_text_queries, read_support_erasure
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.support_conditioned_geometry import SupportConfig
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import summary
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def main(args):
    if args.dataset not in ("vdd", "potsdam") or not args.source_diagnostic:
        raise ValueError("Requires the fixed VDD/Potsdam original readout diagnostic.")
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing outputs.")
    source = Path(args.source_diagnostic)
    original = json.loads(source.read_text())
    if original["status"] != "complete" or original["processed_images"] != 8:
        raise ValueError("Incomplete reference diagnostic.")
    expected = {(record["sample_key"], top, left) for record in original["images"]
                for top, left in record["saved_windows"]}
    if len(expected) != 16:
        raise ValueError("Expected 16 fixed windows.")
    samples, specs, load_rgb, _ = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    bank = geometry.encode_text(specs[args.dataset])
    if bank.alias_names != tuple(original["signature"]["alias_names"]):
        raise ValueError("Alias strings/order changed.")
    queries = full_text_queries(backbone, bank)
    config = SupportConfig()
    matrices = {method: np.zeros((bank.class_count,)*2, np.int64) for method in METHODS}
    transitions = {method: np.zeros((bank.class_count,)*3, np.int64) for method in METHODS if method != "Geometry"}
    records, seen = [], set()
    maximum_control_error = 0.
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for path in sorted((source.parent/"features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = (cache["sample_key"], cache["top"], cache["left"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected/duplicate window.")
        seen.add(key)
        image = load_rgb(lookup[key[0]])
        rgb = _crop_at(image, key[1], key[2], 512).to(geometry.device)
        yy, xx = key[1]+torch.arange(32, device=geometry.device)*16+8, key[2]+torch.arange(32, device=geometry.device)*16+8
        valid = ((yy[:, None] < image.shape[-2]) & (xx[None] < image.shape[-1])).reshape(1, -1)
        prepared = geometry.prepare_image(rgb)
        scores, diagnostics = read_support_erasure(geometry, prepared, bank, queries, rgb, valid, config)
        error = float((scores["Geometry"].cpu()-cache["stage_scores"]["Geometry_original"]).abs().max())
        maximum_control_error = max(error, maximum_control_error)
        if error != 0.:
            raise ValueError(f"Original Geometry cache replay changed: {error}")
        predictions = {method: values[0].argmax(-1).cpu().numpy() for method, values in scores.items()}
        # Inference above receives only RGB, original Geometry and text queries.
        truth = cache["target_patch_centers"][0].numpy()
        geometric_valid = valid[0].cpu().numpy()
        audit_valid = geometric_valid & (truth >= 0) & (truth < bank.class_count)
        for method, prediction in predictions.items():
            matrices[method] += np.bincount(truth[audit_valid].astype(np.int64)*bank.class_count+prediction[audit_valid],
                                            minlength=bank.class_count**2).reshape(bank.class_count, -1)
            if method != "Geometry":
                transitions[method] += transition_counts(predictions["Geometry"], prediction,
                                                         np.where(audit_valid, truth, 255), bank.class_count)
        records.append({"sample_key": key[0], "top": key[1], "left": key[2], "diagnostics": diagnostics})
        result = {"status": "complete" if seen == expected else "running", "snapshots": len(records),
                  "total_snapshots": 16, "dataset": args.dataset, "implementation": IMPLEMENTATION,
                  "signature": {"config": asdict(config), "source_samples": original["signature"]["samples"],
                                "aliases": bank.alias_names, "vocabulary_sha256": original["signature"]["vocabulary_sha256"],
                                "primary": "ScoreInfluence", "support_solver_weight": 1.,
                                "semantic_measurement": "native original-minus-erased class-score change / soft-erasure area",
                                "measurement_normalization": "divide H row and observation by H L2 norm; same for all classes",
                                "fill": "valid-window per-channel RGB mean; same 512px coordinates"},
                  "maximum_control_error": maximum_control_error,
                  "metrics": {method: summary(matrix, bank.class_names, 0) for method, matrix in matrices.items()},
                  "transitions": {method: {"counts": tensor.tolist(), **transition_summary(tensor)} for method, tensor in transitions.items()},
                  "records": records, "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576,
                  "note": "Overlapping centers in exactly 16 fixed windows, not full-image mIoU. Labels audit-only. Frozen weights and same all20 aliases. Actual erasure changes context and may introduce artifacts; no semantic correctness guarantee."}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "windows": len(records),
                          "miou": {method: row["mean_iou_percent"] for method, row in result["metrics"].items()}}), flush=True)
        del cache, prepared, scores, rgb, image
    if seen != expected:
        raise ValueError("Incomplete window coverage.")


if __name__ == "__main__":
    main(parse_args())
