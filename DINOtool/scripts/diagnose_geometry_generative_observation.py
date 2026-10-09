"""Audit one fixed generative observer on the original saved Geometry windows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_generative_observation import (
    FrozenGenerativeObserver, GenerativeObservationConfig, IMPLEMENTATION, METHODS,
)
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_gear_ov import digest, protocol
from eval_grounded_context import write_json
from merge_competitive_evidence_shards import metric_summary


@torch.inference_mode()
def main(args):
    source, output = Path(args.source_diagnostic_dir), Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require a fresh output and a valid shard index.")
    reference = json.loads((source/"results.json").read_text())
    if reference["status"] != "complete" or reference["processed_images"] != 8:
        raise ValueError("Requires the completed original eight-image diagnostic.")
    vocabulary_path = Path(args.vocabulary_config)
    samples, _, load_rgb, _ = protocol(args, load_class_specs(vocabulary_path))
    lookup = {sample.key: sample for sample in samples}
    identity = reference["signature"]
    if (identity["dataset"] != args.dataset
            or digest([sample.key for sample in samples]) != identity["global_sample_keys_sha256"]
            or hashlib.sha256(vocabulary_path.read_bytes()).hexdigest() != identity["vocabulary_sha256"]):
        raise ValueError("The original image protocol or all20 vocabulary changed.")
    paths = sorted((source/"features").glob("*.pt"))
    if len(paths) != 16:
        raise ValueError("Requires the exact original 16 cached windows.")
    selected = paths[args.shard_index::args.num_shards]
    if args.smoke:
        selected = selected[:1]
    images = {row["sample_key"]: row for row in reference["images"]}
    names, aliases = tuple(identity["class_names"]), tuple(identity["alias_names"])
    bank = torch.load(source/"text_bank.pt", map_location="cpu", weights_only=True)
    config = GenerativeObservationConfig()
    observer = FrozenGenerativeObserver(args.generative_source, aliases, bank["parents"], names,
                                         device=args.device, config=config)
    matrices = {method: np.zeros((len(names), len(names)), np.int64) for method in METHODS}
    transitions = {method: np.zeros((len(names),)*3, np.int64) for method in METHODS[1:]}
    output.mkdir(parents=True)
    (output/"observations").mkdir()
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "config": config.signature(),
                 "source_geometry_signature": identity, "generative_source": observer.manifest,
                 "global_snapshot_files": [path.name for path in paths],
                 "num_shards": args.num_shards, "shard_index": args.shard_index, "smoke": args.smoke,
                 "class_names": names, "alias_names": aliases,
                 "parent_indices": bank["parents"].tolist(),
                 "word_count": [int((bank["parents"] == c).sum()) for c in range(len(names))],
                 "note": "Original Geometry/samples unchanged. Diffusion uses one aerial template for every "
                         "unchanged alias and canonical name. Uniform mean epsilon loss is a compatibility "
                         "proxy, not a calibrated class likelihood. No target labels choose supports/prompts/noise."}
    write_json(output/"signature.json", signature)
    records = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for path in selected:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        image = load_rgb(lookup[cache["sample_key"]])
        record = images[cache["sample_key"]]
        if tuple(image.shape[-2:]) != (record["height"], record["width"]):
            raise ValueError("The diagnostic image shape changed.")
        top, left = cache["top"], cache["left"]
        yy, xx = top+torch.arange(32)*16+8, left+torch.arange(32)*16+8
        valid = ((yy[:, None] < image.shape[-2]) & (xx[None] < image.shape[-1])).flatten()
        rgb = _crop_at(image, top, left, 512)
        key = f"{cache['sample_key']}/{top}/{left}"
        scores, raw, diagnostics = observer(rgb, cache["geometry_relation"][0], valid, key)
        scores["Geometry"] = cache["stage_scores"]["Geometry_original"][0].to(observer.device)
        if not all(bool(torch.isfinite(value).all()) for value in scores.values()):
            raise ValueError("Nonfinite observation.")
        predictions = {method: value.argmax(-1).cpu().numpy() for method, value in scores.items()}
        # Persist every observation before accessing target labels in the cache.
        torch.save({"sample_key": cache["sample_key"], "top": top, "left": left,
                    "scores": {name: value.cpu() for name, value in scores.items()},
                    "raw": {name: value.cpu() for name, value in raw.items()},
                    "image_only_valid": valid, "diagnostics": diagnostics},
                   output/"observations"/path.name)
        truth = cache["target_patch_centers"][0].numpy()
        evaluated = valid.numpy() & (truth >= 0) & (truth < len(names))
        for method, prediction in predictions.items():
            encoded = truth[evaluated].astype(np.int64)*len(names)+prediction[evaluated]
            matrices[method] += np.bincount(encoded, minlength=len(names)**2).reshape(len(names), len(names))
            if method != "Geometry":
                transitions[method] += transition_counts(predictions["Geometry"], prediction,
                                                          np.where(evaluated, truth, 255), len(names))
        records.append({"snapshot": path.name, "sample_key": cache["sample_key"], "top": top, "left": left,
                        "evaluated_centers": int(evaluated.sum()), "diagnostics": diagnostics})
        result = {"status": "complete" if len(records) == len(selected) else "running",
                  "implementation": IMPLEMENTATION, "dataset": args.dataset, "signature": signature,
                  "processed_windows": len(records), "total_windows": len(selected),
                  "records": records, "metrics": {method: metric_summary(matrix, names, 0)
                                                    for method, matrix in matrices.items()},
                  "transitions": {method: {"counts": tensor.tolist(), **transition_summary(tensor)}
                                  for method, tensor in transitions.items()},
                  "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576,
                  "note": "Overlapping original patch centers, not complete-image mIoU. The source images "
                          "are development diagnostics. Spatial denoising losses retain full-image "
                          "context; they do not establish isolated regional likelihood or new independent "
                          "semantic labels. No winner is selected as final model from these diagnostic arms."}
        write_json(output/"results.json", result)
        print(json.dumps({"dataset": args.dataset, "shard": args.shard_index, "processed": len(records),
                          "total": len(selected), "snapshot_seconds": diagnostics["wall_seconds"],
                          "split_agreement": diagnostics["split_schedule_support_agreement"],
                          "patch_center_miou": {method: value["mean_iou_percent"] for method, value in result["metrics"].items()}}),
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"), required=True)
    for name in ("data-root", "vocabulary-config", "source-diagnostic-dir", "generative-source", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20261001)
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
