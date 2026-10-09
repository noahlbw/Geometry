"""Bounded frozen-language diagnostic on original Geometry snapshot positions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_language_observation import FrozenLanguageObserver, IMPLEMENTATION, METHODS, LanguageObservationConfig
from dinotool.prompts import load_class_specs
from eval_gear_ov import digest, protocol
from eval_grounded_context import write_json


def query_metric(matrix, names):
    classes = len(names)
    if matrix.shape != (classes, classes+1) or bool((matrix < 0).any()):
        raise ValueError("Expected true-class rows, class-plus-abstention columns.")
    tp = np.diag(matrix[:, :classes])
    target, predicted = matrix.sum(1), matrix[:, :classes].sum(0)
    union = target+predicted-tp
    iou = np.divide(tp, union, out=np.full(classes, np.nan), where=union > 0)
    return {"mean_iou_percent": float(np.nanmean(iou)*100),
            "pixel_accuracy_percent": float(tp.sum()/max(1, int(target.sum()))*100),
            "abstained_queries": int(matrix[:, -1].sum()), "confusion_matrix": matrix.tolist(),
            "per_class": [{"name": name, "iou_percent": None if np.isnan(iou[index]) else float(iou[index]*100),
                "tp": int(tp[index]), "fp": int(predicted[index]-tp[index]), "fn": int(target[index]-tp[index]),
                "precision_percent": float(tp[index]/max(1, int(predicted[index]))*100),
                "recall_percent": float(tp[index]/max(1, int(target[index]))*100)} for index, name in enumerate(names)]}


@torch.inference_mode()
def main(args):
    source, output = Path(args.source_diagnostic_dir), Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require a fresh output and valid shard.")
    reference = json.loads((source/"results.json").read_text())
    identity = reference["signature"]
    if reference["status"] != "complete" or reference["processed_images"] != 8:
        raise ValueError("Requires the original completed eight-image diagnostic.")
    vocabulary_path = Path(args.vocabulary_config)
    samples, _, load_rgb, _ = protocol(args, load_class_specs(vocabulary_path))
    if (identity["dataset"] != args.dataset or digest([sample.key for sample in samples]) != identity["global_sample_keys_sha256"]
            or hashlib.sha256(vocabulary_path.read_bytes()).hexdigest() != identity["vocabulary_sha256"]):
        raise ValueError("Image protocol or vocabulary changed.")
    lookup = {sample.key: sample for sample in samples}
    paths = sorted((source/"features").glob("*.pt"))
    if len(paths) != 16:
        raise ValueError("Require all original16 saved windows.")
    selected = paths[args.shard_index::args.num_shards]
    if args.smoke:
        selected = selected[:1]
    names, aliases = tuple(identity["class_names"]), tuple(identity["alias_names"])
    bank = torch.load(source/"text_bank.pt", map_location="cpu", weights_only=True)
    config = LanguageObservationConfig()
    observer = FrozenLanguageObserver(args.language_source, names, aliases, bank["parents"].tolist(), args.device, config)
    if any(parameter.requires_grad for parameter in observer.model.parameters()):
        raise ValueError("Observer weights must stay frozen.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "config": config.signature(),
                 "source_geometry_signature": identity, "language_source": observer.manifest,
                 "global_snapshot_files": [path.name for path in paths],
                 "num_shards": args.num_shards, "shard_index": args.shard_index, "smoke": args.smoke,
                 "class_names": names, "alias_names": aliases, "parent_indices": bank["parents"].tolist(),
                 "word_count": [int((bank["parents"] == category).sum()) for category in range(len(names))],
                 "target_labels_passed_to_observer": False, "cached_audit_labels_resident": True,
                 "note": "Rank-stratified original predicted-class query centers, not full-image or full-patch mIoU. "
                         "Resident cached labels enter metrics only after observations are saved. No labels fit "
                         "prompts, crop extents, consensus rule or model parameters. Previously audited development images."}
    output.mkdir(parents=True)
    (output/"observations").mkdir()
    write_json(output/"signature.json", signature)
    classes = len(names)
    matrices = {method: np.zeros((classes, classes+1), np.int64) for method in METHODS}
    changes = {method: dict(beneficial=0, harmful=0, wrong_to_wrong=0, unchanged=0) for method in METHODS[1:]}
    records = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for path in selected:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        image = load_rgb(lookup[cache["sample_key"]])
        top, left = cache["top"], cache["left"]
        yy, xx = top+torch.arange(32)*16+8, left+torch.arange(32)*16+8
        valid = ((yy[:, None] < image.shape[-2]) & (xx[None] < image.shape[-1])).flatten()
        local = cache["stage_scores"]["Geometry_original"][0].to(observer.device)
        observed = observer(_crop_at(image, top, left, 512).to(observer.device),
                            cache["geometry_relation"][0].to(observer.device), valid.to(observer.device), local)
        observed = {key: {method: value.cpu() for method, value in group.items()} if key == "predictions"
                    else group.cpu() if isinstance(group, torch.Tensor) else group for key, group in observed.items()}
        torch.save({"sample_key": cache["sample_key"], "top": top, "left": left, **observed}, output/"observations"/path.name)
        indices = observed["indices"].numpy()
        # Cached labels are resident in the saved source; only this audit uses them.
        truth = cache["target_patch_centers"][0].numpy()[indices]
        evaluated = (truth >= 0) & (truth < classes)
        old = observed["predictions"]["Geometry"].numpy()
        for method, tensor in observed["predictions"].items():
            prediction = tensor.numpy()
            encoded = truth[evaluated].astype(np.int64)*(classes+1)+prediction[evaluated]
            matrices[method] += np.bincount(encoded, minlength=classes*(classes+1)).reshape(classes, classes+1)
            if method != "Geometry":
                changed = prediction != old
                beneficial = changed & (prediction == truth) & evaluated
                harmful = changed & (old == truth) & evaluated
                changes[method]["beneficial"] += int(beneficial.sum())
                changes[method]["harmful"] += int(harmful.sum())
                changes[method]["wrong_to_wrong"] += int((changed & ~beneficial & ~harmful & evaluated).sum())
                changes[method]["unchanged"] += int((~changed & evaluated).sum())
        records.append({"snapshot": path.name, "sample_key": cache["sample_key"], "top": top, "left": left,
                        "evaluated_queries": int(evaluated.sum()), "query_indices": indices.tolist(),
                        "diagnostics": observed["diagnostics"]})
        result = {"status": "complete" if len(records) == len(selected) else "running", "signature": signature,
                  "implementation": IMPLEMENTATION, "dataset": args.dataset, "processed_windows": len(records),
                  "total_windows": len(selected), "records": records,
                  "metrics": {method: query_metric(matrix, names) for method, matrix in matrices.items()},
                  "changes": changes, "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576,
                  "note": "Only common, image-selected query centers are scored. Abstentions count as FN, "
                          "not omitted points and not an extra semantic class in mIoU. This is not a full dense "
                          "segmentation or official VIP comparison. No target-mask parameter fitting."}
        write_json(output/"results.json", result)
        print(json.dumps({"dataset": args.dataset, "processed": len(records), "total": len(selected),
                          "query_miou": {method: metric["mean_iou_percent"] for method, metric in result["metrics"].items()},
                          "diagnostics": observed["diagnostics"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"), required=True)
    for name in ("data-root", "vocabulary-config", "source-diagnostic-dir", "language-source", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20261001)
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
