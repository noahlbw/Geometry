"""Independent one-arm window readout cost, with the fixed all20 OEM text bank."""
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_residual_metric import (
    IMPLEMENTATION, PRIMARY, metric_alias_scores, residual_covariance, ResidualMetricReader)
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_cross_view_value_innovation import parse_args
from eval_geometry_residual_metric import CONFIG
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def main(args):
    path = Path(args.output_dir)
    if path.exists():
        raise RuntimeError("Refusing existing benchmark output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    if len(specs) != 1:
        raise RuntimeError("Benchmark requires the one OEM bank.")
    sample = samples[0]
    image = load_image(sample)
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
        TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    reader = ResidualMetricReader(geometry, banks)
    key, bank = next(iter(banks.items()))
    text = reader.texts[key]
    rgb = _crop_at(image, 0, 0, 512).to(geometry.device)
    valid = torch.ones((1, 1024), device=geometry.device, dtype=torch.bool)
    costs = {}
    for method in ("Geometry", PRIMARY):
        def run():
            prepared = geometry.prepare_image(rgb)
            features = prepared.geometry_projected.float()
            if method == "Geometry":
                response = features @ text.T
            else:
                covariance = residual_covariance(features @ reader.basis, prepared.geometry_patch_conditional, valid)
                response, _ = metric_alias_scores(features, text, reader.basis, covariance, valid)
            return alias_class_scores(response, bank.parent_indices, bank.class_count)

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trials = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = run()
            torch.cuda.synchronize()
            trials.append(1000*(time.perf_counter()-start))
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("Nonfinite benchmark class field.")
            del scores
        costs[method] = {"median_ms": statistics.median(trials), "minimum_ms": min(trials),
            "maximum_ms": max(trials), "trials_ms": trials,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": CONFIG.signature(),
        "sample_key": sample.key, "dataset": args.dataset, "image_shape": list(image.shape),
        "warmups": 3, "trials": 20, "costs": costs, "target_masks_loaded": False,
        "weights_frozen": all(not p.requires_grad for p in geometry.backbone.model.parameters()),
        "scope": "Single512-window original Geometry preparation plus all20 alias scoring and class LME. "
            "Primary additionally computes its residual covariance, Cholesky factor and joint metric cosines. "
            "One resident model, no comparator heads/covariances. Excludes text encoding/span construction, "
            "dense image assembly/transfer, decoding and initialization; not optimized or whole-image latency."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
