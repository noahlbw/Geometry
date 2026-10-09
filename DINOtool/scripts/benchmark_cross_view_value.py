"""Single-arm descriptor cost for the fixed Geometry/cross-view candidate."""
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.cross_view_value_innovation import IMPLEMENTATION, PRIMARY
from dinotool.gear_ov import _crop_at
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_cross_view_value_innovation import CONFIG, parse_args, tile_features
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def main(args):
    path = Path(args.output_dir)
    if path.exists():
        raise RuntimeError("Refusing existing benchmark result.")
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
        TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    rgb = _crop_at(image, 0, 0, 512).to(geometry.device)
    costs = {}
    for method in ("Geometry", PRIMARY):
        def run():
            prepared = geometry.prepare_image(rgb)
            if method == "Geometry":
                return prepared.geometry_projected
            return tile_features(image, geometry, prepared, 0, 0, controls=False)[0][PRIMARY]

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
            features = run()
            torch.cuda.synchronize()
            trials.append(1000*(time.perf_counter()-start))
            if not torch.isfinite(features).all():
                raise RuntimeError("Nonfinite standalone descriptor.")
            del features
        costs[method] = {"median_ms": statistics.median(trials), "minimum_ms": min(trials),
            "maximum_ms": max(trials), "trials_ms": trials,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": CONFIG.signature(),
        "sample_key": sample.key, "dataset": args.dataset, "image_shape": list(image.shape),
        "warmups": 3, "trials": 20, "costs": costs, "target_masks_loaded": False,
        "weights_frozen": all(not p.requires_grad for p in geometry.backbone.model.parameters()),
        "scope": "As-implemented single512-window descriptor extraction, one resident frozen model; "
            "candidate includes local preparation, context preparation and coupled head only, no comparator arms. "
            "No text encoding/scoring, dense probability assembly, image decoding or initialization timed. "
            "Preparation computes native/Geometry auxiliary paths; this is not an optimized or full-image latency."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
