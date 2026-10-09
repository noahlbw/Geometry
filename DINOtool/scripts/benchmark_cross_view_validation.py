"""Standalone same-image Geometry/cross-view costs, without target masks."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.cross_view_validation import IMPLEMENTATION, PRIMARY
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_cross_view_validation import predict_image as cross_predict
from eval_frozen_semantic_path import predict_image as local_predict
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


def read_geometry(geometry, prepared, banks, texts, valid):
    return {key: {"Geometry": alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                 bank.parent_indices, bank.class_count)} for key, bank in banks.items()}, {}


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing benchmark output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
                             TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    output.mkdir(parents=True)
    reference, _ = cross_predict(image, geometry, banks, output, ("Geometry", PRIMARY))
    local, _ = local_predict(image, geometry, banks, output, ("Geometry",), read_geometry)
    if any(not np.array_equal(reference[key]["Geometry"], local[key]["Geometry"]) for key in banks):
        raise RuntimeError("Standalone Geometry differs from experiment control.")
    methods = {
        "Geometry": lambda: local_predict(image, geometry, banks, output, ("Geometry",), read_geometry),
        PRIMARY: lambda: cross_predict(image, geometry, banks, output, (PRIMARY,)),
    }
    results = {}
    for method, function in methods.items():
        function()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, _ = function()
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-started)
            if any(not np.array_equal(reference[key][method], prediction[key][method]) for key in banks):
                raise RuntimeError("Standalone timed prediction changed.")
        results[method] = {"seconds": timings, "median_seconds": float(np.median(timings)),
                           "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "target_masks_loaded": False,
              "sample_key": samples[0].key, "shape": list(image.shape), "methods": results,
              "ratio": results[PRIMARY]["median_seconds"]/results["Geometry"]["median_seconds"],
              "exact_prediction_replay": True}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
