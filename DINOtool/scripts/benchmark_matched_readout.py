"""Measure one deployed readout at a time, with exact Geometry replay guards."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.matched_readout_controls import IMPLEMENTATION, METHODS, encode_single
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_geometry_semantic_innovation import parse_args
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


def synchronize():
    torch.cuda.synchronize()


@torch.inference_mode()
def image_forward(backbone, image, bank, method, work):
    height, width = image.shape[-2:]
    text = F.normalize(bank.features.float(), dim=-1)
    blend = hann_blend_window(512)
    calls = 0
    with ProbabilityAccumulator(bank.class_count, height, width, 256, work) as accumulator:
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                feature, _ = encode_single(backbone, _crop_at(image, top, left, 512).to(backbone.device), method)
                scores = alias_class_scores(feature @ text.T, bank.parent_indices, bank.class_count)
                dense = F.interpolate(scores.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                      size=(512, 512), mode="bilinear", align_corners=False)[0]/.07
                ah, aw = min(512, height-top), min(512, width-left)
                accumulator.add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                calls += 1
        prediction, _ = accumulator.finalize(None)
    return prediction, calls


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing benchmark output.")
    specs = load_class_specs(args.vocabulary_config)
    samples, groups, load_image, _ = protocol(args, specs)
    model_started = time.perf_counter()
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    synchronize()
    model_seconds = time.perf_counter()-model_started
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    text_started = time.perf_counter()
    banks = {key: geometry.encode_text(classes) for key, classes in groups.items()}
    synchronize()
    text_seconds = time.perf_counter()-text_started
    key = next(iter(banks))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    prepared = geometry.prepare_image(rgb)
    replay = {}
    for method, reference in (("Geometry", prepared.geometry_projected), ("Native", prepared.native_projected)):
        observed, _ = encode_single(backbone, rgb, method)
        error = float((observed-reference).abs().max())
        if error != 0.:
            raise ValueError(f"Independent {method} replay differs: {error}")
        replay[method] = error
    del prepared, observed, reference
    output.mkdir(parents=True)
    rows = {}
    for method in METHODS:
        for _ in range(5):
            encode_single(backbone, rgb, method)
        synchronize()
        torch.cuda.reset_peak_memory_stats()
        window_times = []
        for _ in range(20):
            synchronize()
            started = time.perf_counter()
            encode_single(backbone, rgb, method)
            synchronize()
            window_times.append(1000*(time.perf_counter()-started))
        window_peak = torch.cuda.max_memory_allocated()/1048576
        image_times = []
        image_forward(backbone, image, banks[key], method, output)
        synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            synchronize()
            started = time.perf_counter()
            _, calls = image_forward(backbone, image, banks[key], method, output)
            synchronize()
            image_times.append(time.perf_counter()-started)
        rows[method] = {"window_median_ms": statistics.median(window_times),
                        "window_mean_ms": statistics.mean(window_times),
                        "window_p95_ms": sorted(window_times)[18],
                        "window_peak_allocated_mib": window_peak,
                        "image_median_seconds": statistics.median(image_times),
                        "image_peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576,
                        "backbone_calls_per_image": calls,
                        "modified_head_blocks": 0 if method == "Native" else 1 if method.endswith("Last") else 2}
        print(method, json.dumps(rows[method]), flush=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "precision": "fp32 weights/bf16 AMP", "dataset": args.dataset,
              "sample_key": samples[0].key, "image_shape": list(image.shape),
              "model_load_seconds": model_seconds, "text_encode_seconds": text_seconds,
              "exact_independent_replay": replay, "metrics": rows,
              "note": "Each arm includes its own single backbone and head only. Text/load excluded from steady-state. "
                      "Whole-image timings include CPU probability transfer/blending and disk-backed accumulator, "
                      "exclude image decoding and target-mask evaluation. These are independent latency measurements."}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main(parse_args())
