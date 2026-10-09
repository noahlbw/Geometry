"""Measure the complete primary alone against original Geometry on one image."""
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_reacquisition import GeometryReacquisition, IMPLEMENTATION, PRIMARY
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.matched_readout_controls import encode_single
from dinotool.model import DINOTextSegmenter
from dinotool.parallel_readout import structural_logits
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def primary_only(backbone, reader, rgb):
    normalized = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
    with backbone._autocast():
        cls, raw, registers = reader.visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else raw.dtype
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                                   geometry_patch_conditional=relation, block_index=len(reader.visual.head.blocks)-1)
        features, _ = reader.read(prepared, controls=False)
    return features[PRIMARY]


@torch.inference_mode()
def predict(image, bank, encode, work):
    height, width = image.shape[-2:]
    text = F.normalize(bank.features.float(), dim=-1)
    blend = hann_blend_window(512)
    calls = 0
    with ProbabilityAccumulator(bank.class_count, height, width, 256, work) as accumulator:
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                feature = encode(_crop_at(image, top, left, 512))
                scores = alias_class_scores(feature @ text.T, bank.parent_indices, bank.class_count)
                dense = F.interpolate(scores.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                      size=(512, 512), mode="bilinear", align_corners=False)[0]/.07
                ah, aw = min(512, height-top), min(512, width-left)
                accumulator.add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                calls += 1
        return accumulator.finalize(None)[0], calls


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing an existing independent benchmark")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    bank = geometry.encode_text(next(iter(specs.values())))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    reader = GeometryReacquisition(backbone)
    prepared = geometry.prepare_image(rgb)
    actual = reader.read(prepared, controls=False)[0][PRIMARY]
    standalone = primary_only(backbone, reader, rgb)
    error = float((actual-standalone).abs().max())
    if error != 0.:
        raise RuntimeError(f"Standalone primary differs from evaluation: {error}")
    reader.close()
    del prepared, actual, standalone, reader
    output.mkdir(parents=True)
    rows = {}
    for method in ("Geometry", PRIMARY):
        reader = GeometryReacquisition(backbone) if method == PRIMARY else None
        encode = ((lambda crop: primary_only(backbone, reader, crop.to(backbone.device)))
                  if reader is not None else
                  (lambda crop: encode_single(backbone, crop.to(backbone.device), "Geometry")[0]))
        try:
            for _ in range(5):
                encode(rgb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            window_times = []
            for _ in range(20):
                torch.cuda.synchronize()
                started = time.perf_counter()
                feature = encode(rgb)
                torch.cuda.synchronize()
                window_times.append((time.perf_counter()-started)*1000)
            window_peak = torch.cuda.max_memory_allocated()/1048576
            del feature
            predict(image, bank, encode, output)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            image_times = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                _, calls = predict(image, bank, encode, output)
                torch.cuda.synchronize()
                image_times.append(time.perf_counter()-started)
            rows[method] = {"window_median_ms": statistics.median(window_times),
                            "window_p95_ms": sorted(window_times)[18],
                            "window_peak_allocated_mib": window_peak,
                            "image_median_seconds": statistics.median(image_times),
                            "image_peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576,
                            "backbone_calls_per_image": calls,
                            "replayed_backbone_blocks_per_window": 4 if reader is not None else 0}
            print(method, json.dumps(rows[method]), flush=True)
        finally:
            if reader is not None:
                reader.close()
    result = {"status": "complete", "implementation": IMPLEMENTATION, "gpu": torch.cuda.get_device_name(),
              "sample_key": samples[0].key, "image_shape": list(image.shape),
              "standalone_primary_max_error": error, "metrics": rows,
              "window_ratio": rows[PRIMARY]["window_median_ms"]/rows["Geometry"]["window_median_ms"],
              "image_ratio": rows[PRIMARY]["image_median_seconds"]/rows["Geometry"]["image_median_seconds"],
              "note": "Separate deployed arms, excluding all comparison heads. One-image benchmark, "
              "not full eight-domain throughput. Fixed diagnostic scalar collection remains in primary. "
              "Text/model load and label evaluation excluded; image timings include probability assembly."}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
