"""Separate deployed primary/Geometry cost on one image, without control-head overhead."""
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from dinotool.gear_ov import _crop_at
from dinotool.matched_readout_controls import encode_single
from dinotool.model import DINOTextSegmenter
from dinotool.parallel_readout import structural_logits
from dinotool.prompts import load_class_specs
from dinotool.sat_geometry_transport import IMPLEMENTATION, PRIMARY, SATGeometryTransport
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from benchmark_geometry_reacquisition import predict
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def primary_only(backbone, reader, rgb):
    normalized = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
    with backbone._autocast():
        visual = backbone.model.visual_model
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else raw.dtype
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(dtype)
    prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                               geometry_patch_conditional=relation, block_index=len(visual.head.blocks)-1,
                               grid_height=rgb.shape[-2]//16, grid_width=rgb.shape[-1]//16)
    valid = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
    return reader.read(prepared, rgb, valid, controls=False)[0][PRIMARY]


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing independent benchmark output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    bank = geometry.encode_text(next(iter(specs.values())))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    output.mkdir(parents=True)
    rows, error = {}, None
    for method in ("Geometry", PRIMARY):
        reader = SATGeometryTransport(backbone) if method == PRIMARY else None
        if reader is not None:
            prepared = geometry.prepare_image(rgb)
            valid = torch.ones((1, 1024), dtype=torch.bool, device=rgb.device)
            actual = reader.read(prepared, rgb, valid, controls=False)[0][PRIMARY]
            standalone = primary_only(backbone, reader, rgb)
            error = float((actual-standalone).abs().max())
            if error != 0.:
                raise RuntimeError(f"Standalone primary differs from evaluator: {error}")
            del prepared, actual, standalone
        encode = ((lambda crop: primary_only(backbone, reader, crop.to(backbone.device)))
                  if reader is not None else
                  (lambda crop: encode_single(backbone, crop.to(backbone.device), "Geometry")[0]))
        for _ in range(3):
            encode(rgb)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            started = time.perf_counter()
            encode(rgb)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-started)*1000)
        window_peak = torch.cuda.max_memory_allocated()/1048576
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
        rows[method] = {"window_median_ms": statistics.median(times),
                        "window_peak_allocated_mib": window_peak,
                        "image_median_seconds": statistics.median(image_times),
                        "image_peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576,
                        "lvd_backbone_calls_per_image": calls,
                        "sat_backbone_calls_per_image": calls if reader is not None else 0,
                        "procrustes_fits_per_image": calls if reader is not None else 0}
        print(method, json.dumps(rows[method]), flush=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION,
              "sample_key": samples[0].key, "image_shape": list(image.shape),
              "gpu": torch.cuda.get_device_name(), "standalone_primary_max_error": error,
              "metrics": rows,
              "image_ratio": rows[PRIMARY]["image_median_seconds"]/rows["Geometry"]["image_median_seconds"],
              "window_ratio": rows[PRIMARY]["window_median_ms"]/rows["Geometry"]["window_median_ms"],
              "note": "One-image native1024 LoveDA benchmark; no labels, no control heads. "
                      "Geometry measured before SAT allocation. Model/text load excluded; "
                      "probability assembly included. Not full-suite deployed throughput."}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
