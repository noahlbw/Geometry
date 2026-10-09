"""Mask-free real-checkpoint replay and independent contrast-reader cost."""
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_conservative_contrast import BASELINES, REFERENCES, IMPLEMENTATION, PRIMARY, settings, read_factors, encode_single
from dinotool.geometry_csa_allocation import read_head as allocation_head
from dinotool.matched_readout_controls import run_head
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    weights = [p.clone() for p in backbone.model.visual_model.head.parameters()]
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        cached = prepared.backbone_tokens.clone()
        features, diagnostics = read_factors(geometry, prepared)
        errors = {"geometry": float((features["Geometry"]-prepared.geometry_projected).abs().max()),
                  "cache": float((cached-prepared.backbone_tokens).abs().max())}
        with backbone._autocast():
            for method in BASELINES:
                expected = run_head(backbone.model.visual_model.head, prepared.backbone_tokens,
                    prepared.backbone_tokens[:, prepared.prefix_tokens:], prepared.geometry_patch_conditional,
                    prepared.prefix_tokens, method, 1)[0]
                errors[method] = float((features[method]-expected).abs().max())
            for method in REFERENCES:
                expected = allocation_head(backbone.model.visual_model.head, prepared, method)[0]
                errors[method] = float((features[method]-expected).abs().max())
        for method in ("Geometry", PRIMARY, "Geometry_ContrastGainOnly"):
            singleton = encode_single(backbone, rgb, method)
            errors[method+"_singleton"] = float((features[method]-singleton).abs().max())
        finite = all(bool(torch.isfinite(value).all()) for value in features.values())
        centered = max(diagnostics[PRIMARY+f"__block{i}_correction_mean_max_error"] for i in (0, 1))
        if any(errors.values()) or not finite or centered > 2e-5:
            raise RuntimeError(json.dumps({"amp": amp, "errors": errors, "finite": finite, "centered_error": centered}))
        checks["bf16" if amp else "fp32"] = {"errors": errors, "finite_features": finite, "diagnostics": diagnostics}
    unchanged = all(torch.equal(a, b) for a, b in zip(weights, backbone.model.visual_model.head.parameters()))
    frozen = all(not p.requires_grad for p in backbone.model.parameters())
    if not unchanged or not frozen:
        raise RuntimeError("Weights not frozen/unchanged.")
    del weights, prepared, features, expected, cached, singleton
    backbone.use_amp = True
    costs = {}
    for method in ("Geometry", PRIMARY, "Geometry_ContrastGainOnly"):
        for _ in range(5):
            encode_single(backbone, rgb, method)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(15):
            torch.cuda.synchronize()
            started = time.perf_counter()
            feature = encode_single(backbone, rgb, method)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-started)
            del feature
        costs[method] = {"window_median_seconds": statistics.median(durations), "runs_seconds": durations,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    row = {"status": "complete", "implementation": IMPLEMENTATION, "config": settings(), "checks": checks,
           "target_masks_loaded": False, "weights_frozen": frozen, "head_weights_unchanged": unchanged,
           "sample_key": samples[0].key, "deployed_window_cost": costs,
           "cost_note": "Five warmups/15 synchronized512-window repeats; one backbone/one selected head; traces and comparison heads disabled."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(parse_args())
