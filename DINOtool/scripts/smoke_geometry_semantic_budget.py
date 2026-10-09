"""Real-checkpoint, mask-free identity/constraint checks and single-arm cost."""
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_semantic_budget import IMPLEMENTATION, PRIMARY, BudgetConfig, read_semantic_budget, encode_single
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
        raise ValueError("Refusing existing smoke output.")
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    weights = [parameter.clone() for parameter in backbone.model.visual_model.head.parameters()]
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    yy = (torch.arange(32, device=backbone.device)+.5)*16
    valid = ((yy[:, None] < min(image.shape[-2], 512)) & (yy[None, :] < min(image.shape[-1], 512))).reshape(1, -1)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        cache = prepared.backbone_tokens.clone()
        features, diagnostics = read_semantic_budget(geometry, prepared, valid)
        error = float((features["Geometry"]-prepared.geometry_projected).abs().max())
        invariant_errors = {key: value for key, value in diagnostics.items()
                            if any(name in key for name in ("constraint_violation", "mass_error", "invalid_edge_error"))}
        if (error != 0. or not torch.equal(cache, prepared.backbone_tokens)
                or any(value > 3e-5 for key, value in diagnostics.items() if "constraint_violation" in key)
                or any(value > 2e-6 for key, value in diagnostics.items() if "mass_error" in key)
                or any(value != 0. for key, value in diagnostics.items() if "invalid_edge_error" in key)):
            raise RuntimeError(json.dumps({"amp": amp, "geometry_error": error,
                "cache_error": float((cache-prepared.backbone_tokens).abs().max()), "invariants": invariant_errors}))
        checks["bf16" if amp else "fp32"] = {"geometry_identity_max_error": error,
            "native_cache_max_error": float((cache-prepared.backbone_tokens).abs().max()),
            "finite_features": all(bool(torch.isfinite(x).all()) for x in features.values()),
            "diagnostics": diagnostics}
    unchanged = all(torch.equal(a, b) for a, b in zip(weights, backbone.model.visual_model.head.parameters()))
    if not unchanged:
        raise RuntimeError("Frozen head parameters changed.")
    del weights, prepared, cache, features
    backbone.use_amp = True
    cost = {}
    for name, mode in (("Geometry", "geometry"), (PRIMARY, "budget")):
        for _ in range(2):
            encode_single(backbone, rgb, valid, mode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            feature = encode_single(backbone, rgb, valid, mode)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
            del feature
        cost[name] = {"window_median_seconds": statistics.median(durations), "runs_seconds": durations,
                      "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": BudgetConfig().signature(),
              "primary": PRIMARY, "target_masks_loaded": False, "checks": checks,
              "sample_key": samples[0].key, "head_weights_unchanged": unchanged,
              "weights_frozen": all(not p.requires_grad for p in backbone.model.parameters()),
              "deployed_window_cost": cost,
              "cost_note": "Two warmups/five synchronized repeats; one backbone and one head, no baseline prepare or control heads."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
