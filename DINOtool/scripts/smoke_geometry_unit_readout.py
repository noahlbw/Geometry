"""Real-checkpoint unit lifting, singleton identity and cost without loading masks."""
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_unit_readout import IMPLEMENTATION, PRIMARY, UnitConfig, read_unit_geometry
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
    nominal = backbone.model.model_config.vision_model_train_img_size
    if nominal // backbone.patch_size != UnitConfig().grid_side:
        raise ValueError("Fixed unit budget differs from author nominal training configuration.")
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    yy = (torch.arange(32, device=backbone.device)+.5)*16
    valid = ((yy[:, None] < min(image.shape[-2], 512)) & (yy[None] < min(image.shape[-1], 512))).reshape(1, -1)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        cache = prepared.backbone_tokens.clone()
        identity, _ = read_unit_geometry(geometry, prepared, valid, controls=False, singleton=True)
        error = float((identity[PRIMARY]-prepared.geometry_projected).abs().max())
        features, diagnostics = read_unit_geometry(geometry, prepared, valid)
        cache_error = float((cache-prepared.backbone_tokens).abs().max())
        finite = all(value.shape == prepared.geometry_projected.shape and bool(torch.isfinite(value).all())
                     for value in features.values())
        if error != 0. or cache_error != 0. or not finite or diagnostics["unit_mean_max_error"] > 2e-4:
            raise RuntimeError("Unit identity, mean reconstruction or cache invariants failed.")
        checks["bf16" if amp else "fp32"] = {"geometry_identity_max_error": error,
            "native_cache_max_error": cache_error, "finite_features": finite, "diagnostics": diagnostics}
    backbone.use_amp = True
    del identity, features, prepared, cache
    cost = {}
    for method in ("Geometry", PRIMARY):
        times = []
        for _ in range(2):
            prepared = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_unit_geometry(geometry, prepared, valid, controls=False)
            del prepared
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            prepared = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_unit_geometry(geometry, prepared, valid, controls=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
            del prepared
        cost[method] = {"window_median_seconds": statistics.median(times), "runs_seconds": times,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "primary": PRIMARY,
              "sample_key": samples[0].key, "target_masks_loaded": False, "checks": checks,
              "nominal_training_image_size": nominal, "unit_budget": UnitConfig().units,
              "weights_frozen": all(not p.requires_grad for p in backbone.model.parameters()),
              "deployed_window_cost": cost,
              "cost_note": "Two warmups/five synchronized single-window repeats including partition; "
                           "no control heads in primary timing. Not full-image end-to-end latency."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
