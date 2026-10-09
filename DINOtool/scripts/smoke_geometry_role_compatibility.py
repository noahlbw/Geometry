"""Check real frozen-head identities, role fallback and window cost, without masks."""
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_role_compatibility import IMPLEMENTATION, PRIMARY, read_role_head, read_role_compatibility
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
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    yy = (torch.arange(32, device=backbone.device)+.5)*16
    valid = ((yy[:, None] < min(image.shape[-2], 512)) & (yy[None, :] < min(image.shape[-1], 512))).reshape(1, -1)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        cache = prepared.backbone_tokens.clone()
        with backbone._autocast():
            replay, _ = read_role_head(backbone.model.visual_model.head, prepared, valid, mode="geometry")
        replay_error = float((replay-prepared.geometry_projected).abs().max())
        features, diagnostics = read_role_compatibility(geometry, prepared, valid)
        cache_error = float((cache-prepared.backbone_tokens).abs().max())
        finite = all(value.shape == prepared.geometry_projected.shape and bool(torch.isfinite(value).all())
                     for value in features.values())
        if replay_error != 0. or cache_error != 0. or not finite or diagnostics["row_mass_error"] > 1e-5:
            raise RuntimeError("Original Geometry identity, cache, relation mass or finite check failed.")
        checks["bf16" if amp else "fp32"] = {"geometry_identity_max_error": replay_error,
            "native_cache_max_error": cache_error, "finite_features": finite, "diagnostics": diagnostics}
    backbone.use_amp = True
    del features, cache, prepared, replay
    cost = {}
    for method in ("Geometry", PRIMARY):
        durations = []
        for _ in range(2):
            current = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_role_compatibility(geometry, current, valid, controls=False)
            del current
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            current = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_role_compatibility(geometry, current, valid, controls=False)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-started)
            del current
        cost[method] = {"window_median_seconds": statistics.median(durations), "runs_seconds": durations,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "primary": PRIMARY,
              "sample_key": samples[0].key, "target_masks_loaded": False, "checks": checks,
              "weights_frozen": all(not parameter.requires_grad for parameter in backbone.model.parameters()),
              "deployed_window_cost": cost,
              "cost_note": "Two warmups/five synchronized single-window repeats, original prepare-image pipeline. "
                           "No control heads in primary timing. Not full-image end-to-end latency."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
