"""Actual-checkpoint execution and deployed-window cost; never load target masks."""
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_region_strong import IMPLEMENTATION, METHODS, PRIMARY, SOURCE, FrozenStrongRegionObserver
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.region_semantic_readout import restricted_pool
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    observer = FrozenStrongRegionObserver(SOURCE, banks, backbone.device)
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    prepared = geometry.prepare_image(rgb)
    native = prepared.backbone_tokens.clone()
    valid = torch.ones((1, 1024), dtype=torch.bool, device=backbone.device)
    hidden, descriptor = observer.visual(rgb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        replay = restricted_pool(observer.model.vision_model.head, hidden,
                                 torch.ones(hidden.shape[:2], device=hidden.device))
    replay_error = float((F.normalize(replay.float(), dim=-1)-descriptor).abs().max())
    scores, diagnostics = observer(geometry, prepared, banks, texts, valid, rgb)
    baseline_error = max(float((scores[key]["Geometry"]-alias_class_scores(
        prepared.geometry_projected.float() @ texts[key].T, bank.parent_indices, bank.class_count)).abs().max())
        for key, bank in banks.items())
    cached_error = float((native-prepared.backbone_tokens).abs().max())
    finite = all(value.shape == (1, 1024, banks[key].class_count) and bool(torch.isfinite(value).all())
                 for key, group in scores.items() for value in group.values())
    frozen = all(not parameter.requires_grad for parameter in observer.model.parameters())
    if replay_error > 3e-3 or baseline_error != 0 or cached_error != 0 or not finite or not frozen:
        raise RuntimeError("Real-checkpoint pool, baseline, cache, finite or frozen invariant failed.")
    cost = {}
    for name in ("Geometry", PRIMARY):
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            current = geometry.prepare_image(rgb)
            if name == PRIMARY:
                observer(geometry, current, banks, texts, valid, rgb)
            else:
                for key, bank in banks.items():
                    alias_class_scores(current.geometry_projected.float() @ texts[key].T,
                                       bank.parent_indices, bank.class_count)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        cost[name] = {"window_median_seconds": statistics.median(times), "runs_seconds": times,
                      "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "primary": PRIMARY,
        "sample_key": samples[0].key, "target_masks_loaded": False, "source": observer.manifest,
        "checks": {"trained_pool_replay_max_error": replay_error, "geometry_baseline_max_error": baseline_error,
                   "native_cache_max_error": cached_error, "finite_scores": finite, "weights_frozen": frozen,
                   "observer_token_count": hidden.shape[1], "semantic_temperature": observer.semantic_temperature},
        "diagnostics": diagnostics, "deployed_window_cost": cost,
        "cost_note": "Window-only timing; both encoders remain resident in both measurements. "
                     "Not independently deployed Geometry peak memory or full-image latency."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
