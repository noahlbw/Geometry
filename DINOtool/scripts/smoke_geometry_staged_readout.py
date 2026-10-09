"""Frozen real-checkpoint stage identities and deployed one-window latency."""
import json
from pathlib import Path
import statistics
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_staged_readout import IMPLEMENTATION, PRIMARY, settings, read_staged, encode_single
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
        cache = prepared.backbone_tokens.clone()
        features, diagnostics = read_staged(geometry, prepared)
        with backbone._autocast():
            blocked, _ = run_head(backbone.model.visual_model.head, prepared.backbone_tokens,
                prepared.backbone_tokens[:, prepared.prefix_tokens:], prepared.geometry_patch_conditional,
                prepared.prefix_tokens, "Geometry_BlockPrefix", 1)
        errors = {"geometry": float((features['Geometry']-prepared.geometry_projected).abs().max()),
                  "blocked": float((features['Geometry_BlockPrefix']-blocked).abs().max()),
                  "cache": float((cache-prepared.backbone_tokens).abs().max())}
        finite = all(bool(torch.isfinite(value).all()) for value in features.values())
        stage0 = all(value == diagnostics['Geometry__'+name.split('__', 1)[1]]
                     for name, value in diagnostics.items() if name.startswith(PRIMARY+'__block0_'))
        if any(errors.values()) or not finite or not stage0:
            raise RuntimeError(json.dumps({"amp": amp, "errors": errors, "finite": finite, "stage0_identity": stage0}))
        checks['bf16' if amp else 'fp32'] = {"errors": errors, "finite_features": finite, "stage0_identity": stage0}
    unchanged = all(torch.equal(a, b) for a, b in zip(weights, backbone.model.visual_model.head.parameters()))
    frozen = all(not p.requires_grad for p in backbone.model.parameters())
    if not unchanged or not frozen:
        raise RuntimeError("Weights are not frozen and unchanged.")
    del weights, prepared, features, blocked, cache
    backbone.use_amp = True
    cost = {}
    for method in ('Geometry', PRIMARY):
        for _ in range(2):
            encode_single(backbone, rgb, method)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            features = encode_single(backbone, rgb, method)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-started)
            del features
        cost[method] = {"window_median_seconds": statistics.median(durations), "runs_seconds": durations,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    row = {"status": "complete", "implementation": IMPLEMENTATION, "config": settings(), "checks": checks,
           "target_masks_loaded": False, "weights_frozen": frozen, "head_weights_unchanged": unchanged,
           "sample_key": samples[0].key, "deployed_window_cost": cost,
           "cost_note": "Two warmups/five synchronized512-window repeats, one backbone and one selected head; no control heads."}
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main(parse_args())
