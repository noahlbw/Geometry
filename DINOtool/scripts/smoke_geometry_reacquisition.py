"""Verify real-checkpoint native identity and frozen Geometry coupling on one tile."""
import json
from pathlib import Path
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.geometry_reacquisition import GeometryReacquisition, IMPLEMENTATION, PRIMARY
from dinotool.matched_readout_controls import run_head
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stride_ov_loveda_e1 import make_checkpoints


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing existing smoke output.")
    samples, _, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    reader = GeometryReacquisition(backbone)
    rows = {}
    try:
        for amp in (False, True):
            backbone.use_amp = amp
            prepared = geometry.prepare_image(rgb)
            cached_tokens = prepared.backbone_tokens.clone()
            with backbone._autocast(), torch.inference_mode():
                identity, _ = reader.replay(prepared, torch.ones_like(prepared.geometry_patch_conditional))
                identity_features, _ = run_head(reader.visual.head, identity, identity[:, prepared.prefix_tokens:],
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, "Geometry", prepared.block_index)
                backbone_error = float((identity-cached_tokens).abs().max())
                geometry_error = float((identity_features-prepared.geometry_projected).abs().max())
                if backbone_error != 0. or geometry_error != 0.:
                    raise RuntimeError(f"Uniform-relation replay differs in AMP={amp}: {backbone_error}/{geometry_error}")
                observed, diagnostics = reader.read(prepared)
                torch.cuda.synchronize()
                started = time.perf_counter()
                reader.read(prepared)
                torch.cuda.synchronize()
                rows["bf16" if amp else "fp32"] = {
                    "uniform_backbone_max_error": backbone_error,
                    "uniform_geometry_max_error": geometry_error,
                    "finite_features": all(bool(torch.isfinite(x).all()) for x in observed.values()),
                    "coupled_shape": list(observed[PRIMARY].shape),
                    "diagnostics": diagnostics,
                    "all_arm_cached_read_seconds": time.perf_counter()-started,
                }
        backbone.use_amp = True
        with torch.inference_mode():
            first = geometry.prepare_image(rgb).geometry_projected.clone()
            reader.close()
            second = geometry.prepare_image(rgb).geometry_projected
        error = float((first-second).abs().max())
        if error != 0.:
            raise RuntimeError(f"Native capture changes original Geometry: {error}")
        result = {"status": "complete", "implementation": IMPLEMENTATION,
                  "sample_key": samples[0].key, "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "capture_geometry_max_error": error, "checks": rows}
        output.mkdir(parents=True)
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main(parse_args())
