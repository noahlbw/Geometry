"""Real-checkpoint feature-state identity checks without loading any target masks."""
import json
from pathlib import Path
import time

import torch

from dinotool.gear_ov import _crop_at
from dinotool.matched_readout_controls import run_head
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.sat_geometry_transport import (
    IMPLEMENTATION, PRIMARY, SATGeometryTransport, native_patch_states, paired_transport,
)
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
    reader = SATGeometryTransport(backbone)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        tokens = prepared.backbone_tokens.clone()
        raw = native_patch_states(prepared)
        valid = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
        identity, _ = reader.read_from_satellite(prepared, raw, valid, controls=False)
        identity_error = float((identity[PRIMARY]-prepared.geometry_projected).abs().max())
        with backbone._autocast():
            baselines = {method: run_head(backbone.model.visual_model.head, tokens, raw,
                         prepared.geometry_patch_conditional, prepared.prefix_tokens,
                         method, prepared.block_index)[0]
                         for method in ("Geometry", "SCLIP_Two", "VIPProxy_Two")}
        torch.cuda.synchronize()
        started = time.perf_counter()
        features, diagnostics = reader.read(prepared, rgb, valid)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        baseline_error = max(float((features[method]-baselines[method]).abs().max())
                             for method in baselines)
        cached_error = float((prepared.backbone_tokens-tokens).abs().max())
        repeat_error = float((geometry.prepare_image(rgb).geometry_projected-prepared.geometry_projected).abs().max())
        padded = valid.clone()
        padded[:, ::7] = False
        fake_satellite = torch.randn_like(raw)
        padded_transport, _ = paired_transport(fake_satellite, raw, padded.float(), padded)
        padding_error = float((padded_transport[~padded]-raw[~padded]).abs().max())
        errors = (identity_error, baseline_error, cached_error, repeat_error, padding_error)
        if any(error != 0. for error in errors):
            raise RuntimeError(f"Native state/head invariant failed AMP={amp}: {errors}")
        if (any(not bool(torch.isfinite(value).all()) for value in features.values())
                or any(value.shape != prepared.geometry_projected.shape for value in features.values())
                or any(parameter.requires_grad for parameter in reader.satellite.parameters())):
            raise RuntimeError("Nonfinite/wrong-shaped features or unfrozen SAT weights.")
        if max(diagnostics["rotation_orthogonality_error"], diagnostics["uniform_rotation_orthogonality_error"]) > 2e-5:
            raise RuntimeError(f"FP32 alignment failed orthogonality accuracy check: {diagnostics}")
        checks["bf16" if amp else "fp32"] = {
            "same_source_head_max_error": identity_error,
            "matched_baseline_max_error": baseline_error,
            "cached_backbone_max_error": cached_error,
            "geometry_repeat_max_error": repeat_error,
            "padding_state_max_error": padding_error,
            "native_state_mean_norm": float(raw.float().norm(dim=-1).mean()),
            "structural_descriptor_mean_norm": float(prepared.raw_patch_tokens.norm(dim=-1).mean()),
            "finite_features": True, "coupled_shape": list(features[PRIMARY].shape),
            "diagnostics": diagnostics, "all_arm_cached_read_seconds": elapsed,
        }
    result = {"status": "complete", "implementation": IMPLEMENTATION,
              "sample_key": samples[0].key, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "target_masks_loaded": False, "checks": checks}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
