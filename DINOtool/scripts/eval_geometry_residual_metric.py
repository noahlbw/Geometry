"""Complete-image frozen Geometry residual metric evaluation and mask-free smoke."""
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_residual_metric import (
    IMPLEMENTATION, PRIMARY, METHODS, ResidualMetricConfig, ResidualMetricReader,
    metric_alias_scores, residual_covariance)
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_cross_view_value_innovation import head_fingerprint, parse_args
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints
import eval_frozen_semantic_path as evaluator


CONFIG = ResidualMetricConfig()


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError("Refusing existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
        TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    reader = ResidualMetricReader(geometry, banks)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    yy = (torch.arange(32, device=geometry.device)+.5)*16
    valid = ((yy[:, None] < min(512, image.shape[-2])) & (yy[None] < min(512, image.shape[-1]))).reshape(1, -1)
    fingerprint = head_fingerprint(geometry.backbone.model.visual_model.head)
    errors = {}
    for key, text in reader.texts.items():
        response, _ = metric_alias_scores(prepared.geometry_projected, text, reader.basis,
            torch.zeros((1, reader.basis.shape[1], reader.basis.shape[1]), device=geometry.device), valid)
        errors[key+"_zero_metric_max_error"] = float((response-prepared.geometry_projected.float() @ text.T).abs().max())
    covariance = residual_covariance(prepared.geometry_projected.float() @ reader.basis,
        torch.eye(1024, device=geometry.device)[None], valid)
    errors["identity_geometry_covariance_max_error"] = float(covariance.abs().max())
    started = time.perf_counter()
    scores, diagnostics = reader(geometry, prepared, banks, reader.texts, valid)
    torch.cuda.synchronize()
    if any(errors.values()) or fingerprint != head_fingerprint(geometry.backbone.model.visual_model.head):
        raise RuntimeError("Original metric/head identity failed.")
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": CONFIG.signature(),
        "errors": errors, "diagnostics": diagnostics, "sample_key": sample.key,
        "target_masks_loaded": False, "methods_finite": all(bool(torch.isfinite(value).all())
            for group in scores.values() for value in group.values()),
        "weights_frozen": all(not p.requires_grad for p in geometry.backbone.model.parameters()),
        "head_weights_unchanged": True, "head_sha256": fingerprint,
        "wall_seconds": time.perf_counter()-started}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke(args)
    else:
        evaluator.main(args, methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
            reader_factory=ResidualMetricReader, readout_settings=CONFIG.signature(), save_per_image=True,
            signature_note="Original Geometry plus bounded joint visual/text residual metric, all20 aliases. "
                "No extra teacher/view or label fitting. Covariance is window-specific inference adaptation, "
                "not semantic reliability certification. Controls include attributed nearest operator adaptations. "
                "Fixed complete-image96 development screen, not full official-VIP reproduction.")
