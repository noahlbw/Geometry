"""Actual-checkpoint decoder controls and primary cost, without loading masks."""
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_demixing import (IMPLEMENTATION, PRIMARY, DemixConfig,
    JointSemanticDecoder, read_semantic_demixing)
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
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    decoders = {key: JointSemanticDecoder(bank.features, bank.parent_indices, bank.class_count)
                for key, bank in banks.items()}
    rgb = _crop_at(image, 0, 0, 512).to(backbone.device)
    yy = (torch.arange(32, device=backbone.device)+.5)*16
    valid = ((yy[:, None] < min(image.shape[-2], 512)) & (yy[None] < min(image.shape[-1], 512))).reshape(1, -1)
    checks = {}
    for amp in (False, True):
        backbone.use_amp = amp
        prepared = geometry.prepare_image(rgb)
        cache, original = prepared.backbone_tokens.clone(), prepared.geometry_projected.clone()
        scores, diagnostics = read_semantic_demixing(geometry, prepared, banks, texts, valid, decoders)
        identity_errors = []
        for key, decoder in decoders.items():
            local, _ = decoder.decode(prepared.geometry_projected[0], None, valid[0])
            identity, _ = decoder.decode(prepared.geometry_projected[0], torch.eye(1024, device=backbone.device), valid[0])
            identity_errors.append(float((local-identity).abs().max()))
        error = max(identity_errors)
        finite = all(bool(torch.isfinite(value).all()) for group in scores.values() for value in group.values())
        cache_error = float((cache-prepared.backbone_tokens).abs().max())
        geometry_error = float((original-prepared.geometry_projected).abs().max())
        if (error != 0. or cache_error != 0. or geometry_error != 0. or not finite
                or diagnostics["solver_kkt_relative"] > DemixConfig().kkt_tolerance):
            raise RuntimeError("Actual-checkpoint semantic decoder controls failed.")
        checks["bf16" if amp else "fp32"] = {"identity_to_text_control_max_error": error,
            "native_cache_max_error": cache_error, "geometry_baseline_max_error": geometry_error,
            "finite_scores": finite, "diagnostics": diagnostics}
    backbone.use_amp = True
    del prepared, cache, scores, original, local, identity
    cost = {}
    for method in ("Geometry", PRIMARY):
        times = []
        for _ in range(2):
            prepared = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_semantic_demixing(geometry, prepared, banks, texts, valid, decoders, controls=False)
            else:
                for key, bank in banks.items():
                    alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                       bank.parent_indices, bank.class_count)
            del prepared
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            prepared = geometry.prepare_image(rgb)
            if method == PRIMARY:
                read_semantic_demixing(geometry, prepared, banks, texts, valid, decoders, controls=False)
            else:
                for key, bank in banks.items():
                    alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                       bank.parent_indices, bank.class_count)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
            del prepared
        cost[method] = {"window_median_seconds": statistics.median(times), "runs_seconds": times,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated()/1048576}
    result = {"status": "complete", "implementation": IMPLEMENTATION, "primary": PRIMARY,
        "sample_key": samples[0].key, "config": DemixConfig().signature(), "target_masks_loaded": False,
        "checks": checks, "weights_frozen": all(not p.requires_grad for p in backbone.model.parameters()),
        "deployed_window_cost": cost,
        "cost_note": "2 warmups/5 synchronized windows, no control heads in primary timing. "
                     "Includes Geometry prepare and decoding; text dictionaries stay resident; not full-image latency."}
    output.mkdir(parents=True)
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
