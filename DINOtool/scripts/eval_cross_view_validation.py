"""Complete-image fixed native-wide cross-view validation experiment."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from dinotool.cross_view_validation import (
    CONFIG, IMPLEMENTATION, METHODS, PRIMARY, assemble_native_view,
    native_patch_features, padded_crop, shifted_starts, validated_scores,
)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
import eval_frozen_semantic_path as evaluator
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_matched_head_fov import resize_rgb
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def predict_image(image, geometry, banks, work, methods=METHODS, reader=None, *, reader_uses_rgb=False):
    h, w = image.shape[-2:]
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    resized = resize_rgb(image, CONFIG["wide_long_edge"])
    rh, rw = resized.shape[-2:]
    broad1, crops1 = assemble_native_view(resized, geometry, banks, texts,
                                         tile_starts(rh, 336, 224), tile_starts(rw, 336, 224))
    broad2, crops2 = assemble_native_view(resized, geometry, banks, texts,
                                         shifted_starts(rh), shifted_starts(rw))
    totals = {key: {"tiles": 0} for key in banks}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for key, bank in banks.items() for method in methods}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                ah, aw = min(512, h-top), min(512, w-left)
                yy = (torch.arange(32, device=geometry.device)+.5)*16
                xx = (torch.arange(32, device=geometry.device)+.5)*16
                valid = ((yy[:, None] < ah) & (xx[None] < aw)).reshape(1, -1)
                for key, bank in banks.items():
                    g = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                           bank.parent_indices, bank.class_count)
                    w1 = sample_broad(broad1[key], top, left, h, w).reshape_as(g)
                    w2 = sample_broad(broad2[key], top, left, h, w).reshape_as(g)
                    scores, diagnostic = validated_scores(g, w1, w2, prepared.geometry_patch_conditional, valid)
                    for method in methods:
                        dense = F.interpolate(scores[method].transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(),
                                                     blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in diagnostic.items():
                        totals[key][field] = totals[key].get(field, 0.)+value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in methods} for key in banks}
    for values in totals.values():
        values["wide_crops_view1_per_tile"] = crops1
        values["wide_crops_view2_per_tile"] = crops2
    return predictions, totals


def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
                             TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    image = load_image(samples[0].image_path if args.dataset == "loveda" else samples[0])
    rgb, _ = padded_crop(resize_rgb(image, 448), -56, -56)
    native = native_patch_features(geometry.backbone, rgb[None])
    prepared = geometry.prepare_image(rgb[None].to(geometry.device))
    official, _ = geometry.backbone.encode_image(rgb[None])
    replay = float((native-official.flatten(2).transpose(1, 2)).abs().max())
    manual_attention_difference = float((native-prepared.native_projected).abs().max())
    if replay > 2e-6 or not all(not p.requires_grad for p in geometry.backbone.model.parameters()):
        raise RuntimeError("Native replay/freeze check failed: "+str(replay))
    output.mkdir(parents=True)
    # Timing uses the actual complete candidate and excludes model/text loading.
    primary, diagnostic = predict_image(image, geometry, banks, output, (PRIMARY,))
    timings = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        current, _ = predict_image(image, geometry, banks, output, (PRIMARY,))
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
        if any(not (current[key][PRIMARY] == primary[key][PRIMARY]).all() for key in banks):
            raise RuntimeError("Nondeterministic full-image primary replay.")
    result = {"status": "complete", "implementation": IMPLEMENTATION, "target_masks_loaded": False,
              "weights_frozen": True, "native_replay_max_error": replay,
              "tcpr_manual_attention_difference": manual_attention_difference,
              "sample_key": samples[0].key, "image_shape": list(image.shape),
              "primary_image_seconds": timings, "diagnostics": diagnostic,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main(args):
    evaluator.predict_image = predict_image
    evaluator.main(args, methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
                   reader_factory=lambda geometry, banks: None, save_per_image=True,
                   readout_settings={"cross_view": CONFIG},
                   signature_note="Frozen native head, all20 RS bank and original Geometry. "
                   "Two offset wide crop grids, analytic opposite-view validation; no masks in prediction. "
                   "No VIP operator. Fixed96 complete-image development screen; correlated views do not certify correctness.")


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    args = parse_args()
    (smoke if is_smoke else main)(args)
