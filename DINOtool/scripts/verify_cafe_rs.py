#!/usr/bin/env python3
"""Real-weight zero-start equivalence for RS, using one COCO train RGB only."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import build_model, image_tensor
from dinotool.cafe_rs import CafeRS, CafeRSConfig
from dinotool.coco_stuff import COCO_CAFE41_NAMES, samples_from_manifest
from dinotool.ov_train import _write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("official-root", "checkpoint", "bpe-path", "data-root", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if Path(args.data_root).name != "COCOStuff2017":
        raise ValueError("Equivalence uses source COCO only")
    sample = samples_from_manifest(Path(args.data_root) / "manifests/train2017_cafe41.json")[0]
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    args.device, args.window_size, args.model_mode = "cuda:0", 224, "eval"
    cafe, _, _ = build_model(args)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = CafeRS(cafe, CafeRSConfig(), official_root=args.official_root).cuda().eval()
    with torch.no_grad():
        text = cafe.build_text_embeddings([list(COCO_CAFE41_NAMES)]).detach()
    rows = []
    for size, queries, amp, rectangular in [(224, 1, "fp32", False), (224, 7, "fp32", False), (224, 41, "fp32", False), (448, 7, "fp32", False), (224, 7, "fp32", True), (224, 41, "bf16", False)]:
        image = image_tensor(sample.image_path, size, torch.device("cuda:0"))
        if rectangular:
            image = image[..., :112, :]
        model._set_aggregator_resolution(image.shape[-2] // 16, image.shape[-1] // 16, image.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp == "bf16"):
            reference = cafe(image, text[:queries], pre_text_emb=True)
            actual = model(image, text[:queries])["logits"]
        tolerance = 0.02 if amp == "bf16" else 1e-4
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        row = dict(shape=list(image.shape), queries=queries, amp=amp, tolerance=tolerance,
                   max_abs_error=float((actual.float() - reference.float()).abs().max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write_json_atomic(Path(args.output), dict(status="passed", source_rgb=str(sample.image_path), architecture=model.architecture(), checks=rows))


if __name__ == "__main__":
    main()
