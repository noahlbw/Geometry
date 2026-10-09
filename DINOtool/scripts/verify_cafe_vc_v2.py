#!/usr/bin/env python3
"""Check zero-start equivalence with the real published weights on a source RGB."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import build_model, image_tensor
from dinotool.cafe_vc import CafeVC, CafeVCConfig
from dinotool.coco_stuff import COCO_CAFE41_NAMES, samples_from_manifest
from dinotool.ov_train import _write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-root", "checkpoint", "bpe-path", "data-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    if root.name != "COCOStuff2017":
        raise ValueError("Only a source COCO RGB is permitted for equivalence checks")
    sample = samples_from_manifest(root / "manifests/train2017_cafe41.json")[0]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    args.device, args.window_size, args.model_mode = str(device), 224, "eval"
    cafe, _, _ = build_model(args)
    # Equivalence is a full-FP32 check, not the TF32-enabled throughput setting.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        text = cafe.build_text_embeddings([list(COCO_CAFE41_NAMES)]).detach()
    rows = []
    counts = {}
    for arm in ("plain", "cost_only", "concat", "vc"):
        model = CafeVC(cafe, CafeVCConfig(arm=arm, blocks=0 if arm == "plain" else 2)).to(device).eval()
        counts[arm] = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cafe."))
        cases = [(224, 7, "fp32")]
        if arm == "vc":
            cases += [(224, 1, "fp32"), (224, 41, "fp32"), (448, 7, "fp32"), (224, 7, "bf16")]
        for size, queries, amp in cases:
            image = image_tensor(sample.image_path, size, device)
            model._set_aggregator_resolution(size // 16, size // 16, device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp == "bf16"):
                reference = cafe(image, text[:queries], pre_text_emb=True)
                actual = model(image, text[:queries])["logits"]
            tolerance = 0.02 if amp == "bf16" else 1e-4
            torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
            row = dict(arm=arm, size=size, queries=queries, amp=amp, atol=tolerance, rtol=tolerance,
                       max_abs_error=float((actual.float() - reference.float()).abs().max()))
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model, reference, actual, image
    nonzero_counts = [count for count in counts.values() if count]
    assert max(nonzero_counts) / min(nonzero_counts) < 1.05, counts
    _write_json_atomic(Path(args.output), dict(status="passed", source_rgb=str(sample.image_path),
                                             new_active_parameters=counts, checks=rows))


if __name__ == "__main__":
    main()
