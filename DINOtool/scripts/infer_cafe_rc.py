#!/usr/bin/env python3
"""Text-query CAFe-RC inference using an adapted checkpoint and the frozen base."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cafedino_locked_loveda import build_model
from dinotool.cafe_rc import CafeRC, CafeRCConfig
from train_cafe_rc import sliding_logits, normalize_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "image", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This CAFe-RC inference entry point requires a CUDA device")
    torch.cuda.set_device(args.device)
    if len(args.classes) != len(set(args.classes)) or len(args.classes) > 65535:
        raise ValueError("Provide distinct query names, at most 65535")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Inference outputs already exist")
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    if payload.get("format") != "cafe_rc_v1":
        raise ValueError("Not a CAFe-RC checkpoint")
    base = Path(args.base_checkpoint)
    if base.stat().st_size != payload["base_checkpoint"]["bytes"]:
        raise ValueError("Frozen base checkpoint size differs from training")
    crop = payload["args"]["crop_size"]
    model_args = argparse.Namespace(official_root=args.official_root, checkpoint=str(base), bpe_path=args.bpe_path,
                                    device=args.device, window_size=crop, model_mode="eval")
    cafe, _, _ = build_model(model_args)
    model = CafeRC(cafe, CafeRCConfig(**payload["rc_config"]), teacher=False).to(args.device).eval()
    model.load_adapted_state_dict(payload["adapted_state"])
    single = len(args.classes) == 1
    names = args.classes + (["background"] if single else [])
    if len(set(names)) != len(names):
        raise ValueError("A single query named background requires an explicit competing class")
    with Image.open(args.image) as handle:
        rgb = np.asarray(handle.convert("RGB")).copy()
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device).float() / 255
    with torch.inference_mode():
        text = cafe.build_text_embeddings([names]).detach()
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = sliding_logits(model, normalize_image(image), text, crop, crop // 2, args.amp)
        prediction = logits.argmax(0).cpu().numpy()
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = ((prediction == 0).astype(np.uint8) * 255) if single else prediction.astype(np.uint8 if len(names) <= 256 else np.uint16)
    Image.fromarray(labels).save(output)
    record = {"model": "CAFe-RC", "classes": names, "weights": args.weights, "input_hw": list(rgb.shape[:2]),
              "seconds": seconds, "output": str(output), "single_query_binary": single,
              "label_mapping": {"0": "background", "255": args.classes[0]} if single else dict(enumerate(names)),
              "note": "Single-query output competes with the explicit background text, not softmax over one class"}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
