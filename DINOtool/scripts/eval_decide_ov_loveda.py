#!/usr/bin/env python3
"""Training-free DECIDE-OV evaluation on locked LoveDA P/D protocols."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.config import CheckpointConfig, TLPConfig
from dinotool.decide_ov import compose_decide_outputs
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import ClassSpec
from dinotool.tlp import apply_tlp_state, build_tlp_state


METHODS = (
    "B0_raw",
    "B1_tlp",
    "B2_view_average",
    "B3_tlp_average",
    "B4_entropy",
    "B4_confidence",
    "B5_direct_1",
    "M1_replay_1",
    "B5_direct_2",
    "M2_replay_2",
)
P_CLASSES = ("building", "road", "water", "barren", "tree", "farm")
D_CLASSES = ("background", *P_CLASSES)


class Confusion:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.matrix = np.zeros((len(names), len(names)), dtype=np.int64)
        self.ignored = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        valid = (target >= 0) & (target < len(self.names))
        self.ignored += int((~valid).sum())
        encoded = target[valid].astype(np.int64) * len(self.names) + prediction[valid].astype(np.int64)
        self.matrix += np.bincount(encoded, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, object]:
        intersection = np.diag(self.matrix)
        target = self.matrix.sum(1)
        predicted = self.matrix.sum(0)
        union = target + predicted - intersection
        iou = np.divide(intersection, union, out=np.full(len(self.names), np.nan), where=union > 0)
        result = {
            "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
            "pixel_accuracy_percent": round(float(intersection.sum() / max(target.sum(), 1)) * 100.0, 4),
            "per_class": [
                {"name": name, "iou_percent": None if np.isnan(iou[i]) else round(float(iou[i]) * 100.0, 4)}
                for i, name in enumerate(self.names)
            ],
            "confusion_matrix": self.matrix.tolist(),
            "ignored_pixels": self.ignored,
        }
        if self.names and self.names[0] == "background":
            result["foreground_mean_iou_percent"] = round(float(np.nanmean(iou[1:])) * 100.0, 4)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("P", "D", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--context-1", type=int, default=64)
    parser.add_argument("--context-2", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--background-threshold", type=float, default=0.6)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(
        dinov3_repo=Path(args.dinov3_repo).resolve(),
        checkpoint_dir=root,
        dinotxt_weights=root / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        lvd_weights=root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        sat_weights=root / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        bpe_path=root / "bpe_simple_vocab_16e6.txt.gz",
    )


def protocol_classes(protocol: str) -> list[ClassSpec]:
    names = P_CLASSES if protocol == "P" else D_CLASSES
    return [ClassSpec.from_name(name) for name in names]


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def crop_context(image: torch.Tensor, left: int, top: int, tile: int, context: int) -> tuple[torch.Tensor, bool]:
    height, width = image.shape[-2:]
    x0, y0 = left - context, top - context
    x1, y1 = left + tile + context, top + tile + context
    source = image[:, max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)].unsqueeze(0)
    padding = (max(0, -x0), max(0, x1 - width), max(0, -y0), max(0, y1 - height))
    if any(padding):
        source = F.pad(source, padding, mode="replicate")
    expected = tile + 2 * context
    if source.shape[-2:] != (expected, expected):
        raise RuntimeError(f"Context crop has shape {source.shape[-2:]}, expected {(expected, expected)}.")
    return source, not any(padding)


def center_crop(scores: torch.Tensor, context: int, patch_size: int, tile_size: int) -> torch.Tensor:
    border = context // patch_size
    side = tile_size // patch_size
    return scores[..., border : border + side, border : border + side]


@torch.inference_mode()
def predict_image(
    model: DINOTextSegmenter,
    image_path: Path,
    text: dict[str, torch.Tensor],
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    protocols = tuple(text)
    with ExitStack() as stack:
        accumulators = {
            (protocol, method): stack.enter_context(
                ProbabilityAccumulator(len(P_CLASSES if protocol == "P" else D_CLASSES), height, width, 2048, work_dir)
            )
            for protocol in protocols
            for method in METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                observations = []
                valid = []
                for context in (0, args.context_1, args.context_2):
                    rgb, coverage = crop_context(image, left, top, args.tile_size, context)
                    rgb = rgb.to(model.device, non_blocking=True)
                    features, _ = model.encode_image(rgb)
                    observations.append((rgb, features))
                    valid.append(coverage)

                for protocol in protocols:
                    text_features = text[protocol]
                    anchor_rgb, anchor_features = observations[0]
                    anchor = model.similarity_logits(anchor_features, text_features)
                    anchor_state = build_tlp_state(
                        anchor,
                        F.interpolate(anchor_rgb, size=anchor.shape[-2:], mode="area"),
                        text_features,
                        TLPConfig(),
                    )
                    candidate, _ = apply_tlp_state(anchor, anchor_state)
                    held_raw = []
                    held_independent = []
                    held_replay = []
                    for observation_index, context in enumerate((args.context_1, args.context_2), start=1):
                        rgb, features = observations[observation_index]
                        full_raw = model.similarity_logits(features, text_features)
                        own_state = build_tlp_state(
                            full_raw,
                            F.interpolate(rgb, size=full_raw.shape[-2:], mode="area"),
                            text_features,
                            TLPConfig(),
                        )
                        full_independent, _ = apply_tlp_state(full_raw, own_state)
                        raw = center_crop(full_raw, context, model.patch_size, args.tile_size)
                        independent = center_crop(full_independent, context, model.patch_size, args.tile_size)
                        replay, _ = apply_tlp_state(raw, anchor_state)
                        held_raw.append(raw)
                        held_independent.append(independent)
                        held_replay.append(replay)

                    dense = lambda value: F.interpolate(
                        value, size=(args.tile_size, args.tile_size), mode="bilinear", align_corners=False
                    )
                    anchor_dense = dense(anchor)
                    candidate_dense = dense(candidate)
                    raw_dense = [dense(value) for value in held_raw]
                    independent_dense = [dense(value) for value in held_independent]
                    replay_dense = [dense(value) for value in held_replay]
                    for index in range(2):
                        if not valid[index + 1]:
                            raw_dense[index] = anchor_dense
                            independent_dense[index] = candidate_dense
                            replay_dense[index] = anchor_dense
                    outputs = compose_decide_outputs(
                        anchor_dense,
                        candidate_dense,
                        raw_dense,
                        independent_dense,
                        replay_dense,
                        temperature=args.output_temperature,
                    )
                    weights = blend[:actual_h, :actual_w]
                    for method in METHODS:
                        probabilities = torch.softmax(outputs.logits[method].float() / args.output_temperature, dim=1)
                        accumulators[(protocol, method)].add(
                            probabilities[0, :, :actual_h, :actual_w].cpu().numpy(), weights, left, top
                        )

        predictions: dict[str, dict[str, np.ndarray]] = {protocol: {} for protocol in protocols}
        for protocol in protocols:
            for method in METHODS:
                labels, confidence = accumulators[(protocol, method)].finalize(None)
                if protocol == "D":
                    labels[confidence < round(args.background_threshold * 255)] = 0
                predictions[protocol][method] = labels
        return predictions


def target_ids(path: Path, protocol: str) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    if raw.ndim == 3:
        raw = raw[..., 0]
    result = np.full(raw.shape, 255, dtype=np.uint8)
    if protocol == "P":
        valid = (raw >= 2) & (raw <= 7)
        result[valid] = raw[valid] - 2
    else:
        valid = (raw >= 1) & (raw <= 7)
        result[valid] = raw[valid] - 1
    return result


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(prediction, mode="L").save(temporary)
    temporary.replace(path)


def load_prediction(path: Path, shape: tuple[int, int], classes: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        prediction = np.asarray(Image.open(path)).copy()
    except OSError:
        return None
    if prediction.shape != shape or prediction.ndim != 2 or np.any(prediction >= classes):
        return None
    return prediction


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or args.context_1 % 16 or args.context_2 % 16:
        raise ValueError("tile and context sizes must be multiples of the DINO patch size.")
    if not 0 <= args.overlap < args.tile_size:
        raise ValueError("overlap must be smaller than tile size.")
    protocols = ("P", "D") if args.protocol == "both" else (args.protocol,)
    samples = discover_loveda_samples(args.data_root)
    if args.max_images is not None:
        random.Random(args.sample_seed).shuffle(samples)
        samples = samples[: args.max_images]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    model = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    text = {protocol: model.encode_text(protocol_classes(protocol)) for protocol in protocols}
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in protocols for method in METHODS
    }
    changes = {
        protocol: {method: {"updated": 0, "beneficial": 0, "harmful": 0, "wrong_to_wrong": 0} for method in METHODS}
        for protocol in protocols
    }
    signature = {
        "method": "DECIDE-OV training-free prototype",
        "sample_count": len(samples),
        "sample_keys_sha256": hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest(),
        "protocols": list(protocols),
        "methods": list(METHODS),
        "config": {key: value for key, value in vars(args).items() if key not in {"device"}},
        "tlp": asdict(TLPConfig()),
        "checkpoints": checkpoint_manifest(checkpoints),
        "note": "Original frozen DINOv3/DINO.text, fixed bilinear upsampling, no GSUP and no learned parameters.",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory has a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {protocol: {} for protocol in protocols}
        complete = True
        for protocol in protocols:
            classes = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in METHODS:
                path = output / "predictions" / protocol / method / sample.key
                loaded = load_prediction(path, shape, classes)
                if loaded is None:
                    complete = False
                else:
                    predictions[protocol][method] = loaded
        if not complete:
            predictions = predict_image(model, sample.image_path, text, args, work)
            for protocol in protocols:
                for method in METHODS:
                    save_prediction(output / "predictions" / protocol / method / sample.key, predictions[protocol][method])
        for protocol in protocols:
            target = target_ids(sample.mask_path, protocol)
            valid = target != 255
            baseline = predictions[protocol]["B0_raw"]
            for method in METHODS:
                prediction = predictions[protocol][method]
                matrices[(protocol, method)].update(prediction, target)
                updated = valid & (prediction != baseline)
                changes[protocol][method]["updated"] += int(updated.sum())
                changes[protocol][method]["beneficial"] += int((updated & (baseline != target) & (prediction == target)).sum())
                changes[protocol][method]["harmful"] += int((updated & (baseline == target) & (prediction != target)).sum())
                changes[protocol][method]["wrong_to_wrong"] += int(
                    (updated & (baseline != target) & (prediction != target)).sum()
                )
        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": {
                    protocol: {method: matrices[(protocol, method)].summary() for method in METHODS}
                    for protocol in protocols
                },
                "change_audit": changes,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2),
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "total": len(samples),
                "mIoU": {
                    protocol: {method: result["metrics"][protocol][method]["mean_iou_percent"] for method in METHODS}
                    for protocol in protocols
                },
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
