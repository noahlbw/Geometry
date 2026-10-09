#!/usr/bin/env python3
"""Locked P/D LoveDA evaluation and source-audit for CAFe-VC-v2."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

ROOT = Path(os.getenv("OVSS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import (
    ConfusionMatrix, LOVE_DA_FOREGROUND_CLASS_NAMES, build_model, image_tensor, target_ids,
)
from dinotool.cafe_vc import CafeVC, CafeVCConfig
from dinotool.cafe_rs import CafeRS, CafeRSConfig
try:
    from dinotool.cafe_joint import JointCafe, JointConfig
except ModuleNotFoundError:
    JointCafe, JointConfig = None, None
from dinotool.cafe_ped import CafePED
from dinotool.cafe_pca import CafePCA, PCADINOConfig, config_from_checkpoint as pca_config_from_checkpoint
from dinotool.cafe_query_reconstruction import (
    CafeQueryReconstruction,
    QueryReconstructionConfig,
    config_from_checkpoint as query_reconstruction_config_from_checkpoint,
)
from dinotool.cafe_support_reobservation import (
    CafeSupportReobservation,
    SupportReobservationConfig,
    config_from_checkpoint as support_reobservation_config_from_checkpoint,
)
from dinotool.cafe_pair import CafePair, CafePairConfig, config_from_checkpoint as pair_config_from_checkpoint
from dinotool.cafe_region_assembly import CafeRegionAssembly, config_from_checkpoint
from dinotool.region_assembly import RegionAssemblyConfig
from dinotool.cafe_qlift import CafeQLift, config_from_checkpoint as qlift_config_from_checkpoint
from dinotool.qlift import QLiftConfig
from dinotool.cafe_psdr import CafePSDRProbe, config_from_checkpoint as psdr_config_from_checkpoint
from dinotool.pixel_query_evidence import PixelQueryEvidenceConfig
from dinotool.parallel_evidence import ParallelEvidenceConfig
from dinotool.coco_stuff import COCO_CAFE41_NAMES, samples_from_manifest
from dinotool.loveda import LOVEDA_CLASS_NAMES, discover_loveda_samples
from dinotool.ov_train import _write_json_atomic
from dinotool.qlift_spatial_audit import (
    NATIVE_ARGMAX_DIGEST_FILENAME,
    NATIVE_ARGMAX_DIGEST_TEMP_DIRNAME,
    merge_native_argmax_digest_shards,
    native_evaluator_source_hashes,
    native_reference_bindings,
    file_sha256,
    prediction_digest_record,
    remove_temporary_tree,
)
from train_cafe_rc import sliding_logits
from train_cafe_coco import evaluate as evaluate_source


def protocol_settings(protocol: str) -> dict:
    if protocol == "P":
        return dict(evaluation_size=512, window_size=224, stride=112,
                    classes=list(LOVE_DA_FOREGROUND_CLASS_NAMES), include_background=False)
    if protocol == "D":
        return dict(evaluation_size=0, window_size=448, stride=224,
                    classes=list(LOVEDA_CLASS_NAMES), include_background=True)
    if protocol == "source-audit":
        return dict(evaluation_size=224, window_size=224, stride=None,
                    classes=list(COCO_CAFE41_NAMES), include_background=False)
    raise ValueError(f"Unknown fixed protocol: {protocol}")


def checkpoint_config(payload: dict) -> CafeVCConfig | CafeRSConfig | ParallelEvidenceConfig | PCADINOConfig | QueryReconstructionConfig | SupportReobservationConfig | PixelQueryEvidenceConfig | QLiftConfig | object:
    if payload.get("format") == "cafe_support_reobservation_v1":
        return support_reobservation_config_from_checkpoint(payload)
    if payload.get("format") == "cafe_query_reconstruction_v1":
        return query_reconstruction_config_from_checkpoint(payload)
    if payload.get("format") == "cafe_pair_v1":
        return pair_config_from_checkpoint(payload)
    if payload.get("format") == "cafe_pca_v1":
        return pca_config_from_checkpoint(payload)
    if payload.get("format") == "cafe_qlift_v1":
        return qlift_config_from_checkpoint(payload)
    if payload.get("format") == "cafe_region_assembly_v1":
        return config_from_checkpoint(payload)
    if payload.get("format") == "cafe_psdr_probe_v1":
        return psdr_config_from_checkpoint(payload)
    if (payload.get("format"), payload.get("architecture", {}).get("name")) in {
        ("cafe_ped_v1", "CAFe-PED-v1"),
        ("cafe_ped_v2", "CAFe-PED-v2"),
    }:
        arch = payload["architecture"]
        keys = (
            "arm", "evidence_dim", "memory_tokens", "memory_heads", "read_heads", "interaction_dim",
            "message_hidden", "visual_blocks", "local_kernel", "stages",
        )
        if payload.get("format") == "cafe_ped_v2":
            keys += ("pyramid_grids", "support_points", "surround_points")
        config = ParallelEvidenceConfig(**{key: arch[key] for key in keys})
        config.validate()
        return config
    if payload.get("format") == "cafe_joint_v1" and payload["architecture"]["name"] == "CAFe-Joint-v1":
        if JointConfig is None:
            raise RuntimeError("This snapshot cannot evaluate legacy Joint checkpoints because cafe_joint.py is absent")
        arch = payload["architecture"]
        config = JointConfig(**{key: arch[key] for key in ("arm", "visual_blocks", "rs_dim", "concat_dim", "heads", "stages", "modes")})
        config.validate()
        return config
    if payload.get("format") == "cafe_rs_coco_v1" and payload["architecture"]["name"] == "CAFe-RS-v1":
        arch = payload["architecture"]
        config = CafeRSConfig(**{key: arch[key] for key in ("arm", "content_dim", "heads", "stages", "modes")})
        config.validate()
        return config
    if payload.get("format") != "cafe_vc_coco_v2" or payload["architecture"]["name"] != "CAFe-VC-v2":
        raise ValueError("v1 weights cannot be evaluated as the corrected v2 model")
    arch = payload["architecture"]
    config = CafeVCConfig(**{key: arch[key] for key in ("arm", "content_dim", "heads", "kernel_size", "blocks")})
    config.validate()
    return config


class PublishedReadout(nn.Module):
    def __init__(self, cafe):
        super().__init__()
        self.cafe = cafe

    def forward(self, image, text):
        return {"logits": self.cafe(image, text, pre_text_emb=True)}


def model_variant(config) -> str:
    if isinstance(config, PixelQueryEvidenceConfig):
        return f"psdr_{config.address_mode}_{config.content_mode}"
    if isinstance(config, QueryReconstructionConfig):
        return "query_reconstruction"
    return "frozen" if config is None else config.arm


class QLiftAuditReadout(nn.Module):
    """Counterfactuals over Q-Lift's complete query-conditioned P/U pathway.

    The native semantic text remains fixed for the initial cost state and for
    the shared context/detail update inputs. The intervention can still alter
    factorized states and P/U-derived update modulation, so it is not a pure
    analysis/synthesis-only attribution.
    """

    def __init__(self, model: CafeQLift, mode: str) -> None:
        super().__init__()
        if model.config.arm != "query_lift":
            raise ValueError("Q-Lift counterfactuals require a query_lift checkpoint")
        if mode not in {
            "pu-shuffled", "pu-image-only", "details-zero",
            "pu-query-gain-image", "pu-image-gain-query", "pu-image-gain-image",
        }:
            raise ValueError(f"Unknown Q-Lift audit mode: {mode}")
        self.model = model
        self.mode = mode

    def forward(self, image, text):
        arguments = {}
        if self.mode == "pu-shuffled":
            if len(text) < 2:
                raise ValueError("P/U text shuffling needs at least two queries")
            arguments["lifting_text"] = torch.roll(text, shifts=1, dims=0)
        elif self.mode == "pu-image-only":
            arguments["lifting_query_conditioned"] = False
        elif self.mode == "pu-query-gain-image":
            # Keep P/U analysis and cached inverse query-conditioned, but
            # force only the state-update modulation field to image-only.
            arguments["gain_query_conditioned"] = False
        elif self.mode == "pu-image-gain-query":
            # Retain a query-conditioned update gain while P/U itself is
            # image-only. This exposes the previous v1 coupling directly.
            arguments["lifting_query_conditioned"] = False
            arguments["gain_query_conditioned"] = True
        elif self.mode == "pu-image-gain-image":
            arguments["lifting_query_conditioned"] = False
            arguments["gain_query_conditioned"] = False
        elif self.mode == "details-zero":
            arguments["detail_mode"] = "zero"
        return self.model(image, text, **arguments)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-root", "base-checkpoint", "bpe-path", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--weights")
    selection.add_argument("--frozen", action="store_true")
    parser.add_argument("--protocol", choices=("P", "D", "source-audit"), required=True)
    parser.add_argument("--qlift-audit", choices=(
        "native", "pu-shuffled", "pu-image-only", "details-zero",
        "pu-query-gain-image", "pu-image-gain-query", "pu-image-gain-image",
    ),
                        default="native", help="Inference-only complete P/U-pathway counterfactual; never used for selection.")
    parser.add_argument("--native-reference-dir",
                        help="Required for a Q-Lift counterfactual: completed native P/D output to bind checkpoint/sample provenance.")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    return parser.parse_args()


def _sample_key_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _verified_mechanism_reference(
    reference_dir: Path,
    *,
    protocol: str,
    samples: list,
    weights: Path,
    architecture: dict,
) -> dict:
    """Bind a counterfactual to its completed native query-lift evaluation."""

    config_path, result_path = reference_dir / "evaluation_config.json", reference_dir / "results.json"
    if not config_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("Q-Lift counterfactual requires a completed native reference")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(result, dict):
        raise ValueError("Native Q-Lift reference metadata must be JSON objects")
    checks = {
        "config.protocol": (config.get("protocol"), protocol),
        "result.protocol": (result.get("protocol"), protocol),
        "config.qlift_audit": (config.get("qlift_audit"), "native"),
        "result.qlift_audit": (result.get("qlift_audit"), "native"),
        "result.status": (result.get("status"), "complete"),
        "config.model_variant": (config.get("model_variant"), "query_lift"),
        "result.model_variant": (result.get("model_variant"), "query_lift"),
        "config.architecture": (config.get("architecture"), architecture),
        "config.max_images": (config.get("max_images"), None),
        "config.images": (config.get("images"), len(samples)),
        "result.images": (result.get("images"), len(samples)),
        "config.target_used_for_training_or_selection": (config.get("target_used_for_training_or_selection"), False),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Native reference mismatch for mechanism {label}")
    keys = [sample.key for sample in samples]
    if config.get("sample_keys") != keys:
        raise ValueError("Mechanism reference sample ordering differs from this evaluation")
    if not isinstance(config.get("weights"), str) or Path(config["weights"]).resolve() != weights.resolve():
        raise ValueError("Mechanism reference uses a different selected checkpoint path")
    bindings = config.get("integrity_bindings")
    selected = bindings.get("selected_checkpoint") if isinstance(bindings, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("Mechanism reference lacks selected-checkpoint binding")
    if (
        selected.get("path") != str(weights.resolve())
        or selected.get("bytes") != weights.stat().st_size
        or selected.get("sha256") != file_sha256(weights)
    ):
        raise ValueError("Mechanism reference selected checkpoint is not the current Q-Lift checkpoint")
    return {
        "reference_dir": str(reference_dir),
        "evaluation_config_sha256": file_sha256(config_path),
        "results_sha256": file_sha256(result_path),
        "selected_checkpoint": selected,
        "sample_keys_sha256": _sample_key_sha256(keys),
    }


@torch.inference_mode()
def evaluate_loveda(model, samples, text, settings, args, rank, world, device, output, *,
                    native_argmax_digest_dir: Path | None = None):
    """Run locked P/D inference, optionally recording label-free native argmax digests.

    Digest rows are written only to a private temporary DDP shard.  Rank zero
    validates and atomically promotes the merged file only after every rank
    has completed its native run, so neither a partial run nor an integrity
    failure exposes a misleading per-image prediction artifact.
    """

    metrics = ConfusionMatrix(tuple(settings["classes"]))
    local_samples = samples[rank::world]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    digest_stream = None
    try:
        if native_argmax_digest_dir is not None:
            shard = native_argmax_digest_dir / f"rank{rank:02d}.jsonl"
            digest_stream = shard.open("w", encoding="utf-8")
        for index, sample in enumerate(local_samples, 1):
            image = image_tensor(sample.image_path, settings["evaluation_size"], device)
            logits = sliding_logits(model, image, text, settings["window_size"], settings["stride"], args.amp)
            prediction = logits.argmax(0).cpu().numpy()
            if digest_stream is not None:
                sample_index = rank + (index - 1) * world
                digest_stream.write(json.dumps(
                    prediction_digest_record(prediction, sample_index=sample_index, key=sample.key),
                    ensure_ascii=True, separators=(",", ":"),
                ) + "\n")
            # Ground truth is loaded only after this image's prediction is fixed.
            target = target_ids(sample.mask_path, settings["evaluation_size"], include_background=settings["include_background"])
            metrics.update(prediction, target)
            if rank == 0 and (index % 24 == 0 or index == len(local_samples)):
                progress = dict(status="evaluating", rank0_images=index, rank0_total=len(local_samples),
                                elapsed_seconds=time.perf_counter() - started)
                _write_json_atomic(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
    finally:
        if digest_stream is not None:
            digest_stream.close()
    matrix = torch.as_tensor(metrics.matrix, device=device)
    ignored = torch.tensor(metrics.ignored_pixels, dtype=torch.int64, device=device)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
    if world > 1:
        dist.all_reduce(matrix)
        dist.all_reduce(ignored)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    torch.cuda.synchronize(device)
    metrics.matrix, metrics.ignored_pixels = matrix.cpu().numpy(), int(ignored.item())
    summary = metrics.summary()
    summary.update(images=len(samples), seconds=time.perf_counter() - started, max_rank_peak_gib=float(peak.item()))
    if settings["include_background"]:
        summary["foreground_mean_with_background_false_positives"] = float(np.nanmean([
            row["iou"] if row["iou"] is not None else np.nan for row in summary["per_class"][1:]
        ]))
    return summary


def main():
    args = parse_args()
    rank, local_rank, world = (int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    output = Path(args.output_dir).resolve()
    native_argmax_digest_dir: Path | None = None
    mechanism_native_reference: dict | None = None
    try:
        if args.max_images is not None and args.max_images < world:
            raise ValueError("Bounded evaluation needs at least one image per rank")
        settings = protocol_settings(args.protocol)
        root = Path(args.data_root).resolve()
        if args.protocol == "source-audit":
            if root.name != "COCOStuff2017":
                raise ValueError("source-audit requires the locked COCOStuff2017 root")
            samples = samples_from_manifest(root / "manifests/val2017_source_audit.json")
            train = samples_from_manifest(root / "manifests/train2017_cafe41.json")
            dev = samples_from_manifest(root / "manifests/val2017_source_dev.json")
            if {s.key for s in samples} & {s.key for s in train + dev}:
                raise ValueError("Source-audit must not overlap train or source-dev")
            expected_count = 2500
        else:
            if root.name.lower() not in {"val", "validation"} or "loveda" not in str(root).lower():
                raise ValueError("Only external LoveDA validation is allowed")
            samples = discover_loveda_samples(root)
            expected_count = 1669
        if len(samples) != expected_count:
            raise ValueError(f"Locked split needs {expected_count} images, found {len(samples)}")
        if args.max_images:
            samples = samples[:args.max_images]
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Evaluation output must be fresh: {output}")
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        payload = None if args.frozen else torch.load(args.weights, map_location="cpu", weights_only=False)
        config = None if payload is None else checkpoint_config(payload)
        if payload is not None:
            recorded = payload["base_checkpoint"]["checkpoint"]
            if Path(args.base_checkpoint).resolve() != Path(recorded["path"]).resolve() or Path(args.base_checkpoint).stat().st_size != recorded["bytes"]:
                raise ValueError("Base checkpoint differs from the source-training base")
        cafe, _, manifest = build_model(argparse.Namespace(
            official_root=args.official_root, checkpoint=args.base_checkpoint, bpe_path=args.bpe_path,
            device=str(device), window_size=settings["window_size"], model_mode="eval",
        ))
        if payload is None:
            model = PublishedReadout(cafe).to(device).eval()
        else:
            if isinstance(config, PixelQueryEvidenceConfig):
                model = CafePSDRProbe(cafe, config).to(device).eval()
            elif isinstance(config, RegionAssemblyConfig):
                model = CafeRegionAssembly(cafe, config).to(device).eval()
            elif isinstance(config, QLiftConfig):
                model = CafeQLift(cafe, config).to(device).eval()
            elif isinstance(config, ParallelEvidenceConfig):
                model = CafePED(cafe, config).to(device).eval()
            elif isinstance(config, QueryReconstructionConfig):
                model = CafeQueryReconstruction(cafe, config).to(device).eval()
            elif isinstance(config, SupportReobservationConfig):
                model = CafeSupportReobservation(cafe, config).to(device).eval()
            elif isinstance(config, PCADINOConfig):
                model = CafePCA(cafe, config).to(device).eval()
            elif isinstance(config, CafePairConfig):
                model = CafePair(cafe, config).to(device).eval()
            elif JointConfig is not None and isinstance(config, JointConfig):
                model = JointCafe(cafe, config, official_root=args.official_root).to(device).eval()
            else:
                model = (CafeRS(cafe, config, official_root=args.official_root)
                         if isinstance(config, CafeRSConfig) else CafeVC(cafe, config)).to(device).eval()
            model.load_adapted_state_dict(payload["adapted_state"])
        if args.qlift_audit != "native":
            if not isinstance(config, QLiftConfig):
                raise ValueError("--qlift-audit requires a Q-Lift checkpoint")
            if args.protocol not in {"P", "D"}:
                raise ValueError("Q-Lift counterfactuals are only permitted on locked LoveDA P/D")
            if config.arm != "query_lift":
                raise ValueError("Q-Lift counterfactuals require the query_lift checkpoint")
            if not args.native_reference_dir:
                raise ValueError("Q-Lift counterfactuals require --native-reference-dir")
            mechanism_native_reference = _verified_mechanism_reference(
                Path(args.native_reference_dir).resolve(), protocol=args.protocol, samples=samples,
                weights=Path(args.weights).resolve(), architecture=payload["architecture"],
            )
            model = QLiftAuditReadout(model, args.qlift_audit).to(device).eval()
        elif args.native_reference_dir:
            raise ValueError("--native-reference-dir is only valid for a Q-Lift counterfactual")
        with torch.no_grad():
            text = cafe.build_text_embeddings([settings["classes"]]).detach()
        native_argmax_reference = (
            args.protocol in {"P", "D"}
            and isinstance(config, QLiftConfig)
            and args.qlift_audit == "native"
        )
        record = dict(protocol=args.protocol, **settings, amp=args.amp, model_mode="eval", world_size=world,
                      images=len(samples), max_images=args.max_images, background_threshold=None,
                      weights=args.weights, base_checkpoint=manifest, model_variant=model_variant(config),
                      qlift_audit=args.qlift_audit,
                      checkpoint_selection=("mean(COCO source-dev, OEM native-val) mIoU" if isinstance(config, (ParallelEvidenceConfig, PCADINOConfig, CafePairConfig, PixelQueryEvidenceConfig, RegionAssemblyConfig, QLiftConfig)) or (JointConfig is not None and isinstance(config, JointConfig))
                                            else "COCO source-dev only; no target selection" if isinstance(config, QueryReconstructionConfig)
                                            else "COCO source-dev only; no target selection"), target_used_for_training_or_selection=False,
                      sample_keys=[sample.key for sample in samples],
                      architecture=None if payload is None else payload["architecture"])
        if native_argmax_reference:
            if not args.weights:
                raise AssertionError("Native Q-Lift references require a selected checkpoint")
            native_argmax_digest_dir = output / NATIVE_ARGMAX_DIGEST_TEMP_DIRNAME
        if payload is not None and "protocol" in payload.get("source", {}):
            record["source_training_protocol"] = payload["source"]["protocol"]
            record["checkpoint_selection"] = payload["source"].get("selection", record["checkpoint_selection"])
        if rank == 0:
            if native_argmax_reference:
                # Content-hash all 1,669 RGB/mask pairs exactly once per
                # native P/D run. Other DDP ranks wait at the barrier below.
                record["integrity_bindings"] = native_reference_bindings(
                    project_root=ROOT,
                    official_root=Path(args.official_root),
                    selected_checkpoint=Path(args.weights),
                    base_checkpoint=Path(args.base_checkpoint),
                    bpe_path=Path(args.bpe_path),
                    samples=samples,
                    data_root=root,
                )
            if mechanism_native_reference is not None:
                mechanism_native_reference["counterfactual_evaluator_source_sha256"] = native_evaluator_source_hashes(
                    ROOT, Path(args.official_root),
                )
                record["mechanism_native_reference"] = mechanism_native_reference
            _write_json_atomic(output / "evaluation_config.json", record)
            if native_argmax_digest_dir is not None:
                native_argmax_digest_dir.mkdir()
        if world > 1:
            dist.barrier()
        if args.protocol == "source-audit":
            source_args = argparse.Namespace(crop_size=224, batch_size=2, workers=4, amp=args.amp)
            result = evaluate_source(model, samples, text, source_args, rank, world, device)
            result["mean_iou_percent"] = 100 * result["mean_iou"]
        else:
            result = evaluate_loveda(
                model, samples, text, settings, args, rank, world, device, output,
                native_argmax_digest_dir=native_argmax_digest_dir,
            )
        result.update(status="complete", protocol=args.protocol, model_variant=record["model_variant"],
                      weights=args.weights, world_size=world, qlift_audit=args.qlift_audit, **settings)
        if rank == 0:
            if native_argmax_digest_dir is not None:
                record["integrity_bindings"]["native_argmax_digests"] = merge_native_argmax_digest_shards(
                    shard_dir=native_argmax_digest_dir,
                    output_path=output / NATIVE_ARGMAX_DIGEST_FILENAME,
                    sample_keys=record["sample_keys"],
                    world_size=world,
                )
                _write_json_atomic(output / "evaluation_config.json", record)
            _write_json_atomic(output / "results.json", result)
            _write_json_atomic(output / "status.json", {k: result[k] for k in ("status", "images", "mean_iou_percent")})
            print(json.dumps({k: result[k] for k in ("status", "protocol", "model_variant", "images", "mean_iou_percent")}), flush=True)
        if world > 1:
            dist.barrier()
    except BaseException as error:
        if rank == 0 and native_argmax_digest_dir is not None:
            remove_temporary_tree(native_argmax_digest_dir)
        if rank == 0 and output.is_dir() and not (output / "results.json").exists():
            _write_json_atomic(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
