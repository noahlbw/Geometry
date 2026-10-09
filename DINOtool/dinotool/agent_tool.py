from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .agent_ovss import AgentOVSSConfig, AgentSam3DinoSession
from .cli import build_parser, run_infer, run_infer_dino_dense
from .config import CheckpointConfig, TLPConfig
from .fast_dense import FastDenseConfig, FastDenseSession
from .prompts import class_specs_from_labels, load_class_specs
from .sam3 import Sam3Config


def request_to_argv(request: dict[str, object]) -> list[str]:
    required = ("image", "output_dir")
    missing = [name for name in required if not request.get(name)]
    if missing:
        raise ValueError(f"Missing required request fields: {', '.join(missing)}")
    mode = str(request.get("mode", "dinosplat"))
    dense_mode = mode in {"fast-dense", "dino-dense", "dense"}
    argv = [
        "infer-dino-dense" if dense_mode else "infer",
        "--image",
        str(request["image"]),
        "--output-dir",
        str(request["output_dir"]),
    ]
    if not dense_mode:
        argv.extend(("--mode", mode))
    if request.get("class_config"):
        argv.extend(("--class-config", str(request["class_config"])))
    else:
        classes = request.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("Request must include a non-empty classes list or class_config.")
        argv.append("--classes")
        argv.extend(str(item) for item in classes)
    scalar_options = (
        {
            "confidence_threshold": "--confidence-threshold",
            "device": "--device",
            "code_root": "--code-root",
            "dinov3_repo": "--dinov3-repo",
            "checkpoint_dir": "--checkpoint-dir",
            "dino_input_resolution": "--dino-input-resolution",
            "input_resolution": "--input-resolution",
            "output_temperature": "--output-temperature",
            "image_cache_size": "--image-cache-size",
            "cost_aggregation_checkpoint": "--cost-aggregation-checkpoint",
        }
        if dense_mode
        else {
            "tile_size": "--tile-size",
            "overlap": "--overlap",
            "confidence_threshold": "--confidence-threshold",
            "device": "--device",
            "code_root": "--code-root",
            "dinov3_repo": "--dinov3-repo",
            "checkpoint_dir": "--checkpoint-dir",
            "gsup_steps": "--gsup-steps",
        }
    )
    for key, option in scalar_options.items():
        if request.get(key) is not None:
            argv.extend((option, str(request[key])))
    return argv


def _service_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dino-ovss-agent --serve-sam3-dino",
        description="Persistent dynamic-vocabulary SAM3+DINO OVSS worker using JSONL requests",
    )
    parser.add_argument("--serve-sam3-dino", action="store_true", required=True)
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam3-bpe-path")
    parser.add_argument("--sam3-class-file")
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-probability-threshold", type=float, default=0.5)
    parser.add_argument("--profile", choices=("fast", "balanced", "accurate"), default="fast")
    parser.add_argument("--max-active-classes", type=int)
    parser.add_argument("--max-prompts-per-class", type=int)
    parser.add_argument("--dino-input-resolution", type=int)
    parser.add_argument("--image-cache-size", type=int, default=1)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run a synthetic image through SAM3 and DINO before accepting JSONL requests",
    )
    parser.add_argument("--include-aerial-variants", action="store_true")
    parser.add_argument("--code-root")
    parser.add_argument("--dinov3-repo")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def _dense_service_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dino-ovss-agent --serve-dino-dense",
        description="Persistent one-pass DINOv3 dense OVSS worker using JSONL requests",
    )
    parser.add_argument("--serve-dino-dense", action="store_true", required=True)
    parser.add_argument("--dino-input-resolution", type=int, default=512)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--image-cache-size", type=int, default=1)
    parser.add_argument(
        "--cost-aggregation-checkpoint",
        help="optional external-data CAFe cost decoder; improves quality but is slower",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run a synthetic image through DINO before accepting JSONL requests",
    )
    parser.add_argument("--code-root")
    parser.add_argument("--dinov3-repo")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def _service_config(args: argparse.Namespace) -> AgentOVSSConfig:
    config = AgentOVSSConfig.for_profile(args.profile)
    overrides: dict[str, object] = {"image_cache_size": args.image_cache_size}
    if args.max_active_classes is not None:
        overrides["maximum_active_classes"] = args.max_active_classes
    if args.max_prompts_per_class is not None:
        overrides["maximum_prompts_per_class"] = args.max_prompts_per_class
    if args.dino_input_resolution is not None:
        overrides["dino_input_resolution"] = args.dino_input_resolution
    if args.include_aerial_variants:
        overrides["include_aerial_variants"] = True
    config = replace(config, **overrides)
    config.validate()
    return config


def _service_checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    checkpoints = CheckpointConfig.from_roots(args.code_root, args.checkpoint_dir)
    if args.dinov3_repo:
        checkpoints = replace(checkpoints, dinov3_repo=Path(args.dinov3_repo))
    return checkpoints


def _request_classes(request: dict[str, object]):
    if request.get("class_config"):
        return load_class_specs(str(request["class_config"]))
    classes = request.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Request must include a non-empty classes list or class_config.")
    return class_specs_from_labels([str(item) for item in classes])


def serve_sam3_dino(args: argparse.Namespace) -> None:
    sam3_config = Sam3Config.from_root(
        args.sam3_root,
        checkpoint_path=args.sam3_checkpoint,
        bpe_path=args.sam3_bpe_path,
        class_file=args.sam3_class_file,
        device=args.device,
        resolution=args.sam3_resolution,
        confidence_threshold=args.sam3_confidence_threshold,
        probability_threshold=args.sam3_probability_threshold,
        amp=not args.no_amp,
    )
    session = AgentSam3DinoSession(
        sam3_config,
        _service_checkpoints(args),
        _service_config(args),
        TLPConfig(),
    )
    try:
        if args.warmup:
            session.warmup()
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("The JSON request must be an object.")
                image = request.get("image")
                output_dir = request.get("output_dir")
                if not image or not output_dir:
                    raise ValueError("Missing required request fields: image, output_dir.")
                result = session.run_file(
                    str(image),
                    _request_classes(request),
                    str(output_dir),
                    image_key=str(request["image_id"]) if request.get("image_id") else None,
                )
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response, ensure_ascii=True), flush=True)
    finally:
        session.close()


def serve_dino_dense(args: argparse.Namespace) -> None:
    config = FastDenseConfig(
        input_resolution=args.dino_input_resolution,
        output_temperature=args.output_temperature,
        confidence_threshold=args.confidence_threshold,
        image_cache_size=args.image_cache_size,
    )
    session = FastDenseSession(
        _service_checkpoints(args),
        config,
        device=args.device,
        amp=not args.no_amp,
        cost_aggregation_checkpoint=args.cost_aggregation_checkpoint,
    )
    try:
        if args.warmup:
            session.warmup()
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("The JSON request must be an object.")
                image = request.get("image")
                output_dir = request.get("output_dir")
                if not image or not output_dir:
                    raise ValueError("Missing required request fields: image, output_dir.")
                result = session.run_file(
                    str(image),
                    _request_classes(request),
                    str(output_dir),
                    image_key=str(request["image_id"]) if request.get("image_id") else None,
                )
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response, ensure_ascii=True), flush=True)
    finally:
        session.close()


def main() -> None:
    if "--serve-sam3-dino" in sys.argv[1:]:
        serve_sam3_dino(_service_parser().parse_args())
        return
    if "--serve-dino-dense" in sys.argv[1:]:
        serve_dino_dense(_dense_service_parser().parse_args())
        return
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("The JSON request must be an object.")
        args = build_parser().parse_args(request_to_argv(request))
        runner = run_infer_dino_dense if args.command == "infer-dino-dense" else run_infer
        response = {"ok": True, "result": runner(args)}
    except Exception as error:
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(response, ensure_ascii=True))


if __name__ == "__main__":
    main()
