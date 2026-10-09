from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .agent_ovss import AgentOVSSConfig, AgentSam3DinoSession
from .config import CheckpointConfig, GSUPConfig, InferenceConfig, TLPConfig
from .cost_aggregation import CostAggregatedDINOTextSegmenter
from .cost_train import OemCostTrainingConfig, train_oem_cost_aggregation
from .fast_dense import FastDenseConfig, FastDenseSession
from .inference import InferenceRunner
from .loveda import LoveDABenchmark, SUPPORTED_MODES, default_loveda_classes
from .mask_ov import MaskOVDINOTextSegmenter
from .mask_train import MaskOemTrainingConfig, train_oem_mask_ov
from .model import DINOTextSegmenter, checkpoint_manifest
from .ov_adapter import AdaptedDINOTextSegmenter
from .ov_train import OemTrainingConfig, train_oem_adapter
from .prompts import class_specs_from_labels, load_class_specs
from .region_verifier import RegionVerifierConfig
from .sam3 import Sam3Config, Sam3GeometryConfig, Sam3LoveDABenchmark, SegEarthOV3Predictor
from .sam3_dino import Sam3DinoFusionConfig, Sam3DinoLoveDABenchmark, Sam3DinoPredictor
from .uot import UOTAggregatedDINOTextSegmenter
from .uot_train import OemUOTTrainingConfig, train_oem_uot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dino-ovss",
        description="Training-free DINOv3 open-vocabulary segmentation for remote sensing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="segment an image or GeoTIFF")
    infer.add_argument("--image", required=True)
    class_group = infer.add_mutually_exclusive_group(required=True)
    class_group.add_argument("--classes", nargs="+", help="ordered class names")
    class_group.add_argument("--class-config", help="JSON class and synonym configuration")
    infer.add_argument("--output-dir", required=True)
    infer_checkpoints = infer.add_mutually_exclusive_group()
    infer_checkpoints.add_argument(
        "--adapter-checkpoint",
        help="optional external-data DINOv3-SAT open-vocabulary adapter checkpoint",
    )
    infer_checkpoints.add_argument(
        "--cost-aggregation-checkpoint",
        help="optional external-data DINOv3 dino.txt class-agnostic cost-aggregation checkpoint",
    )
    infer_checkpoints.add_argument(
        "--mask-ov-checkpoint",
        help="optional external-data DINOv3 multi-scale mask-query open-vocabulary checkpoint",
    )
    infer_checkpoints.add_argument(
        "--uot-checkpoint",
        help="optional external-data DINOv3 multi-prototype unbalanced-OT checkpoint",
    )
    _add_method_arguments(infer, benchmark=False)

    agent_sam3_infer = subparsers.add_parser(
        "infer-sam3-dino",
        help="dynamic-vocabulary, agent-oriented SAM3 + DINOv3 OVSS for one RGB image",
    )
    agent_sam3_infer.add_argument("--image", required=True)
    agent_class_group = agent_sam3_infer.add_mutually_exclusive_group(required=True)
    agent_class_group.add_argument("--classes", nargs="+", help="foreground class names; background is added when absent")
    agent_class_group.add_argument("--class-config", help="JSON class and synonym configuration")
    agent_sam3_infer.add_argument("--output-dir", required=True)
    agent_sam3_infer.add_argument("--sam3-root", required=True, help="official SegEarth-OV3 repository root")
    agent_sam3_infer.add_argument("--sam3-checkpoint")
    agent_sam3_infer.add_argument("--sam3-bpe-path")
    agent_sam3_infer.add_argument("--sam3-class-file")
    agent_sam3_infer.add_argument("--sam3-resolution", type=int, default=1008)
    agent_sam3_infer.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    agent_sam3_infer.add_argument("--sam3-probability-threshold", type=float, default=0.5)
    agent_sam3_infer.add_argument("--profile", choices=("fast", "balanced", "accurate"), default="fast")
    agent_sam3_infer.add_argument(
        "--max-active-classes",
        type=int,
        help="maximum foreground classes routed to SAM3; zero retains every supplied class",
    )
    agent_sam3_infer.add_argument(
        "--max-prompts-per-class",
        type=int,
        help="maximum dynamic SAM3 prompts for each routed class",
    )
    agent_sam3_infer.add_argument("--dino-input-resolution", type=int, help="DINO long-side resolution")
    agent_sam3_infer.add_argument("--image-cache-size", type=int, default=1)
    agent_sam3_infer.add_argument("--include-aerial-variants", action="store_true")
    _add_tlp_arguments(agent_sam3_infer)
    _add_dino_checkpoint_arguments(agent_sam3_infer)
    agent_sam3_infer.add_argument("--device", default="cuda")
    agent_sam3_infer.add_argument("--no-amp", action="store_true")

    fast_dense_infer = subparsers.add_parser(
        "infer-dino-dense",
        help="one-pass, SAM3-free DINOv3 dense OVSS for one RGB image",
    )
    fast_dense_infer.add_argument("--image", required=True)
    fast_dense_class_group = fast_dense_infer.add_mutually_exclusive_group(required=True)
    fast_dense_class_group.add_argument(
        "--classes",
        nargs="+",
        help="foreground class names; background is added when absent",
    )
    fast_dense_class_group.add_argument("--class-config", help="JSON class and synonym configuration")
    fast_dense_infer.add_argument("--output-dir", required=True)
    fast_dense_infer.add_argument(
        "--dino-input-resolution",
        "--input-resolution",
        dest="dino_input_resolution",
        type=int,
        default=512,
        help="DINO long-side resolution; lower values trade detail for latency",
    )
    fast_dense_infer.add_argument("--output-temperature", type=float, default=0.07)
    fast_dense_infer.add_argument("--confidence-threshold", type=float)
    fast_dense_infer.add_argument("--image-cache-size", type=int, default=1)
    fast_dense_infer.add_argument(
        "--cost-aggregation-checkpoint",
        help="optional external-data CAFe cost decoder; improves quality but is slower",
    )
    _add_dino_checkpoint_arguments(fast_dense_infer)
    fast_dense_infer.add_argument("--device", default="cuda")
    fast_dense_infer.add_argument("--no-amp", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark-loveda",
        help="evaluate one or more modes on the labeled LoveDA validation split",
    )
    benchmark.add_argument("--data-root", required=True)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--class-config", help="optional LoveDA class/synonym override")
    benchmark.add_argument("--max-images", type=int, help="deterministic prefix for smoke tests")
    benchmark.add_argument("--progress-every", type=int, default=10)
    benchmark_checkpoints = benchmark.add_mutually_exclusive_group()
    benchmark_checkpoints.add_argument(
        "--adapter-checkpoint",
        help="optional external-data DINOv3-SAT open-vocabulary adapter checkpoint",
    )
    benchmark_checkpoints.add_argument(
        "--cost-aggregation-checkpoint",
        help="optional external-data DINOv3 dino.txt class-agnostic cost-aggregation checkpoint",
    )
    benchmark_checkpoints.add_argument(
        "--mask-ov-checkpoint",
        help="optional external-data DINOv3 multi-scale mask-query open-vocabulary checkpoint",
    )
    benchmark_checkpoints.add_argument(
        "--uot-checkpoint",
        help="optional external-data DINOv3 multi-prototype unbalanced-OT checkpoint",
    )
    _add_method_arguments(benchmark, benchmark=True)

    train_oem = subparsers.add_parser(
        "train-oem-dino",
        help="train a DINOv3-SAT open-vocabulary feature adapter on external OpenEarthMap labels",
    )
    train_oem.add_argument("--data-root", required=True, help="unpacked official OpenEarthMap root containing train.txt and val.txt")
    train_oem.add_argument("--output-dir", required=True)
    train_oem.add_argument("--train-split", default="train")
    train_oem.add_argument("--val-split", default="val")
    train_oem.add_argument("--crop-size", type=int, default=512)
    train_oem.add_argument("--batch-size", type=int, default=8)
    train_oem.add_argument("--epochs", type=int, default=20)
    train_oem.add_argument("--lr", type=float, default=2e-4)
    train_oem.add_argument("--weight-decay", type=float, default=0.01)
    train_oem.add_argument("--warmup-steps", type=int, default=200)
    train_oem.add_argument("--num-workers", type=int, default=8)
    train_oem.add_argument("--adapter-hidden-dim", type=int, default=256)
    train_oem.add_argument("--adapter-context-blocks", type=int, default=2)
    train_oem.add_argument("--output-temperature", type=float, default=0.07)
    train_oem.add_argument("--dice-weight", type=float, default=0.5)
    train_oem.add_argument(
        "--class-balance-power",
        type=float,
        default=0.5,
        help="inverse-frequency exponent for source class weights; zero disables balancing",
    )
    train_oem.add_argument("--class-balance-max-weight", type=float, default=4.0)
    train_oem.add_argument("--label-smoothing", type=float, default=0.02)
    train_oem.add_argument(
        "--open-vocabulary-preservation-weight",
        type=float,
        default=0.25,
        help="preserve frozen dino.txt scores for auxiliary remote-sensing concepts",
    )
    train_oem.add_argument(
        "--ignored-feature-preservation-weight",
        type=float,
        default=0.1,
        help="preserve frozen features at ignored/unlabeled source pixels",
    )
    train_oem.add_argument("--gradient-clip", type=float, default=1.0)
    train_oem.add_argument("--seed", type=int, default=3407)
    train_oem.add_argument("--max-train-images", type=int, help="deterministic OpenEarthMap train subset for smoke tests")
    train_oem.add_argument("--max-val-images", type=int, help="deterministic OpenEarthMap validation subset for smoke tests")
    train_oem.add_argument(
        "--allow-missing-source-images",
        action="store_true",
        help="allow only the known omitted xBD RGB files in the official OpenEarthMap_wo_xBD archive",
    )
    train_oem.add_argument("--resume-checkpoint", help="resume from a compatible adapter checkpoint")
    _add_runtime_arguments(train_oem)

    train_oem_cost = subparsers.add_parser(
        "train-oem-cost-dino",
        help="train a DINOv3 dino.txt class-agnostic cost-aggregation decoder on external OpenEarthMap labels",
    )
    train_oem_cost.add_argument(
        "--data-root",
        required=True,
        help="unpacked official OpenEarthMap root containing train.txt and val.txt",
    )
    train_oem_cost.add_argument("--output-dir", required=True)
    train_oem_cost.add_argument("--train-split", default="train")
    train_oem_cost.add_argument("--val-split", default="val")
    train_oem_cost.add_argument("--crop-size", type=int, default=512)
    train_oem_cost.add_argument("--batch-size", type=int, default=8)
    train_oem_cost.add_argument("--epochs", type=int, default=30)
    train_oem_cost.add_argument("--lr", type=float, default=2e-4)
    train_oem_cost.add_argument("--visual-lr", type=float, default=1e-5)
    train_oem_cost.add_argument("--weight-decay", type=float, default=0.01)
    train_oem_cost.add_argument("--warmup-steps", type=int, default=300)
    train_oem_cost.add_argument("--num-workers", type=int, default=8)
    train_oem_cost.add_argument("--cost-hidden-dim", type=int, default=128)
    train_oem_cost.add_argument("--cost-context-blocks", type=int, default=6)
    train_oem_cost.add_argument("--cost-attention-heads", type=int, default=8)
    train_oem_cost.add_argument("--cost-window-size", type=int, default=7)
    train_oem_cost.add_argument("--cost-dropout", type=float, default=0.1)
    train_oem_cost.add_argument(
        "--train-last-visual-blocks",
        type=int,
        default=2,
        help="number of final DINO visual blocks to tune; the text tower and vision head remain frozen",
    )
    train_oem_cost.add_argument("--output-temperature", type=float, default=0.07)
    train_oem_cost.add_argument("--dice-weight", type=float, default=0.5)
    train_oem_cost.add_argument("--class-balance-power", type=float, default=0.5)
    train_oem_cost.add_argument("--class-balance-max-weight", type=float, default=4.0)
    train_oem_cost.add_argument("--label-smoothing", type=float, default=0.02)
    train_oem_cost.add_argument("--open-vocabulary-preservation-weight", type=float, default=0.15)
    train_oem_cost.add_argument("--ignored-cost-preservation-weight", type=float, default=0.1)
    train_oem_cost.add_argument("--gradient-clip", type=float, default=1.0)
    train_oem_cost.add_argument("--seed", type=int, default=3407)
    train_oem_cost.add_argument("--max-train-images", type=int, help="deterministic OpenEarthMap train subset for smoke tests")
    train_oem_cost.add_argument("--max-val-images", type=int, help="deterministic OpenEarthMap validation subset for smoke tests")
    train_oem_cost.add_argument(
        "--allow-missing-source-images",
        action="store_true",
        help="allow only the known omitted xBD RGB files in the official OpenEarthMap_wo_xBD archive",
    )
    train_oem_cost.add_argument("--resume-checkpoint", help="resume from a compatible cost-aggregation checkpoint")
    _add_runtime_arguments(train_oem_cost)

    train_oem_uot_parser = subparsers.add_parser(
        "train-oem-uot",
        help="train a multi-prototype unbalanced-OT DINOv3 OVSS decoder on external OpenEarthMap labels",
    )
    train_oem_uot_parser.add_argument("--data-root", required=True)
    train_oem_uot_parser.add_argument("--output-dir", required=True)
    train_oem_uot_parser.add_argument("--train-split", default="train")
    train_oem_uot_parser.add_argument("--val-split", default="val")
    train_oem_uot_parser.add_argument("--crop-size", type=int, default=512)
    train_oem_uot_parser.add_argument("--batch-size", type=int, default=8)
    train_oem_uot_parser.add_argument("--epochs", type=int, default=30)
    train_oem_uot_parser.add_argument("--lr", type=float, default=2e-4)
    train_oem_uot_parser.add_argument("--visual-lr", type=float, default=1e-5)
    train_oem_uot_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_oem_uot_parser.add_argument("--warmup-steps", type=int, default=300)
    train_oem_uot_parser.add_argument("--num-workers", type=int, default=8)
    train_oem_uot_parser.add_argument("--uot-num-modes", type=int, default=4)
    train_oem_uot_parser.add_argument("--uot-epsilon", type=float, default=0.08)
    train_oem_uot_parser.add_argument("--uot-marginal-relaxation", type=float, default=0.5)
    train_oem_uot_parser.add_argument("--uot-unknown-prior", type=float, default=0.20)
    train_oem_uot_parser.add_argument("--uot-unknown-cost", type=float, default=0.80)
    train_oem_uot_parser.add_argument("--uot-mode-offset-scale", type=float, default=0.08)
    train_oem_uot_parser.add_argument("--uot-base-score-weight", type=float, default=0.35)
    train_oem_uot_parser.add_argument("--uot-transport-score-weight", type=float, default=1.0)
    train_oem_uot_parser.add_argument("--uot-sinkhorn-iterations", type=int, default=15)
    train_oem_uot_parser.add_argument("--uot-mode-diversity-weight", type=float, default=0.02)
    train_oem_uot_parser.add_argument("--train-last-visual-blocks", type=int, default=2)
    train_oem_uot_parser.add_argument("--output-temperature", type=float, default=0.07)
    train_oem_uot_parser.add_argument("--dice-weight", type=float, default=0.5)
    train_oem_uot_parser.add_argument("--class-balance-power", type=float, default=0.5)
    train_oem_uot_parser.add_argument("--class-balance-max-weight", type=float, default=4.0)
    train_oem_uot_parser.add_argument("--label-smoothing", type=float, default=0.02)
    train_oem_uot_parser.add_argument("--open-vocabulary-preservation-weight", type=float, default=0.15)
    train_oem_uot_parser.add_argument("--unknown-supervision-weight", type=float, default=0.10)
    train_oem_uot_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_oem_uot_parser.add_argument("--seed", type=int, default=3407)
    train_oem_uot_parser.add_argument("--max-train-images", type=int)
    train_oem_uot_parser.add_argument("--max-val-images", type=int)
    train_oem_uot_parser.add_argument("--allow-missing-source-images", action="store_true")
    train_oem_uot_parser.add_argument("--resume-checkpoint")
    _add_runtime_arguments(train_oem_uot_parser)

    train_oem_mask = subparsers.add_parser(
        "train-oem-mask-ov",
        help="train a DINOv3 multi-scale class-agnostic mask-query OVSS decoder on external OpenEarthMap labels",
    )
    train_oem_mask.add_argument(
        "--data-root",
        required=True,
        help="unpacked official OpenEarthMap root containing train.txt and val.txt",
    )
    train_oem_mask.add_argument("--output-dir", required=True)
    train_oem_mask.add_argument("--train-split", default="train")
    train_oem_mask.add_argument("--val-split", default="val")
    train_oem_mask.add_argument("--crop-size", type=int, default=768)
    train_oem_mask.add_argument("--batch-size", type=int, default=4)
    train_oem_mask.add_argument("--epochs", type=int, default=20)
    train_oem_mask.add_argument("--lr", type=float, default=1e-4)
    train_oem_mask.add_argument("--visual-lr", type=float, default=5e-6)
    train_oem_mask.add_argument("--weight-decay", type=float, default=0.01)
    train_oem_mask.add_argument("--warmup-steps", type=int, default=300)
    train_oem_mask.add_argument("--num-workers", type=int, default=8)
    train_oem_mask.add_argument("--adapter-hidden-dim", type=int, default=512)
    train_oem_mask.add_argument("--adapter-context-blocks", type=int, default=4)
    train_oem_mask.add_argument("--fusion-dim", type=int, default=256)
    train_oem_mask.add_argument("--num-queries", type=int, default=64)
    train_oem_mask.add_argument("--attention-heads", type=int, default=8)
    train_oem_mask.add_argument("--query-layers", type=int, default=2)
    train_oem_mask.add_argument("--dropout", type=float, default=0.1)
    train_oem_mask.add_argument("--feature-layers", type=int, nargs="+", default=(5, 11, 17, 23))
    train_oem_mask.add_argument("--residual-scale", type=float, default=0.05)
    train_oem_mask.add_argument("--output-temperature", type=float, default=0.07)
    train_oem_mask.add_argument("--dice-weight", type=float, default=0.5)
    train_oem_mask.add_argument("--class-balance-power", type=float, default=0.5)
    train_oem_mask.add_argument("--class-balance-max-weight", type=float, default=4.0)
    train_oem_mask.add_argument("--label-smoothing", type=float, default=0.02)
    train_oem_mask.add_argument("--query-mask-weight", type=float, default=1.0)
    train_oem_mask.add_argument("--query-class-weight", type=float, default=0.25)
    train_oem_mask.add_argument("--semantic-mask-weight", type=float, default=1.0)
    train_oem_mask.add_argument("--open-vocabulary-preservation-weight", type=float, default=0.15)
    train_oem_mask.add_argument("--ignored-feature-preservation-weight", type=float, default=0.1)
    train_oem_mask.add_argument("--gradient-clip", type=float, default=1.0)
    train_oem_mask.add_argument(
        "--train-last-visual-blocks",
        type=int,
        default=0,
        help="number of final DINO visual blocks to tune; text tower and vision head remain frozen",
    )
    train_oem_mask.add_argument("--seed", type=int, default=3407)
    train_oem_mask.add_argument("--max-train-images", type=int, help="deterministic train subset for smoke tests")
    train_oem_mask.add_argument("--max-val-images", type=int, help="deterministic validation subset for smoke tests")
    train_oem_mask.add_argument(
        "--allow-missing-source-images",
        action="store_true",
        help="allow only the known omitted xBD RGB files in the official OpenEarthMap_wo_xBD archive",
    )
    train_oem_mask.add_argument(
        "--init-adapter-checkpoint",
        help="initialize the dense SAT adapter from an existing external-data adapter checkpoint",
    )
    train_oem_mask.add_argument("--resume-checkpoint", help="resume from a compatible mask OV checkpoint")
    _add_runtime_arguments(train_oem_mask)

    sam3_benchmark = subparsers.add_parser(
        "benchmark-loveda-sam3",
        help="evaluate the official SegEarth-OV3 SAM 3 pipeline on labeled LoveDA validation",
    )
    sam3_benchmark.add_argument("--data-root", required=True)
    sam3_benchmark.add_argument("--output-dir", required=True)
    sam3_benchmark.add_argument("--sam3-root", required=True, help="official SegEarth-OV3 repository root")
    sam3_benchmark.add_argument("--sam3-checkpoint")
    sam3_benchmark.add_argument("--sam3-bpe-path")
    sam3_benchmark.add_argument("--sam3-class-file")
    sam3_benchmark.add_argument("--sam3-resolution", type=int, default=1008)
    sam3_benchmark.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    sam3_benchmark.add_argument("--sam3-probability-threshold", type=float, default=0.5)
    sam3_benchmark.add_argument("--max-images", type=int, help="deterministic prefix for smoke tests")
    sam3_benchmark.add_argument("--progress-every", type=int, default=10)
    sam3_benchmark.add_argument("--device", default="cuda")
    sam3_benchmark.add_argument("--no-amp", action="store_true")

    sam3_dino_benchmark = subparsers.add_parser(
        "benchmark-loveda-sam3-dino",
        help="evaluate experimental training-free SAM 3 plus DINOv3 fusion on labeled LoveDA validation",
    )
    sam3_dino_benchmark.add_argument("--data-root", required=True)
    sam3_dino_benchmark.add_argument("--output-dir", required=True)
    sam3_dino_benchmark.add_argument("--sam3-root", required=True, help="official SegEarth-OV3 repository root")
    sam3_dino_benchmark.add_argument("--sam3-checkpoint")
    sam3_dino_benchmark.add_argument("--sam3-bpe-path")
    sam3_dino_benchmark.add_argument("--sam3-class-file")
    sam3_dino_benchmark.add_argument("--sam3-resolution", type=int, default=1008)
    sam3_dino_benchmark.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    sam3_dino_benchmark.add_argument("--sam3-probability-threshold", type=float, default=0.5)
    sam3_dino_benchmark.add_argument("--dino-input-resolution", type=int, default=1024)
    sam3_dino_benchmark.add_argument("--sam3-posterior-temperature", type=float, default=0.25)
    sam3_dino_benchmark.add_argument("--dino-posterior-temperature", type=float, default=0.07)
    sam3_dino_benchmark.add_argument("--dino-semantic-strength", type=float, default=0.0)
    sam3_dino_benchmark.add_argument(
        "--dino-structure-strength",
        type=float,
        default=0.0,
        help="legacy posterior-fusion metadata only; SAT use is controlled by --no-dino-sat",
    )
    sam3_dino_benchmark.add_argument("--dino-uncertainty-power", type=float, default=1.0)
    sam3_dino_benchmark.add_argument(
        "--dino-tlp",
        action="store_true",
        help="enable experimental DINO Text-aware Laplacian Propagation for region evidence",
    )
    sam3_dino_benchmark.add_argument("--no-dino-tlp", action="store_true", help=argparse.SUPPRESS)
    sam3_dino_benchmark.add_argument(
        "--legacy-pixel-fusion",
        action="store_true",
        help="run the retired posterior-blending experiment instead of region verification",
    )
    sam3_dino_benchmark.add_argument(
        "--constrained-prompts",
        action="store_true",
        help="add experimental aerial prompt variants after the official SegEarth aliases",
    )
    sam3_dino_benchmark.add_argument("--no-constrained-prompts", action="store_true", help=argparse.SUPPRESS)
    sam3_dino_benchmark.add_argument(
        "--sam3-max-prompts-per-class",
        type=int,
        default=5,
        help="target prompt limit; official SegEarth aliases are always retained",
    )
    sam3_dino_benchmark.add_argument("--pgrf", action="store_true", help="enable experimental PGRF aggregation")
    sam3_dino_benchmark.add_argument("--no-pgrf", action="store_true", help=argparse.SUPPRESS)
    sam3_dino_benchmark.add_argument(
        "--geometry-reprompt",
        action="store_true",
        help="enable experimental SAM3 box re-prompting for text-grounded components",
    )
    sam3_dino_benchmark.add_argument("--no-geometry-reprompt", action="store_true", help=argparse.SUPPRESS)
    sam3_dino_benchmark.add_argument("--geometry-probability-threshold", type=float, default=0.60)
    sam3_dino_benchmark.add_argument("--geometry-minimum-area", type=int, default=80)
    sam3_dino_benchmark.add_argument("--geometry-max-area-ratio", type=float, default=0.25)
    sam3_dino_benchmark.add_argument("--no-dino-region-verifier", action="store_true")
    sam3_dino_benchmark.add_argument("--region-minimum-area", type=int, default=64)
    sam3_dino_benchmark.add_argument("--region-sam-confident-score", type=float, default=0.72)
    sam3_dino_benchmark.add_argument("--region-sam-ambiguous-margin", type=float, default=0.12)
    sam3_dino_benchmark.add_argument("--region-dino-minimum-margin", type=float, default=0.015)
    sam3_dino_benchmark.add_argument("--no-dino-prototypes", action="store_true")
    sam3_dino_benchmark.add_argument(
        "--dino-sat",
        action="store_true",
        help="load the experimental frozen SAT structure encoder (requires --dino-tlp)",
    )
    sam3_dino_benchmark.add_argument("--no-dino-sat", action="store_true", help=argparse.SUPPRESS)
    sam3_dino_benchmark.add_argument("--max-images", type=int, help="deterministic prefix for smoke tests")
    sam3_dino_benchmark.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based LoveDA sample offset, applied before --max-images for development/hold-out splits",
    )
    sam3_dino_benchmark.add_argument("--progress-every", type=int, default=10)
    _add_tlp_arguments(sam3_dino_benchmark)
    _add_dino_checkpoint_arguments(sam3_dino_benchmark)
    sam3_dino_benchmark.add_argument("--device", default="cuda")
    sam3_dino_benchmark.add_argument("--no-amp", action="store_true")

    inspect = subparsers.add_parser("inspect", help="check runtime and local checkpoints")
    _add_runtime_arguments(infer)
    _add_runtime_arguments(benchmark)
    _add_runtime_arguments(inspect)
    return parser


def _add_method_arguments(parser: argparse.ArgumentParser, benchmark: bool) -> None:
    if benchmark:
        parser.add_argument(
            "--modes",
            nargs="+",
            choices=SUPPORTED_MODES,
            default=("baseline", "tlp", "dinosplat"),
        )
    else:
        parser.add_argument("--mode", choices=SUPPORTED_MODES, default="dinosplat")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--bands", type=int, nargs=3, default=(1, 2, 3), metavar=("R", "G", "B"))
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--no-global-anchor", action="store_true")
    parser.add_argument("--global-anchor-temperature", type=float, default=0.07)
    parser.add_argument("--global-anchor-sigma", type=float, default=0.5)
    parser.add_argument("--max-in-memory-mb", type=int, default=2048)

    _add_tlp_arguments(parser)

    parser.add_argument("--gsup-steps", type=int, default=10)
    parser.add_argument("--gsup-lr", type=float, default=0.05)
    parser.add_argument("--gsup-neighbors", type=int, default=16)
    parser.add_argument("--gsup-optimization-size", type=int, default=256)
    parser.add_argument("--gsup-chunk-pixels", type=int, default=65536)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dino_checkpoint_arguments(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")


def _add_dino_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-root", default=os.environ.get("DINO_CODE_ROOT"))
    parser.add_argument("--dinov3-repo", default=os.environ.get("DINOV3_REPO"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("DINO_CHECKPOINT_DIR"))


def _add_tlp_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tlp-strength", type=float, default=0.25)
    parser.add_argument("--tlp-semantic-temperature", type=float, default=0.07)
    parser.add_argument("--tlp-probability-temperature", type=float, default=0.07)
    parser.add_argument("--tlp-diagonal-boost", type=float, default=1.0)
    parser.add_argument("--tlp-cg-iterations", type=int, default=50)
    parser.add_argument("--tlp-cg-tolerance", type=float, default=1e-4)


def make_checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    checkpoints = CheckpointConfig.from_roots(args.code_root, args.checkpoint_dir)
    if args.dinov3_repo:
        checkpoints = replace(checkpoints, dinov3_repo=Path(args.dinov3_repo))
    return checkpoints


def run_infer(args: argparse.Namespace) -> dict[str, object]:
    classes = load_class_specs(args.class_config) if args.class_config else class_specs_from_labels(args.classes)
    inference = make_inference_config(args, args.mode)
    tlp = make_tlp_config(args)
    gsup = make_gsup_config(args)
    model = make_segmentation_model(args, use_satellite=inference.use_satellite)
    return InferenceRunner(model, inference, tlp, gsup).run(args.image, classes, args.output_dir)


def run_infer_dino_dense(args: argparse.Namespace) -> dict[str, object]:
    classes = load_class_specs(args.class_config) if args.class_config else class_specs_from_labels(args.classes)
    config = FastDenseConfig(
        input_resolution=args.dino_input_resolution,
        output_temperature=args.output_temperature,
        confidence_threshold=args.confidence_threshold,
        image_cache_size=args.image_cache_size,
    )
    session = FastDenseSession(
        make_checkpoints(args),
        config,
        device=args.device,
        amp=not args.no_amp,
        cost_aggregation_checkpoint=args.cost_aggregation_checkpoint,
    )
    try:
        return session.run_file(args.image, classes, args.output_dir)
    finally:
        session.close()


def make_agent_ovss_config(args: argparse.Namespace) -> AgentOVSSConfig:
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


def run_agent_sam3_dino(args: argparse.Namespace) -> dict[str, object]:
    classes = load_class_specs(args.class_config) if args.class_config else class_specs_from_labels(args.classes)
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
        make_checkpoints(args),
        make_agent_ovss_config(args),
        make_tlp_config(args),
    )
    try:
        return session.run_file(args.image, classes, args.output_dir)
    finally:
        session.close()


def run_loveda(args: argparse.Namespace) -> dict[str, object]:
    classes = load_class_specs(args.class_config) if args.class_config else default_loveda_classes()
    modes = tuple(dict.fromkeys(args.modes))
    representative_mode = "dinosplat-sat" if "dinosplat-sat" in modes else modes[-1]
    inference = make_inference_config(args, representative_mode)
    if inference.confidence_threshold is not None:
        raise ValueError("LoveDA benchmarking requires no confidence threshold so every labeled pixel is evaluated.")
    model = make_segmentation_model(args, use_satellite="dinosplat-sat" in modes)
    benchmark = LoveDABenchmark(
        model,
        inference,
        make_tlp_config(args),
        make_gsup_config(args),
        modes,
    )
    return benchmark.run(
        args.data_root,
        classes,
        args.output_dir,
        max_images=args.max_images,
        progress_every=args.progress_every,
    )


def run_loveda_sam3(args: argparse.Namespace) -> dict[str, object]:
    config = Sam3Config.from_root(
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
    benchmark = Sam3LoveDABenchmark(SegEarthOV3Predictor(config))
    return benchmark.run(
        args.data_root,
        args.output_dir,
        max_images=args.max_images,
        progress_every=args.progress_every,
    )


def run_train_oem_dino(args: argparse.Namespace) -> dict[str, object]:
    config = OemTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        hidden_dim=args.adapter_hidden_dim,
        context_blocks=args.adapter_context_blocks,
        output_temperature=args.output_temperature,
        dice_weight=args.dice_weight,
        class_balance_power=args.class_balance_power,
        class_balance_max_weight=args.class_balance_max_weight,
        label_smoothing=args.label_smoothing,
        open_vocabulary_preservation_weight=args.open_vocabulary_preservation_weight,
        ignored_feature_preservation_weight=args.ignored_feature_preservation_weight,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        allow_missing_source_images=args.allow_missing_source_images,
        amp=not args.no_amp,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )
    return train_oem_adapter(make_checkpoints(args), config)


def run_train_oem_cost_dino(args: argparse.Namespace) -> dict[str, object]:
    config = OemCostTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        visual_learning_rate=args.visual_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        hidden_dim=args.cost_hidden_dim,
        context_blocks=args.cost_context_blocks,
        attention_heads=args.cost_attention_heads,
        window_size=args.cost_window_size,
        dropout=args.cost_dropout,
        train_last_visual_blocks=args.train_last_visual_blocks,
        output_temperature=args.output_temperature,
        dice_weight=args.dice_weight,
        class_balance_power=args.class_balance_power,
        class_balance_max_weight=args.class_balance_max_weight,
        label_smoothing=args.label_smoothing,
        open_vocabulary_preservation_weight=args.open_vocabulary_preservation_weight,
        ignored_cost_preservation_weight=args.ignored_cost_preservation_weight,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        allow_missing_source_images=args.allow_missing_source_images,
        amp=not args.no_amp,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )
    return train_oem_cost_aggregation(make_checkpoints(args), config)


def run_train_oem_uot(args: argparse.Namespace) -> dict[str, object]:
    config = OemUOTTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        visual_learning_rate=args.visual_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        num_modes=args.uot_num_modes,
        epsilon=args.uot_epsilon,
        marginal_relaxation=args.uot_marginal_relaxation,
        unknown_prior=args.uot_unknown_prior,
        unknown_cost=args.uot_unknown_cost,
        mode_offset_scale=args.uot_mode_offset_scale,
        base_score_weight=args.uot_base_score_weight,
        transport_score_weight=args.uot_transport_score_weight,
        sinkhorn_iterations=args.uot_sinkhorn_iterations,
        mode_diversity_weight=args.uot_mode_diversity_weight,
        train_last_visual_blocks=args.train_last_visual_blocks,
        output_temperature=args.output_temperature,
        dice_weight=args.dice_weight,
        class_balance_power=args.class_balance_power,
        class_balance_max_weight=args.class_balance_max_weight,
        label_smoothing=args.label_smoothing,
        open_vocabulary_preservation_weight=args.open_vocabulary_preservation_weight,
        unknown_supervision_weight=args.unknown_supervision_weight,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        allow_missing_source_images=args.allow_missing_source_images,
        amp=not args.no_amp,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )
    return train_oem_uot(make_checkpoints(args), config)


def run_train_oem_mask_ov(args: argparse.Namespace) -> dict[str, object]:
    config = MaskOemTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        visual_learning_rate=args.visual_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_context_blocks=args.adapter_context_blocks,
        fusion_dim=args.fusion_dim,
        num_queries=args.num_queries,
        attention_heads=args.attention_heads,
        query_layers=args.query_layers,
        dropout=args.dropout,
        feature_layers=tuple(args.feature_layers),
        residual_scale=args.residual_scale,
        output_temperature=args.output_temperature,
        dice_weight=args.dice_weight,
        class_balance_power=args.class_balance_power,
        class_balance_max_weight=args.class_balance_max_weight,
        label_smoothing=args.label_smoothing,
        query_mask_weight=args.query_mask_weight,
        query_class_weight=args.query_class_weight,
        semantic_mask_weight=args.semantic_mask_weight,
        open_vocabulary_preservation_weight=args.open_vocabulary_preservation_weight,
        ignored_feature_preservation_weight=args.ignored_feature_preservation_weight,
        gradient_clip=args.gradient_clip,
        train_last_visual_blocks=args.train_last_visual_blocks,
        seed=args.seed,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        allow_missing_source_images=args.allow_missing_source_images,
        amp=not args.no_amp,
        device=args.device,
        init_adapter_checkpoint=args.init_adapter_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )
    return train_oem_mask_ov(make_checkpoints(args), config)


def make_segmentation_model(args: argparse.Namespace, *, use_satellite: bool):
    checkpoints = make_checkpoints(args)
    adapter_checkpoint = getattr(args, "adapter_checkpoint", None)
    cost_checkpoint = getattr(args, "cost_aggregation_checkpoint", None)
    mask_checkpoint = getattr(args, "mask_ov_checkpoint", None)
    uot_checkpoint = getattr(args, "uot_checkpoint", None)
    selected = [path for path in (adapter_checkpoint, cost_checkpoint, mask_checkpoint, uot_checkpoint) if path]
    if len(selected) > 1:
        raise ValueError(
            "Choose only one of --adapter-checkpoint, --cost-aggregation-checkpoint, --mask-ov-checkpoint, or --uot-checkpoint."
        )
    if mask_checkpoint:
        return MaskOVDINOTextSegmenter(
            checkpoints,
            mask_checkpoint,
            device=args.device,
            amp=not args.no_amp,
        )
    if cost_checkpoint:
        if use_satellite:
            raise ValueError(
                "The DINO cost-aggregation checkpoint is DINO-only and does not support dinosplat-sat. "
                "Use baseline, tlp, or dinosplat."
            )
        return CostAggregatedDINOTextSegmenter(
            checkpoints,
            cost_checkpoint,
            device=args.device,
            amp=not args.no_amp,
        )
    if uot_checkpoint:
        if use_satellite:
            raise ValueError(
                "The DINO UOT checkpoint is DINO-only and does not support dinosplat-sat. "
                "Use baseline, tlp, or dinosplat."
            )
        return UOTAggregatedDINOTextSegmenter(
            checkpoints,
            uot_checkpoint,
            device=args.device,
            amp=not args.no_amp,
        )
    if adapter_checkpoint:
        return AdaptedDINOTextSegmenter(
            checkpoints,
            adapter_checkpoint,
            device=args.device,
            amp=not args.no_amp,
        )
    return DINOTextSegmenter(
        checkpoints,
        device=args.device,
        amp=not args.no_amp,
        use_satellite=use_satellite,
    )


def run_loveda_sam3_dino(args: argparse.Namespace) -> dict[str, object]:
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
    geometry = Sam3GeometryConfig(
        enabled=getattr(args, "geometry_reprompt", False),
        probability_threshold=getattr(args, "geometry_probability_threshold", 0.60),
        minimum_component_area=getattr(args, "geometry_minimum_area", 80),
        maximum_mask_area_ratio=getattr(args, "geometry_max_area_ratio", 0.25),
    )
    region_verifier = RegionVerifierConfig(
        enabled=not getattr(args, "no_dino_region_verifier", False),
        minimum_region_area=getattr(args, "region_minimum_area", 64),
        sam_confident_score=getattr(args, "region_sam_confident_score", 0.72),
        sam_ambiguous_margin=getattr(args, "region_sam_ambiguous_margin", 0.12),
        minimum_dino_margin=getattr(args, "region_dino_minimum_margin", 0.015),
        use_per_image_prototypes=not getattr(args, "no_dino_prototypes", False),
    )
    fusion_config = Sam3DinoFusionConfig(
        dino_input_resolution=args.dino_input_resolution,
        use_constrained_prompts=getattr(args, "constrained_prompts", False),
        maximum_prompts_per_class=getattr(args, "sam3_max_prompts_per_class", 5),
        use_pgrf=getattr(args, "pgrf", False),
        geometry=geometry,
        region_verifier=region_verifier,
        use_satellite_structure=getattr(args, "dino_sat", False),
        legacy_pixel_fusion=getattr(args, "legacy_pixel_fusion", False),
        sam3_posterior_temperature=args.sam3_posterior_temperature,
        dino_posterior_temperature=args.dino_posterior_temperature,
        semantic_strength=args.dino_semantic_strength,
        structure_strength=args.dino_structure_strength,
        uncertainty_power=args.dino_uncertainty_power,
        use_dino_tlp=getattr(args, "dino_tlp", False),
    )
    predictor = Sam3DinoPredictor(
        sam3_config,
        make_checkpoints(args),
        fusion_config,
        make_tlp_config(args),
    )
    return Sam3DinoLoveDABenchmark(predictor).run(
        args.data_root,
        args.output_dir,
        max_images=args.max_images,
        start_index=args.start_index,
        progress_every=args.progress_every,
    )


def make_inference_config(args: argparse.Namespace, mode: str) -> InferenceConfig:
    return InferenceConfig(
        mode=mode,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_temperature=args.output_temperature,
        confidence_threshold=args.confidence_threshold,
        use_global_anchor=not args.no_global_anchor,
        global_anchor_temperature=args.global_anchor_temperature,
        global_anchor_sigma=args.global_anchor_sigma,
        max_in_memory_mb=args.max_in_memory_mb,
        bands=tuple(args.bands),
        amp=not args.no_amp,
    )


def make_tlp_config(args: argparse.Namespace) -> TLPConfig:
    return TLPConfig(
        smoothing_strength=args.tlp_strength,
        semantic_temperature=args.tlp_semantic_temperature,
        probability_temperature=args.tlp_probability_temperature,
        diagonal_boost=args.tlp_diagonal_boost,
        cg_max_iterations=args.tlp_cg_iterations,
        cg_tolerance=args.tlp_cg_tolerance,
    )


def make_gsup_config(args: argparse.Namespace) -> GSUPConfig:
    return GSUPConfig(
        steps=args.gsup_steps,
        learning_rate=args.gsup_lr,
        neighbors=args.gsup_neighbors,
        optimization_size=args.gsup_optimization_size,
        chunk_pixels=args.gsup_chunk_pixels,
    )


def run_inspect(args: argparse.Namespace) -> dict[str, object]:
    checkpoints = make_checkpoints(args)
    required = checkpoints.required(use_satellite=True)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "required_files": {str(path): path.exists() for path in required},
        "checkpoints": checkpoint_manifest(checkpoints),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "infer":
        result = run_infer(args)
    elif args.command == "infer-sam3-dino":
        result = run_agent_sam3_dino(args)
    elif args.command == "infer-dino-dense":
        result = run_infer_dino_dense(args)
    elif args.command == "benchmark-loveda":
        result = run_loveda(args)
    elif args.command == "train-oem-dino":
        result = run_train_oem_dino(args)
    elif args.command == "train-oem-cost-dino":
        result = run_train_oem_cost_dino(args)
    elif args.command == "train-oem-uot":
        result = run_train_oem_uot(args)
    elif args.command == "train-oem-mask-ov":
        result = run_train_oem_mask_ov(args)
    elif args.command == "benchmark-loveda-sam3":
        result = run_loveda_sam3(args)
    elif args.command == "benchmark-loveda-sam3-dino":
        result = run_loveda_sam3_dino(args)
    else:
        result = run_inspect(args)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
