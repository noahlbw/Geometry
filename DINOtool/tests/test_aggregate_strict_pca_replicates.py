"""Contracts for source-only paired PCA-DINO aggregation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strict_aggregate", ROOT / "scripts" / "aggregate_strict_pca_replicates.py")
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def arm(root: Path, name: str, arm_name: str, *, resume: str | None = None) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "best_inference.pt").write_bytes(b"checkpoint")
    write_json(directory / "source_manifest.json", {"coco": {"count": 1}, "oem": {"count": 1}})
    write_json(directory / "training_config.json", {
        "args": {"arm": arm_name, "output_dir": str(directory), "resume": resume,
                 "seed": 7, "warmup_steps": 0, "fod_weight": 0},
        "world_size": 4, "max_global_batch": 32, "base_checkpoint": "/frozen/weights.pth",
    })
    validation = {"validation_step": 2612, "coco": {"mean_iou": 0.5},
                  "oem": {"mean_iou": 0.6}, "selection_score": 0.55}
    write_json(directory / "status.json", {"status": "paused_at_validation", "step": 2612,
                                             "validation": validation, "elapsed_seconds": 1.0})
    (directory / "steps.jsonl").write_text('{"step": 1500}\n{"step": 1520}\n', encoding="utf-8")
    return directory


class StrictAggregateTest(unittest.TestCase):
    def test_clean_pair_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parallel = arm(root, "parallel", "parallel")
            epl = arm(root, "epl", "pca_epl")
            result = aggregate.verify("7", parallel, epl)
            self.assertAlmostEqual(result["epl_minus_parallel"]["source_mean"], 0.0)
            self.assertIsNone(result["protocol"]["resume_exception"])

    def test_resume_requires_explicit_continuity_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parallel = arm(root, "parallel", "parallel")
            epl = arm(root, "epl", "pca_epl", resume=str(root / "epl" / "last.pt"))
            with self.assertRaisesRegex(ValueError, "without approved provenance"):
                aggregate.verify("7", parallel, epl)

    def test_only_hashed_self_resume_boundary_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parallel = arm(root, "parallel", "parallel")
            epl = arm(root, "epl", "pca_epl", resume=str(root / "epl" / "last.pt"))
            log = root / "epl_resume.log"
            log.write_text('{"status": "training", "step": 1501}\n', encoding="utf-8")
            provenance = root / "provenance.json"
            write_json(provenance, {
                "schema": "self_resume_continuity_v1", "seed": "7", "arm": "pca_epl",
                "resume_argument": str(epl / "last.pt"), "resumed_from_step": 1500,
                "first_resumed_step": 1501, "resume_log": str(log),
                "resume_log_sha256": aggregate.sha256(log),
            })
            result = aggregate.verify("7", parallel, epl, resume_provenance=provenance)
            self.assertTrue(result["protocol"]["resume_exception"]["approved"])
            log.write_text('{"status": "training", "step": 1502}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                aggregate.verify("7", parallel, epl, resume_provenance=provenance)

    def test_clean_seed_sensitivity_excludes_resumed_pair(self):
        def row(seed, delta, resumed):
            return {"seed": seed,
                    "epl_minus_parallel": {"coco_miou": delta,
                                           "oem_miou": delta * 2,
                                           "source_mean": delta * 1.5},
                    "protocol": {"resume_exception": {"approved": True} if resumed else None}}

        summary = aggregate.clean_seed_sensitivity([
            row("resumed", 0.50, True), row("clean-a", 0.01, False),
            row("clean-b", 0.03, False),
        ])
        self.assertEqual(summary["clean_pair_count"], 2)
        self.assertEqual(summary["clean_seeds"], ["clean-a", "clean-b"])
        source = summary["aggregate_epl_minus_parallel"]["source_mean"]
        self.assertAlmostEqual(source["mean"], 0.03)
        self.assertAlmostEqual(source["sample_std"], 0.021213203435596427)

    def test_main_writes_resumed_main_and_clean_only_summaries(self):
        def result(seed, delta, resumed):
            return {"seed": seed,
                    "parallel": {"source_mean": 0.5},
                    "pca_epl": {"source_mean": 0.5 + delta},
                    "epl_minus_parallel": {"coco_miou": delta,
                                           "oem_miou": delta * 2,
                                           "source_mean": delta * 1.5},
                    "protocol": {"resume_exception": {"approved": True} if resumed else None}}

        rows = {"20260923": result("20260923", 0.50, True),
                "34117": result("34117", 0.01, False),
                "73531": result("73531", 0.03, False)}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aggregate.json"
            argv = ["aggregate", "--pair", "20260923", "p0", "e0",
                    "--pair", "34117", "p1", "e1", "--pair", "73531", "p2", "e2",
                    "--output", str(output)]
            with patch.object(aggregate, "verify", side_effect=lambda seed, *_args, **_kwargs: rows[seed]), \
                    patch.object(sys, "argv", argv):
                aggregate.main()
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["clean_seed_sensitivity"]["clean_pair_count"], 2)
        self.assertEqual(report["clean_seed_sensitivity"]["clean_seeds"], ["34117", "73531"])
        self.assertAlmostEqual(
            report["aggregate_epl_minus_parallel"]["source_mean"]["mean"], 0.27)
        self.assertAlmostEqual(
            report["clean_seed_sensitivity"]["aggregate_epl_minus_parallel"]["source_mean"]["mean"],
            0.03)


if __name__ == "__main__":
    unittest.main()
