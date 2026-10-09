import unittest
from pathlib import Path

from run_geometry_region_strong_suite import METHODS, PRIMARY, SETTINGS, launch, promotion_gate


def metric(correct):
    return {"confusion_matrix": [[correct, 100-correct], [100-correct, correct]]}


def outcomes():
    return {row[0]: {"metrics": {key: {method: metric(85 if method == PRIMARY else 80)
                                     for method in METHODS}
                                for key in (("P", "D") if row[0] == "loveda" else (row[0],))}}
            for row in SETTINGS}


class StrongSuiteTests(unittest.TestCase):
    def test_unauthorized_physical_gpu_rejected(self):
        for gpu in range(4):
            with self.assertRaises(ValueError):
                launch(Path("."), SETTINGS[0], 0, gpu, "screen")

    def test_uniform_improvement_passes(self):
        self.assertTrue(promotion_gate(outcomes(), {})["passed"])

    def test_missing_dataset_cannot_pass(self):
        rows = outcomes(); rows.pop("oem")
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_failure_cannot_pass(self):
        self.assertFalse(promotion_gate(outcomes(), {"worker": "failed"})["passed"])

    def test_loveda_p_regression_not_hidden(self):
        rows = outcomes()
        rows["loveda"]["metrics"]["P"][PRIMARY] = metric(70)
        result = promotion_gate(rows, {})
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["retain_loveda_P"])

    def test_beating_geometry_without_beating_matched_fusion_is_insufficient(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["RegionFusion"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])


if __name__ == "__main__":
    unittest.main()
