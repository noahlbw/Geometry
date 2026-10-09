import unittest
from pathlib import Path

from run_sat_geometry_transport_suite import METHODS, PRIMARY, SETTINGS, exact_miou, launch, promotion_gate


def metric(correct):
    return {"confusion_matrix": [[correct, 100-correct], [100-correct, correct]]}


def outcomes():
    rows = {}
    for setting in SETTINGS:
        dataset = setting[0]
        groups = ("P", "D") if dataset == "loveda" else (dataset,)
        rows[dataset] = {"metrics": {key: {method: metric(85 if method == PRIMARY else 80)
                                         for method in METHODS} for key in groups}}
    return rows


class SATSuiteTests(unittest.TestCase):
    def test_exact_miou_is_computed_from_counts(self):
        self.assertAlmostEqual(exact_miou(metric(80)), 100*80/120)

    def test_launcher_refuses_unauthorized_physical_gpu(self):
        for gpu in range(4):
            with self.assertRaises(ValueError):
                launch(Path("."), SETTINGS[0], 0, gpu, "screen")

    def test_complete_uniformly_better_candidate_passes(self):
        result = promotion_gate(outcomes(), {})
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_missing_dataset_and_run_failure_block_promotion(self):
        rows = outcomes()
        rows.pop("vdd")
        self.assertFalse(promotion_gate(rows, {})["passed"])
        self.assertFalse(promotion_gate(outcomes(), {"worker": "failed"})["passed"])

    def test_no_domain_specific_winner_selection(self):
        rows = outcomes()
        rows["vdd"]["metrics"]["vdd"][PRIMARY] = metric(79)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_loveda_p_regression_is_not_hidden_by_domain_mean(self):
        rows = outcomes()
        rows["loveda"]["metrics"]["P"][PRIMARY] = metric(70)
        result = promotion_gate(rows, {})
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["retain_loveda_P"])

    def test_beating_only_weak_unaligned_control_does_not_pass(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["SAT_UniformTransport"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])


if __name__ == "__main__":
    unittest.main()
