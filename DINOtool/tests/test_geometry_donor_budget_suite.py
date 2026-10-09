import unittest
from pathlib import Path

from run_geometry_donor_budget_suite import METHODS, PRIMARY, SETTINGS, counts, launch, promotion_gate


def metric(correct):
    return {"confusion_matrix": [[correct, 100-correct], [100-correct, correct]]}


def outcomes():
    return {row[0]: {"metrics": {key: {method: metric(85 if method == PRIMARY else 80) for method in METHODS}
                                for key in (("P", "D") if row[0] == "loveda" else (row[0],))}}
            for row in SETTINGS}


class BudgetSuiteTests(unittest.TestCase):
    def test_unauthorized_physical_gpus_rejected(self):
        for gpu in range(4):
            with self.assertRaises(ValueError):
                launch(Path("."), SETTINGS[0], 0, gpu, "screen")

    def test_udd5_has_four_screen_shards_not_duplicate_full_runs(self):
        self.assertEqual(counts("udd5", "screen"), 4)
        self.assertEqual(counts("vdd", "screen"), 1)

    def test_complete_improvement_passes(self):
        self.assertTrue(promotion_gate(outcomes(), {})["passed"])

    def test_missing_dataset_or_execution_failure_cannot_pass(self):
        rows = outcomes(); rows.pop("vdd")
        self.assertFalse(promotion_gate(rows, {})["passed"])
        self.assertFalse(promotion_gate(outcomes(), {"worker": "failed"})["passed"])

    def test_doubly_stochastic_control_superiority_blocks_promotion(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["UniformDonorBudget"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_fixed_fusion_superiority_blocks_promotion(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["MeanLogit_DonorBudget"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_loveda_p_loss_not_hidden_by_domain_mean(self):
        rows = outcomes(); rows["loveda"]["metrics"]["P"][PRIMARY] = metric(70)
        self.assertFalse(promotion_gate(rows, {})["passed"])


if __name__ == "__main__":
    unittest.main()
