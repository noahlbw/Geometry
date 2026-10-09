import unittest
from pathlib import Path

from run_geometry_unit_suite import METHODS, PRIMARY, SETTINGS, counts, launch, promotion_gate


def metric(correct):
    return {"confusion_matrix": [[correct, 100-correct], [100-correct, correct]]}


def outcomes():
    return {row[0]: {"metrics": {key: {method: metric(85 if method == PRIMARY else 80) for method in METHODS}
                                for key in (("P", "D") if row[0] == "loveda" else (row[0],))}}
            for row in SETTINGS}


class UnitSuiteTests(unittest.TestCase):
    def test_authorized_gpu_scope(self):
        for gpu in range(4):
            with self.assertRaises(ValueError):
                launch(Path("."), SETTINGS[0], 0, gpu, "screen")

    def test_full_udd5_partition(self):
        self.assertEqual(counts("udd5", "screen"), 4)

    def test_complete_success(self):
        self.assertTrue(promotion_gate(outcomes(), {})["passed"])

    def test_missing_data_and_failures_rejected(self):
        rows = outcomes()
        rows.pop("udd5")
        self.assertFalse(promotion_gate(rows, {})["passed"])
        self.assertFalse(promotion_gate(outcomes(), {"job": "failure"})["passed"])

    def test_other_domain_decline_cannot_be_hidden_by_mean(self):
        rows = outcomes()
        rows["oem"]["metrics"]["oem"][PRIMARY] = metric(79)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_potsdam_requires_more_than_generic_compression(self):
        rows = outcomes()
        rows["potsdam"]["metrics"]["potsdam"]["Spatial_UnitDetail"] = metric(90)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_same_information_fusion_must_not_be_better(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["MeanLogit_Unit"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])


if __name__ == "__main__":
    unittest.main()
