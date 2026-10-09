import unittest
from pathlib import Path

from run_geometry_semantic_demixing_suite import METHODS, PRIMARY, SETTINGS, counts, launch, promotion_gate


def metric(correct):
    return {"confusion_matrix": [[correct, 100-correct], [100-correct, correct]]}


def outcomes():
    return {row[0]: {"metrics": {key: {method: metric(85 if method == PRIMARY else 80) for method in METHODS}
        for key in (("P", "D") if row[0] == "loveda" else (row[0],))}} for row in SETTINGS}


class DemixingSuiteTests(unittest.TestCase):
    def test_only_authorized_gpus(self):
        for gpu in (0, 1, 2, 3, 8):
            with self.assertRaises(ValueError):
                launch(Path("."), SETTINGS[0], 0, gpu, "screen")

    def test_full_udd5_screen_partition(self):
        self.assertEqual(counts("udd5", "screen"), 4)

    def test_all_comparators_and_domains_pass(self):
        self.assertTrue(promotion_gate(outcomes(), {})["passed"])

    def test_missing_data_or_failure_rejects(self):
        rows = outcomes()
        del rows["udd5"]
        self.assertFalse(promotion_gate(rows, {})["passed"])
        self.assertFalse(promotion_gate(outcomes(), {"shard": "failed"})["passed"])

    def test_other_domain_loss_cannot_be_hidden(self):
        rows = outcomes()
        rows["oem"]["metrics"]["oem"][PRIMARY] = metric(79)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_loveda_p_loss_is_not_hidden_by_d(self):
        rows = outcomes()
        rows["loveda"]["metrics"]["P"][PRIMARY] = metric(79)
        self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_text_only_or_fusion_is_not_enough(self):
        for method in ("TextOnlyDemix", "MeanLogit_Demix"):
            rows = outcomes()
            rows["potsdam"]["metrics"]["potsdam"][method] = metric(86)
            self.assertFalse(promotion_gate(rows, {})["passed"])

    def test_shuffled_relation_must_not_win(self):
        rows = outcomes()
        for row in rows.values():
            for group in row["metrics"].values():
                group["ShuffledGeometryDemix"] = metric(86)
        self.assertFalse(promotion_gate(rows, {})["passed"])


if __name__ == "__main__":
    unittest.main()
