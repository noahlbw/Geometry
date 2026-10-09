import copy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_semantic_readout_interactions as module


def payload(matrices):
    return {
        "status": "complete", "coverage_verified": True,
        "processed_images": 2, "total_images": 2,
        "metrics": {arm: {"per_class": [{"name": "a"}, {"name": "b"}],
                          "confusion_matrix": matrix}
                    for arm, matrix in zip(module.ARMS, matrices)},
    }


class ReadoutInteractionTest(unittest.TestCase):
    def test_unchanged_has_no_interaction(self):
        result = module.audit(payload([[[8, 2], [2, 8]]] * 4))
        self.assertEqual(result["miou_interaction_pp"], 0)
        self.assertTrue(all(row["tp_interaction"] == row["fp_interaction"] == 0
                            for row in result["per_class_interactions"]))

    def test_head_independent_text_change_has_no_interaction(self):
        first, second = [[8, 2], [2, 8]], [[9, 1], [1, 9]]
        result = module.audit(payload([first, first, second, second]))
        self.assertEqual(result["miou_interaction_pp"], 0)
        self.assertGreater(result["contrasts"]["text_geometry"]["delta_miou_pp"], 0)

    def test_differential_text_change(self):
        baseline, improved, degraded = [[8, 2], [2, 8]], [[9, 1], [1, 9]], [[7, 3], [3, 7]]
        result = module.audit(payload([baseline, baseline, degraded, improved]))
        self.assertGreater(result["miou_interaction_pp"], 0)
        self.assertEqual(result["per_class_interactions"][0]["tp_interaction"], 2)
        self.assertEqual(result["per_class_interactions"][0]["fp_interaction"], -2)

    def test_changed_ground_truth_rejected(self):
        with self.assertRaises(ValueError):
            module.audit(payload([[[8, 2], [2, 8]]] * 3 + [[[8, 3], [2, 7]]]))

    def test_changed_class_order_rejected(self):
        data = payload([[[8, 2], [2, 8]]] * 4)
        data["metrics"]["VIP_RS"]["per_class"].reverse()
        with self.assertRaises(ValueError):
            module.audit(data)

    def test_incomplete_or_unverified_rejected(self):
        original = payload([[[8, 2], [2, 8]]] * 4)
        for key, value in (("status", "running"), ("coverage_verified", False),
                           ("processed_images", 1)):
            with self.subTest(key=key):
                data = copy.deepcopy(original)
                data[key] = value
                with self.assertRaises(ValueError):
                    module.audit(data)

    def test_zero_prediction_has_no_precision(self):
        result = module.audit(payload([[[10, 0], [10, 0]]] * 4))
        row = result["contrasts"]["text_geometry"]["per_class"][1]
        self.assertIsNone(row["candidate_rates"]["precision_percent"])
        self.assertEqual(row["candidate_rates"]["recall_percent"], 0)


if __name__ == "__main__":
    unittest.main()
