import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/audit_semantic_correction_operating_point.py"
spec = importlib.util.spec_from_file_location("operating_point", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OperatingPointTest(unittest.TestCase):
    def test_false_positive_only_improvement(self):
        result = module.iou_decomposition([[8, 2], [5, 5]], [[8, 2], [1, 9]], ["a", "b"])[0]
        self.assertEqual(result["tp_contribution_pp"], 0)
        self.assertGreater(result["fp_contribution_pp"], 0)

    def test_tp_loss_can_coexist_with_iou_gain(self):
        result = module.iou_decomposition([[9, 1], [80, 20]], [[8, 2], [10, 90]], ["a", "b"])[0]
        self.assertLess(result["delta_tp"], 0)
        self.assertGreater(result["delta_iou_pp"], 0)
        self.assertAlmostEqual(result["delta_iou_pp"], result["tp_contribution_pp"] + result["fp_contribution_pp"])

    def test_unchanged(self):
        matrix = [[9, 1], [2, 8]]
        result = module.iou_decomposition(matrix, matrix, ["a", "b"])
        self.assertTrue(all(row["delta_iou_pp"] == 0 for row in result))

    def test_changed_ground_truth_rejected(self):
        with self.assertRaises(ValueError):
            module.iou_decomposition([[1, 0], [0, 1]], [[2, 0], [0, 1]], ["a", "b"])

    def test_no_union_not_zero_iou(self):
        result = module.iou_decomposition([[1, 0], [0, 0]], [[1, 0], [0, 0]], ["a", "b"])
        self.assertIsNone(result[1]["delta_iou_pp"])

    def test_break_even(self):
        result = module.routing_summary(100, 200, 50, 50)
        self.assertEqual(result["net_correct_pixels"], 0)
        self.assertEqual(result["minimum_harmful_rejection_for_nonnegative_accuracy"], 0.75)

    def test_no_available_harm(self):
        result = module.routing_summary(10, 0, 2, 0)
        self.assertIsNone(result["harmful_rejection"])
        self.assertEqual(result["net_correct_pixels"], 2)

    def test_impossible_retention_rejected(self):
        with self.assertRaises(ValueError):
            module.routing_summary(10, 10, 11, 2)


if __name__ == "__main__":
    unittest.main()
