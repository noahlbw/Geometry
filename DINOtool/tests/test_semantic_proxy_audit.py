import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/audit_semantic_proxy_reliability.py"
spec = importlib.util.spec_from_file_location("semantic_proxy_audit", path)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def entry(name, proxy, likelihood, flips):
    return {"class": name, "alias": str(proxy), "native_counterfactual_gain": proxy,
            "mean_logprob_gain": likelihood, "flip_balance": flips,
            "beneficial_flips": max(flips, 0), "harmful_flips": max(-flips, 0)}


class SemanticProxyAuditTest(unittest.TestCase):
    def test_auc_ties(self):
        self.assertEqual(audit_module.auc([1, 1], [True, False]), (0.5, 1))

    def test_no_comparable_auc(self):
        self.assertEqual(audit_module.auc([1, 2], [True, True]), (None, 0))

    def test_inverse_direction_and_different_targets(self):
        result = audit_module.audit({"status": "complete", "dataset": "fixture",
                                     "images": 1, "tiles": 1,
                                     "aliases": [entry("a", -1, 1, -1),
                                                 entry("a", 1, -1, 1)]})
        self.assertEqual(result["likelihood_retention"]["inverse_native_auc"], 1)
        self.assertEqual(result["correct_pixel_retention"]["inverse_native_auc"], 0)
        self.assertEqual(result["positive_likelihood_negative_flips_count"], 1)

    def test_neutral_flips_excluded(self):
        result = audit_module.audit({"status": "complete", "dataset": "fixture",
                                     "images": 1, "tiles": 1,
                                     "aliases": [entry("a", -1, 1, 0)]})
        self.assertEqual(result["correct_pixel_retention"]["neutral"], 1)
        self.assertIsNone(result["correct_pixel_retention"]["inverse_native_auc"])

    def test_per_class_not_cross_class_pairs(self):
        result = audit_module.audit({"status": "complete", "dataset": "fixture",
                                     "images": 1, "tiles": 1,
                                     "aliases": [entry("a", -1, 1, 1),
                                                 entry("a", 1, -1, -1),
                                                 entry("b", -10, 1, 1)]})
        self.assertEqual(result["correct_pixel_retention"]["within_class_comparable_pairs"], 1)

    def test_wrong_flip_convention_rejected(self):
        row = entry("a", -1, 1, 1)
        row["flip_balance"] = -1
        with self.assertRaises(ValueError):
            audit_module.audit({"status": "complete", "aliases": [row]})


if __name__ == "__main__":
    unittest.main()
