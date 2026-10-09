import unittest
import numpy as np

from dinotool.cross_support_alias_diagnostic import (
    CONFIG, cross_fit, effective_count, fit_predict, fold_indices, semantic_audit)


class CrossSupportDiagnosticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.base = rng.normal(size=(1024, 4))
        self.alias = rng.normal(size=1024)
        self.valid = np.ones(1024, dtype=bool)
        self.weight = np.ones(1024)

    def test_folds_disjoint_with_guard(self):
        left, right = fold_indices(self.valid)
        self.assertEqual(len(left), 448)
        self.assertEqual(len(right), 448)
        self.assertEqual(len(np.intersect1d(left, right)), 0)
        self.assertTrue((left%32 < 14).all())
        self.assertTrue((right%32 >= 18).all())

    def test_genuine_increment_survives_cross_fit_and_position_null(self):
        target = self.base[:, 0]+2*self.alias
        rows = cross_fit(self.base, self.alias, target, self.weight, self.valid, 10)
        for row in rows:
            self.assertEqual(row['status'], 'tested')
            self.assertGreater(row['increment_fraction'], .95)
            self.assertGreater(row['increment'], max(row['null_increments']))

    def test_redundant_alias_cannot_split_regularization_penalty(self):
        alias = 3*self.base[:, 1]-self.base[:, 3]+4
        target = self.base[:, 0]+self.base[:, 1]
        for row in cross_fit(self.base, alias, target, self.weight, self.valid, 11):
            self.assertAlmostEqual(row['increment'], 0., places=12)

    def test_class_only_target_has_no_large_alias_gain(self):
        target = self.base[:, 0]
        rows = cross_fit(self.base, self.alias, target, self.weight, self.valid, 12)
        self.assertLess(max(abs(row['increment']) for row in rows), .001)

    def test_heldout_reference_does_not_change_training_fit(self):
        train, test = fold_indices(self.valid)
        y = self.base[:, 0]+self.alias
        before = fit_predict(self.base[train], y[train], self.weight[train], self.base[test])[0]
        y[test] += 1000
        after = fit_predict(self.base[train], y[train], self.weight[train], self.base[test])[0]
        np.testing.assert_array_equal(before, after)

    def test_effective_support_rejects_concentration_and_padding(self):
        support = np.zeros(1024)
        support[[0, 31]] = 1
        rows = cross_fit(self.base, self.alias, self.alias, support, self.valid, 13)
        self.assertTrue(all(row['status'] == 'insufficient_support' for row in rows))
        self.assertAlmostEqual(effective_count(support), 2)
        self.valid[:32] = False
        self.assertTrue(all(not np.isin([0, 31], f).any() for f in fold_indices(self.valid)))

    def test_semantic_audit_cannot_change_source_increment(self):
        target = self.base[:, 0]+self.alias
        rows = cross_fit(self.base, self.alias, target, self.weight, self.valid, 14)
        pending = []
        original = []
        for row in rows:
            predictions = {k: row.pop(k) for k in ('position', 'weight', 'before', 'after', 'nulls')}
            row.update(tile=0, owner=0, rival=1)
            original.append(row['increment'])
            pending.append((row, predictions))
        labels = (target <= 0).astype(int)
        semantic_audit(pending, [labels])
        self.assertEqual(original, [row['increment'] for row, _ in pending])
        self.assertTrue(all(row['audit_status'] == 'tested' for row, _ in pending))

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            fit_predict(self.base[:5], np.full(5, np.nan), np.ones(5), self.base[:2])


if __name__ == '__main__':
    unittest.main()
