"""Small checks for cached pair directions, alias identities and soft support."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_context_reliability_source import alias_groups, describe, support_audit, audit_groups
from dinotool.prompts import ClassSpec


class SourceAuditTest(unittest.TestCase):
    def test_actual_alias_identity(self):
        groups = [[f'c{c}']+[f'c{c} alias{i}' for i in range(19)] for c in range(2)]
        specs = [ClassSpec(row[0], tuple(row)) for row in groups]
        members, canonical = alias_groups(specs, sum(groups, []), [20, 20])
        np.testing.assert_array_equal(canonical, [0, 20])
        self.assertEqual(members.shape, (2, 20))
        with self.assertRaises(ValueError):
            alias_groups(specs, sum(groups[::-1], []), [20, 20])

    def test_empty_summary(self):
        self.assertIsNone(describe([])['mean'])
        self.assertEqual(describe([-1, 0, 2])['negative_fraction'], 1/3)

    def test_soft_support_uses_pixel_fractions(self):
        target = np.zeros((512, 512), np.int64)
        target[:16, :8] = 1
        target[16:32, :16] = -1
        masks = np.zeros((16, 32, 32))
        masks[:, 0, 0] = 1
        masks[:, 1, 0] = 1
        support = support_audit(target, masks, np.zeros(1024, int), np.ones(1024, bool), 2)
        np.testing.assert_array_equal(support['class_mass'][0], [.5, .5])
        self.assertEqual(support['labeled_fraction'][0], .5)
        self.assertEqual(support['effective_patches'][0], 2)

    def test_pair_truth_groups_and_direction(self):
        n = 3
        scalar = np.array([[2., 0.], [2., 0.], [0., 2.]])
        fields = {'baseline': scalar, 'RivalPreserving_Exact__scores': scalar[:, ::-1],
            'valid': np.ones(n, bool), 'assignments': np.zeros(n, int), 'supported_units': np.ones(1, bool),
            'capacity': np.ones((n, 2))}
        for name in ('capacity_weighted_risk', 'directed', 'requested', 'alias_full_margin',
                     'alias_kept_margin0', 'alias_kept_margin1', 'alias_removed_margin0', 'alias_removed_margin1'):
            fields[name] = np.zeros((n, 2, 2))
        for name in ('potential', 'receiving', 'local', 'broad', 'canonical_full', 'canonical_kept0',
                     'canonical_kept1', 'canonical_removed0', 'canonical_removed1'):
            fields[name] = scalar
        for name in ('ContrastReversal_Exact', 'RivalShuffledSupport_Exact'):
            fields[name+'__scores'] = scalar
        support = {'class_mass': np.ones((n, 2))*.5, 'cell_class_mass': np.ones((n, 2))*.5,
            'dominant_fraction': np.ones(n)*.5, 'effective_patches': np.ones(n), 'labeled_fraction': np.ones(n)}
        store = {}
        audit_groups(fields, support, np.array([0, 1, 0]), store)
        self.assertEqual(store['0/1/true_positive']['candidate_correct'][0].tolist(), [0.])
        self.assertEqual(store['0/1/false_positive']['candidate_correct'][0].tolist(), [1.])
        self.assertEqual(store['0/1/miss']['candidate_correct'][0].tolist(), [1.])
        self.assertEqual(store['0/1/false_positive']['receiving_margin_change'][0].tolist(), [2.])


if __name__ == '__main__':
    unittest.main()
