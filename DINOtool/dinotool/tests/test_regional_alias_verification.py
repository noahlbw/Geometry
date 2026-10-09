from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from dinotool.gear_ov import class_scores
from dinotool.regional_alias_verification import (
    RegionalAliasVerificationReadout, fixed_denominator_scores,
)


class RegionalAliasVerificationTests(unittest.TestCase):
    def test_rejecting_weak_alias_cannot_raise_parent_score(self):
        parents = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([[2.0, -1.0, 0.5, 0.0]])
        reference = torch.zeros_like(scores)
        original = class_scores(scores, parents, 2)
        retention = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        corrected = fixed_denominator_scores(scores, reference, retention, parents, 2)
        self.assertTrue(torch.allclose(corrected, original))
        retention[0, 0] = 0
        corrected = fixed_denominator_scores(scores, reference, retention, parents, 2)
        self.assertLess(float(corrected[0, 0]), float(original[0, 0]))
        self.assertEqual(float(corrected[0, 1]), float(original[0, 1]))

    def test_vectorized_family_reference_matches_direct_logmeanexp(self):
        readout = object.__new__(RegionalAliasVerificationReadout)
        readout.parents = torch.tensor([0, 0, 0, 1, 1, 1])
        readout.aliases = 6
        readout.classes = 2
        readout.families = [[0, 3], [1, 2], [4], [5]]
        family_index = torch.tensor([0, 1, 1, 0, 2, 3])
        same_class = readout.parents[:, None] == readout.parents[None]
        same_family = family_index[:, None] == family_index[None]
        readout.excluded_alias_matrix = (same_class & same_family).float()
        readout.excluded_alias_count = readout.excluded_alias_matrix.sum(0)
        readout.class_count = torch.bincount(readout.parents, minlength=2)
        readout.parent_matrix = F.one_hot(readout.parents, 2).float()
        flat_group_class = family_index * 2 + readout.parents
        readout.group_class_matrix = F.one_hot(flat_group_class, 8).float()
        readout.group_class_count = readout.group_class_matrix.sum(0).reshape(4, 2)
        values = torch.tensor([[0.7, -0.3, 0.1, 0.6, -0.2, 0.2],
                               [-1.5, 1.0, 0.4, -0.1, 0.5, -0.4]])
        teacher, reference = readout._leave_family_out(values)
        for row in range(2):
            for family, members in enumerate(readout.families):
                for cls in range(2):
                    remaining = [idx for idx in range(6)
                                 if idx not in members and int(readout.parents[idx]) == cls]
                    expected = torch.logsumexp(values[row, remaining], 0) - math.log(len(remaining))
                    self.assertAlmostEqual(float(teacher[row, family, cls]),
                                           float(expected), places=5)
                    for alias in members:
                        if int(readout.parents[alias]) == cls:
                            self.assertAlmostEqual(float(reference[row, alias]),
                                                   float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
