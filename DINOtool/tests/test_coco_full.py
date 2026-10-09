"""The repair must recover ignored concepts without shifting label indices."""
from pathlib import Path
import sys

import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinotool.coco_full import ACTIVE_IDS, UNUSED_THING_IDS, remap_labels


class CocoFullTests(unittest.TestCase):
    def test_person_sky_walls_and_ignore_are_distinct(self):
        raw = torch.tensor([[0, 1, 156, 170, 181, 255]])
        actual = remap_labels(raw)
        expected = torch.tensor([[0, 1, 145, 159, 170, 255]])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[0, 0].item(), 0)
        self.assertEqual(len(ACTIVE_IDS), 171)


    def test_unexpected_labels_fail_instead_of_silent_ignore(self):
        for raw_id in [UNUSED_THING_IDS[0], 182, 254, -1, 256]:
            with self.subTest(raw_id=raw_id), self.assertRaises(ValueError):
                remap_labels(torch.tensor([[raw_id]]))


if __name__ == "__main__":
    unittest.main()
