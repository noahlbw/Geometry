import unittest

import numpy as np

from report_matched_readout_publication import class_errors, group_key, miou


class MatchedPublicationReportTests(unittest.TestCase):
    def test_parent_grouping(self):
        self.assertEqual(group_key("potsdam", "top_potsdam_2_13_RGB_y00_x01.tif"), "top_potsdam_2_13_RGB")
        self.assertEqual(group_key("vaihingen", "top_mosaic_09cm_area10_y01_x00.tif"), "top_mosaic_09cm_area10")
        self.assertEqual(group_key("landcoverai", "M-33-20-D-c-4-2_103"), "M-33-20-D-c-4-2")
        self.assertEqual(group_key("flair1", "D012_2019/Z10_UU/img/IMG_062613.tif"), "D012_2019")
        self.assertEqual(group_key("oem", "aachen/images/aachen_11.tif"), "aachen")
        self.assertEqual(group_key("udd5", "000061.JPG"), "000061.JPG")

    def test_confusion_orientation_and_absent_class(self):
        rows = class_errors([[8, 2, 0], [3, 7, 0], [0, 0, 0]], ["a", "b", "absent"])
        self.assertEqual((rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]), (8, 3, 2))
        self.assertAlmostEqual(rows[0]["precision_percent"], 800/11)
        self.assertEqual(rows[0]["recall_percent"], 80)
        self.assertIsNone(rows[2]["predicted_target_ratio"])
        self.assertIsNone(rows[2]["precision_percent"])
        self.assertAlmostEqual(float(miou(np.array([[8, 2, 0], [3, 7, 0], [0, 0, 0]]))), (8/13+7/12)*50)

    def test_batched_miou_uses_summed_counts(self):
        matrices = np.array([[[8, 2], [3, 7]], [[2, 8], [0, 10]]])
        np.testing.assert_allclose(miou(matrices), [(8/13+7/12)*50, (2/10+10/18)*50])
        self.assertNotAlmostEqual(float(miou(matrices.sum(0))), float(miou(matrices).mean()))


if __name__ == "__main__":
    unittest.main()
