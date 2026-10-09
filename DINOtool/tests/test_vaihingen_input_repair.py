import importlib.util
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1]/"scripts/repair_vaihingen_inputs.py"
spec = importlib.util.spec_from_file_location("vaihingen_repair", path)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class InputRepairTests(unittest.TestCase):
    def test_offsets_match_historical_overlap(self):
        self.assertEqual(repair.offsets(2557), [0, 1000, 1557])
        self.assertEqual(repair.offsets(1887), [0, 887])

    def test_three_bands_are_preserved(self):
        source = np.empty((1100, 1200, 3), np.uint8)
        source[:] = [32, 71, 192]
        crop = repair.source_crop(source, "top_mosaic_09cm_area10_y01_x01.tif")
        np.testing.assert_array_equal(crop, source[100:1100, 200:1200])
        self.assertEqual(crop.shape, (1000, 1000, 3))

    def test_malformed_source_is_rejected(self):
        with self.assertRaises(ValueError):
            repair.source_crop(np.zeros((1000, 1000), np.uint8), "area10_y00_x00.tif")

    def test_palette_and_ignore_are_fixed(self):
        colors = np.array(list(repair.PALETTE), np.uint8)[None]
        np.testing.assert_array_equal(repair.remap_label(colors),
                                      np.array([list(repair.PALETTE.values())], np.uint8))

    def test_unknown_colors_are_not_default_background(self):
        with self.assertRaises(ValueError):
            repair.remap_label(np.array([[[1, 2, 3]]], np.uint8))


if __name__ == "__main__":
    unittest.main()
