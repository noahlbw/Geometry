import unittest

from run_geometry_row_composition_screen import panel


class PanelTests(unittest.TestCase):
    def test_excludes_old_keys_and_is_deterministic(self):
        keys = [f"scene{i}/images/image{j}.tif" for i in range(10) for j in range(8)]
        old = keys[:8]
        a = panel("oem", keys, old)
        self.assertEqual(a, panel("oem", keys, old))
        selected = a["signature"]["samples"]
        self.assertEqual(len(selected), 32)
        self.assertFalse(set(selected).intersection(old))
        self.assertEqual(selected, [key for key in keys if key in selected])
        self.assertEqual(a["source_groups"], 9)

    def test_udd5_explicitly_reuses_full40(self):
        keys = [str(i) for i in range(40)]
        a = panel("udd5", keys, keys)
        self.assertEqual(a["signature"]["samples"], keys)
        self.assertEqual(a["old_panel_exact_key_overlap"], 40)

    def test_invalid_or_insufficient_inputs_fail(self):
        for keys, old in ((["a"]*40, []), ([str(i) for i in range(40)], ["unknown"]),
                          ([str(i) for i in range(31)], [])):
            with self.assertRaises(ValueError):
                panel("vdd", keys, old)


if __name__ == "__main__":
    unittest.main()
