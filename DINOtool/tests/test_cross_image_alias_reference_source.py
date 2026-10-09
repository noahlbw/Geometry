from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval_cross_image_alias_reference import panel_indices, select_calibration


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    key: str


class CalibrationSourceTests(unittest.TestCase):
    def samples(self, count):
        return [Sample(Path('/RGB_ONLY/val/'+str(i)+'.png'), Path('/NEVER_READ/'+str(i)+'.png'), str(i)) for i in range(count)]

    def test64_unique_sources_exclude_pilot_and_timing(self):
        samples = self.samples(80)
        selected, row = select_calibration(samples, SimpleNamespace(dataset='vdd'))
        self.assertEqual(len(set(s.key for s in selected)), 64)
        forbidden = set(panel_indices(80, 'vdd')+[0, 40, 79])
        self.assertFalse(forbidden & {int(s.key) for s in selected})
        self.assertEqual(row['origin'], 'held-out evaluation RGB; transductive')

    def test_insufficient_disjoint_rgb_fails_without_mask_reads(self):
        with self.assertRaises(RuntimeError):
            select_calibration(self.samples(70), SimpleNamespace(dataset='vdd'))

    def test_udd5_uses_train_src_and_never_gt(self):
        images = [Path('/RGB_ONLY/UDD5/extracted/UDD/UDD5/train/src/'+str(i)+'.png') for i in range(70)]
        with patch.object(Path, 'iterdir', return_value=iter(images)) as enumerate_rgb, \
             patch.object(Path, 'is_file', return_value=True):
            selected, row = select_calibration(self.samples(40), SimpleNamespace(dataset='udd5', data_root='/RGB_ONLY/UDD5/extracted/UDD/UDD5'))
        enumerate_rgb.assert_called_once()
        self.assertEqual(len(selected), 64)
        self.assertTrue(all(s.mask_path == Path('/MASKS_FORBIDDEN') for s in selected))
        self.assertTrue(all('/train/src/' in str(s.image_path) for s in selected))
        self.assertTrue(row['calibration_evaluation_disjoint'])

    def test_source_selection_deterministic(self):
        args = SimpleNamespace(dataset='potsdam')
        first, first_meta = select_calibration(self.samples(504), args)
        second, second_meta = select_calibration(self.samples(504), args)
        self.assertEqual(first, second)
        self.assertEqual(first_meta, second_meta)


if __name__ == '__main__':
    unittest.main()
