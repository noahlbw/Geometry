from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))

from build_context_sense_vocab import context_candidate


class ContextSenseVocabularyTest(unittest.TestCase):
    def test_transfer_repairs_do_not_change_category_order_or_invent_selection(self):
        names = ('background', 'mouse', 'pottedplant', 'tvmonitor')
        official = (('background',), ('computer mouse',), ('potted plant',), ('television monitor',))
        candidate = context_candidate(names, official, 'context60')
        self.assertEqual(candidate['class_names'], list(names))
        self.assertEqual([g[0] for g in candidate['aliases_by_class']],
                         ['background', 'computer mouse', 'potted plant', 'television monitor'])
        self.assertIsNone(candidate['source_choice'])
        self.assertFalse(candidate['source_image_selection_exists'])
        self.assertFalse(candidate['target_masks_loaded'])
        self.assertTrue(all(n < 20 for n in candidate['alias_counts']))


if __name__ == '__main__':
    unittest.main()
