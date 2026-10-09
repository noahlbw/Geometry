import unittest

import numpy as np

from dinotool.context_sense_evaluation import confusion, frozen_source_input


class ContextSenseEvaluationTest(unittest.TestCase):
    def test_source_preset_does_not_fabricate_image_selection_or_copy_metadata(self):
        names, groups = ['background', 'mouse'], [['background'], ['mouse', 'computer mouse']]
        candidate = dict(dataset='context60', class_names=names,
            aliases_by_class=[['background'], ['computer mouse', 'mouse']],
            source_identity=dict(dataset='context60', classes=names, groups=groups,
                                 source_type='metadata only'))
        row = frozen_source_input(candidate, dict(classes=[dict(name=n) for n in names]),
                                  dict(checkpoints={'frozen': 1}, upstream_commit='pinned'), 'source-v1')
        self.assertEqual(row['identity']['groups'], groups)
        self.assertNotIn('source_type', row['identity'])
        self.assertEqual(row['selected_indices'], [0, 1, 2])
        self.assertEqual(row['chosen']['prob_thd'], .1)
        self.assertEqual(row['processed_images'], 0)
        self.assertFalse(row['source_image_selection_exists'])
        self.assertFalse(row['target_masks_loaded'])

    def test_source_preset_rejects_changed_class_order(self):
        candidate = dict(dataset='context59', class_names=['mouse'],
            source_identity=dict(dataset='context59', classes=['mouse'], groups=[['mouse']]))
        with self.assertRaises(ValueError):
            frozen_source_input(candidate, dict(classes=[dict(name='dog')]),
                                dict(checkpoints={}, upstream_commit='pinned'), 'source-v1')

    def test_foreground_rejects_count_as_false_negatives(self):
        cm, ignored = confusion(np.array([[0, 1, -1]]), np.array([[0, 255, 255]]), 2, True)
        np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 0, 1]])
        self.assertEqual(ignored, 1)
        self.assertEqual(cm.sum(), 2)

    def test_background_is_a_scored_row_not_an_ignore(self):
        cm, ignored = confusion(np.array([[0, 1]]), np.array([[0, 0]]), 2)
        np.testing.assert_array_equal(cm, [[1, 0], [1, 0]])
        self.assertEqual(ignored, 0)

    def test_unsupported_rejection_or_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            confusion(np.array([[0]]), np.array([[255]]), 2)
        with self.assertRaises(ValueError):
            confusion(np.array([[0]]), np.array([0]), 2)


if __name__ == '__main__':
    unittest.main()
