import copy
import unittest

from dinotool.natural_alias_calibration import validate_selections


class NaturalAliasCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(status='complete', dataset='voc20', target_masks_loaded=False,
            target_label_tuning=False, identity={'classes': ['a', 'b'], 'groups': [['a', 'aa'], ['b', 'bb']]},
            chosen={'tau': 5., 'tem': .3, 'prob_thd': 0.}, image_keys=['one', 'two'],
            selected_counts=[2, 2], selected_indices=[0, 1, 2, 3], pool_counts=[2, 2])
        self.count = dict(status='complete', dataset='voc20', target_masks_loaded=False,
            target_label_tuning=False, smoke_only=False, weights_frozen=True, head_weights_unchanged=True,
            source_selection_sha256='source', source_identity=copy.deepcopy(self.source['identity']),
            source_choice=copy.deepcopy(self.source['chosen']), selected_counts=[2, 2],
            image_keys=['one', 'two'], processed_images=2, total_images=2,
            count_calibration={'strength': .5})

    def validate(self, curated=None):
        return validate_selections(self.source, self.count, curated, 'source')

    def test_frozen_unlabeled_selection_is_accepted(self):
        self.assertEqual(self.validate(), .5)

    def test_smoke_cannot_select_deployed_strength(self):
        self.count['smoke_only'] = True
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_source_hash_or_mask_use_is_rejected(self):
        self.count['source_selection_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            self.validate()
        self.count['source_selection_sha256'] = 'source'
        self.source['target_masks_loaded'] = True
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_calibration_image_coverage_is_rejected(self):
        self.count['image_keys'] = ['two', 'one']
        with self.assertRaises(ValueError):
            self.validate()

    def test_text_only_nonanchor_removal_is_accepted(self):
        curated = dict(self.source, selected_indices=[0, 2, 3], selected_counts=[1, 2])
        self.assertEqual(self.validate(curated), .5)

    def test_anchor_removal_and_wrong_count_are_rejected(self):
        for indices, counts in (([1, 2, 3], [1, 2]), ([0, 2, 3], [2, 2])):
            with self.assertRaises(ValueError):
                self.validate(dict(self.source, selected_indices=indices, selected_counts=counts))

    def test_curation_cannot_change_threshold(self):
        curated = copy.deepcopy(self.source)
        curated['chosen']['prob_thd'] = .2
        with self.assertRaises(ValueError):
            self.validate(curated)


if __name__ == '__main__':
    unittest.main()
