import re
import unittest

import torch

from dinotool.grounded_local_observer import prompt_chunks, fractional_box_support, box_observation


class OffsetTokenizer:
    def __call__(self, text, **kwargs):
        spans = [match.span() for match in re.finditer(r"[a-z]+|\.", text)]
        output = {"input_ids": [101, *range(len(spans)), 102]}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = [(0, 0), *spans, (0, 0)]
        return output


class GroundedObserverTests(unittest.TestCase):
    def test_chunks_preserve_every_alias_and_parent(self):
        chunks = prompt_chunks(OffsetTokenizer(), ["house", "large house", "road", "busy road"], [0, 0, 1, 1], 7)
        self.assertEqual([c["indices"] for c in chunks], [[0, 2], [1], [3]])
        self.assertTrue(all(c["token_masks"].any(1).all() for c in chunks))

    def test_alias_token_masks_exclude_separator_and_special_tokens(self):
        row = prompt_chunks(OffsetTokenizer(), ["large house"], [0])[0]
        self.assertEqual(row["token_masks"].tolist(), [[False, True, True, False, False]])

    def test_no_silent_alias_truncation(self):
        with self.assertRaises(ValueError):
            prompt_chunks(OffsetTokenizer(), ["a very large house"], [0], 4)

    def test_exact_partial_patch_area(self):
        box = torch.tensor([[0., 0., .25, .5]])
        torch.testing.assert_close(fractional_box_support(box, 2), torch.tensor([[.5, 0., 0., 0.]]))

    def test_no_boxes_provides_no_absence_evidence(self):
        value = box_observation(torch.empty(0, 4), torch.empty(0, 1), [0], [0], 2, 2)
        self.assertEqual(value.tolist(), [[0., 0.]]*4)

    def test_box_alias_parent_mapping_and_fixed_confidence(self):
        value = box_observation(torch.tensor([[0., 0., .5, .5]]), torch.tensor([[.2, .8]]),
                                [2, 0], [1, 1, 0], 2, 2)
        torch.testing.assert_close(value, torch.tensor([[0., .8], [0., 0.], [0., 0.], [0., 0.]]))

    def test_inverted_boxes_fail(self):
        with self.assertRaises(ValueError):
            fractional_box_support(torch.tensor([[.6, 0., .2, .5]]))


if __name__ == "__main__":
    unittest.main()
