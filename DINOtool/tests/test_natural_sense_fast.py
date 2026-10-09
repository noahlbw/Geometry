from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dinotool.natural_variable_alias_reader import retained_variable_scores
from eval_excess_alias_rejection import prepare_wide
from eval_natural_sense_fast import prepare_wide_observations, predict_fast


def dependency_probe(*args, **kwargs):
    return prepare_wide, retained_variable_scores, args, kwargs, torch.is_inference_mode_enabled()


class NaturalSenseFastTest(unittest.TestCase):
    def test_observer_preserves_crops_and_visual_forward_count(self):
        torch.manual_seed(14)
        image = torch.rand(3, 317, 401)
        features = torch.linspace(-1., 1., 441*4).reshape(1, 441, 4)
        query = SimpleNamespace(features=torch.randn(5, 2, 4), parents=torch.tensor([0, 0, 1, 1, 1]),
                                class_names=('a', 'b'))
        bank = SimpleNamespace(features=torch.randn(5, 4))
        variants = {'text': ({'default': bank, 'frozen': bank}, {'default': query, 'frozen': query})}
        calls = []

        def crop_features(rgb):
            calls.append(tuple(rgb.shape))
            return features+rgb.mean()

        vip = SimpleNamespace(device=torch.device('cpu'), crop_patch_features=crop_features)
        with patch('torch.autocast', return_value=nullcontext()), patch(
                'eval_excess_alias_rejection.imagenet_geometry_logits',
                return_value=torch.zeros(2, 336, 336)) as unused:
            _, expected, _, count = prepare_wide(image, vip, variants)
            original_calls = len(calls)
            self.assertEqual(unused.call_count, original_calls*2)
            calls.clear()
            maps, actual, auxiliary, fast_count = prepare_wide_observations(image, vip, variants)
            self.assertEqual(unused.call_count, original_calls*2)
        self.assertEqual(maps, {})
        self.assertEqual(auxiliary, {})
        self.assertEqual(len(calls), original_calls)
        self.assertTrue(torch.equal(count, fast_count))
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertEqual(len(actual[key]), len(expected[key]))
            for old, fast in zip(expected[key], actual[key]):
                self.assertTrue(torch.equal(old.alias_logits, fast.alias_logits))
                self.assertTrue(torch.equal(old.salience, fast.salience))
                self.assertEqual((old.top, old.left, old.actual_height, old.actual_width),
                                 (fast.top, fast.left, fast.actual_height, fast.actual_width))

    def test_dependency_substitution_is_scoped_and_preserves_inference_mode(self):
        import eval_natural_sense_adaptation as reference
        from dinotool.natural_variable_alias_fast import retained_variable_scores_fast

        original = reference.predict
        with patch('eval_natural_sense_fast.reference_predict', torch.inference_mode()(dependency_probe)):
            wide, reader, args, kwargs, inference = predict_fast('image', calibrated=None)
        self.assertIs(wide, prepare_wide_observations)
        self.assertIs(reader, retained_variable_scores_fast)
        self.assertEqual(args, ('image',))
        self.assertEqual(kwargs, {'calibrated': None})
        self.assertTrue(inference)
        self.assertIs(reference.predict, original)
        self.assertIs(dependency_probe.__globals__['prepare_wide'], prepare_wide)


if __name__ == '__main__':
    unittest.main()
