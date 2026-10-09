"""Mechanism identities, not evidence of segmentation improvement."""
import unittest

import numpy as np


def lme(scores, temperature=0.07):
    maximum = scores.max(axis=-1, keepdims=True)
    return maximum[..., 0] + temperature * np.log(
        np.exp((scores - maximum) / temperature).mean(axis=-1))


class ReadoutAlgebraTest(unittest.TestCase):
    def test_common_alias_shift_with_different_class_counts(self):
        rng = np.random.default_rng(24)
        scores = [rng.normal(size=(8, count)) for count in (1, 2, 20)]
        common = rng.normal(size=(8, 1))
        before = np.stack([lme(arm) for arm in scores], axis=-1)
        after = np.stack([lme(arm - common) for arm in scores], axis=-1)
        np.testing.assert_allclose(after, before - common, atol=1e-14)
        np.testing.assert_array_equal(before.argmax(axis=-1), after.argmax(axis=-1))

    def test_surgery_centering_preserves_fixed_weight_competition(self):
        rng = np.random.default_rng(25)
        image, text = rng.normal(size=(5, 7)), rng.normal(size=(60, 7))
        weights = rng.random(60)
        products = image[:, None, :] * text[None, :, :] * weights[None, :, None]
        raw = products.sum(axis=-1)
        centered = (products - products.mean(axis=1, keepdims=True)).sum(axis=-1)
        np.testing.assert_allclose(centered, raw - raw.mean(axis=-1, keepdims=True), atol=1e-14)
        np.testing.assert_allclose(lme(centered.reshape(5, 3, 20)),
                                   lme(raw.reshape(5, 3, 20)) - raw.mean(axis=-1, keepdims=True),
                                   atol=1e-14)

    def test_alias_dependent_weights_can_change_argmax(self):
        scores = np.array([[0.8, 0.6]])
        weights = np.array([[0.4, 1.6]])
        self.assertEqual(scores.argmax(), 0)
        self.assertEqual((scores * weights).argmax(), 1)

    def test_shared_redundant_text_subtraction_is_common_score_shift(self):
        rng = np.random.default_rng(26)
        image, text = rng.normal(size=(8, 5)), rng.normal(size=(3, 5))
        redundant = rng.normal(size=(1, 5))
        np.testing.assert_allclose(image @ (text - redundant).T,
                                   image @ text.T - image @ redundant.T, atol=1e-14)

    def test_geometry_preserves_shared_value_mode(self):
        rng = np.random.default_rng(27)
        relation = rng.random((9, 9))
        relation /= relation.sum(axis=-1, keepdims=True)
        values = rng.normal(size=(9, 6))
        shared = values.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(relation @ values,
                                   shared + relation @ (values - shared), atol=1e-14)

    def test_spatial_normalization_is_not_classification_invariant(self):
        scores = np.array([[0.8, 0.6], [0.9, 0.0], [0.7, 1.0]])
        normalized = (scores - scores.min(axis=0)) / np.ptp(scores, axis=0)
        self.assertEqual(scores[0].argmax(), 0)
        self.assertEqual(normalized[0].argmax(), 1)


if __name__ == "__main__":
    unittest.main()
