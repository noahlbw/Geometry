import numpy as np
import torch

from dinotool.oem import OEM_CLASSES
from dinotool.ov_train import (
    _class_balanced_weights,
    _ignored_feature_preservation_loss,
    _open_vocabulary_preservation_loss,
    _segmentation_loss,
)


def test_class_balancing_upweights_the_rare_source_class() -> None:
    counts = np.array([1000, 100, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)
    weights = _class_balanced_weights(counts, power=0.5, maximum=4.0)

    assert weights.shape == (len(OEM_CLASSES),)
    assert weights[1] > weights[0]
    assert torch.all(weights > 0)


def test_class_balancing_leaves_unobserved_subset_classes_neutral() -> None:
    counts = np.array([1000, 0, 100, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)
    weights = _class_balanced_weights(counts, power=0.5, maximum=4.0)

    assert weights[1].item() == 1.0
    assert weights[2] > weights[0]


def test_open_vocabulary_preservation_is_zero_for_identity_and_positive_after_shift() -> None:
    base = torch.nn.functional.normalize(torch.randn(2, 4, 3, 3), dim=1)
    text = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    identity = _open_vocabulary_preservation_loss(base, base, text)
    shifted = _open_vocabulary_preservation_loss(base, torch.roll(base, shifts=1, dims=1), text)

    assert identity.item() == 0.0
    assert shifted.item() > 0.0


def test_ignored_feature_preservation_only_uses_ignored_regions() -> None:
    base = torch.nn.functional.normalize(torch.randn(1, 4, 2, 2), dim=1)
    shifted = torch.roll(base, shifts=1, dims=1)
    labeled = torch.zeros((1, 32, 32), dtype=torch.long)
    ignored = labeled.clone()
    ignored[:, :16, :16] = 255

    assert _ignored_feature_preservation_loss(base, shifted, labeled).item() == 0.0
    assert _ignored_feature_preservation_loss(base, shifted, ignored).item() > 0.0


def test_segmentation_loss_accepts_balanced_weights_and_label_smoothing() -> None:
    logits = torch.randn(2, len(OEM_CLASSES), 4, 4, requires_grad=True)
    target = torch.zeros((2, 4, 4), dtype=torch.long)
    weights = torch.ones(len(OEM_CLASSES))

    loss = _segmentation_loss(logits, target, dice_weight=0.5, class_weights=weights, label_smoothing=0.02)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
