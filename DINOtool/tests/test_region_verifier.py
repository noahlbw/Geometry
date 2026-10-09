import numpy as np
import torch

import dinotool.region_verifier as region_verifier
from dinotool.region_verifier import DINORegionVerifier, RegionVerifierConfig


def _verification_inputs(
    *,
    classes: int,
    region_scores: tuple[float, ...],
    dino_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sam_scores = torch.zeros((classes, 4, 4), dtype=torch.float32)
    sam_scores[0].fill_(0.90)
    sam_scores[1:].fill_(0.02)
    sam_scores[:, 1:3, 1:3] = torch.tensor(region_scores, dtype=torch.float32).view(classes, 1, 1)
    patch_features = torch.zeros((1, classes, 2, 2), dtype=torch.float32)
    patch_features[:, dino_class].fill_(1.0)
    text_features = torch.eye(classes, dtype=torch.float32)
    global_anchor = text_features[dino_class].clone()
    return sam_scores, patch_features, text_features, global_anchor


def _verifier() -> DINORegionVerifier:
    return DINORegionVerifier(
        RegionVerifierConfig(
            minimum_region_area=4,
            sam_confident_score=0.80,
            sam_ambiguous_margin=0.12,
            sam_candidate_count=3,
            minimum_dino_margin=0.01,
            minimum_boundary_contrast=-0.01,
            global_top_k=4,
            use_per_image_prototypes=False,
        )
    )


def test_region_verifier_relabels_an_ambiguous_region_with_dino_support() -> None:
    inputs = _verification_inputs(
        classes=3,
        region_scores=(0.10, 0.55, 0.52),
        dino_class=2,
    )

    labels, diagnostics = _verifier().verify(*inputs)

    assert torch.equal(labels[1:3, 1:3], torch.full((2, 2), 2, dtype=torch.long))
    assert diagnostics.regions_relabelled == 1


def test_region_verifier_keeps_a_confident_sam_region() -> None:
    inputs = _verification_inputs(
        classes=3,
        region_scores=(0.10, 0.95, 0.55),
        dino_class=2,
    )

    labels, diagnostics = _verifier().verify(*inputs)

    assert torch.equal(labels[1:3, 1:3], torch.full((2, 2), 1, dtype=torch.long))
    assert diagnostics.regions_relabelled == 0


def test_region_verifier_rejects_dino_classes_outside_sam_candidates() -> None:
    inputs = _verification_inputs(
        classes=4,
        region_scores=(0.48, 0.50, 0.49, 0.10),
        dino_class=3,
    )

    labels, diagnostics = _verifier().verify(*inputs)

    assert torch.equal(labels[1:3, 1:3], torch.full((2, 2), 1, dtype=torch.long))
    assert diagnostics.regions_relabelled == 0
    assert diagnostics.skipped_no_dino_candidate == 1


def test_compiled_component_extraction_matches_python_8_connectivity() -> None:
    mask = np.zeros((8, 9), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True  # Diagonal pixels must remain one 8-connected component.
    mask[3:6, 4:7] = True
    mask[7, 8] = True  # Filtered by the minimum-area threshold.

    expected = region_verifier._connected_components_python(mask, class_index=3, minimum_area=2)
    actual = region_verifier._connected_components(mask, class_index=3, minimum_area=2)

    def signature(regions):
        return [(region.class_index, region.area, region.mask) for region in regions]

    expected_signature = signature(expected)
    actual_signature = signature(actual)
    assert [(class_index, area) for class_index, area, _ in actual_signature] == [
        (class_index, area) for class_index, area, _ in expected_signature
    ]
    for (_, _, actual_mask), (_, _, expected_mask) in zip(actual_signature, expected_signature):
        assert np.array_equal(actual_mask, expected_mask)
