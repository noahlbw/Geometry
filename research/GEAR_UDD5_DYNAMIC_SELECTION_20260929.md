# UDD5 variable-count alias screening: frozen exploratory test

Date: 2026-09-29. Target-label results below are pending; no claim of a winning
selector is made at launch.

## Question

The prior fixed-15 screen reduced UDD5 GEAR-OV mIoU from 50.0877 to 48.1441,
with road IoU falling from 44.0687 to 28.8562. Is this because 15 is too few,
because the five omitted *road* aliases were valuable, or because changing
other classes altered the competition for road pixels?

## Frozen dynamic rule

`DINOtool/scripts/select_gear_dynamic_udd5.py` uses the original 20 aliases
per class and only unlabeled UDD5 images (40 images, two fixed-seed tiles each).
For each alias it measures (1) the image-level native-counterfactual score
used in the previous screen and (2) how often the alias wins *within its
class* on patches where Geometry ranks that class in the top two. Remove an
alias only if its counterfactual score is positive with image-level z >= 1
**and** its competitive support is below 1%. Keep at least four aliases per
class. No fixed 15-word quota is imposed. The rule and thresholds were frozen
before this full target-mask evaluation. The resulting retained counts are
vegetation 13, building 19, road 18, vehicle 18, other 13.

This rule was motivated after inspecting an earlier labeled UDD5 *audit* of
why fixed-15 failed; it is therefore exploratory and not an untouched-set
validation. The selector itself never reads target labels.

## Full 40-image evaluation arms

All arms use the same frozen weights, class list, all-20 source vocabulary,
exact alias groups, Geometry/Multiscale/GEAR-OV readouts, and UDD5 protocol.

| Arm | Alias counts | Question |
|---|---|---|
| all20 (existing) | 20/20/20/20/20 | Reference |
| fixed15 (existing) | 15/15/15/15/15 | Failed previous screen |
| road20_others15 | 15/15/20/15/15 | Did omitted road aliases cause the road collapse? |
| road15_others20 | 20/20/15/20/20 | Did non-road pruning change class competition? |
| dynamic | 13/19/18/18/13 | Does variable-count, support-protected pruning help? |
| random15_s0..s2 | 15/15/15/15/15 | Is fixed15 better than equal-count random deletion? |
| randommatch_s0..s2 | 13/19/18/18/13 | Is dynamic better than random at the same class counts? |

The factorial controls use the exact fixed-15 selected words except in the
restored classes. Random controls use deterministic seeds and source-word order.
The first eight new arms run concurrently on A800 GPUs 0–7; the ninth starts
only after a GPU is idle. The held-out comparison of interest is full-image
mIoU and per-class IoU, especially road, vehicle, building, and other.

## Decision rule

If dynamic does not beat all20 and the matched random controls, it is not a
useful selector. If restoring road aliases alone repairs road IoU, the fixed-15
failure is mainly deletion of road evidence; if restoring only non-road aliases
repairs it, class competition is the dominant mechanism. An interaction is
possible, so no single-arm conclusion is forced. Any observed gain on these
40 UDD5 images remains exploratory because the rule was informed by UDD5 audit.

## Verified full 40-image results

All nine new arms completed. Merges have `coverage_verified=true`, 40/40
processed images, 40 unique sample keys, the same global sample-key SHA as the
all20 reference, `exact_alias_groups=true`, and alias counts identical to the
frozen vocabulary files. The baseline and fixed15 rows are the previously
verified full UDD5 results with the same sample keys and readouts.

| Arm | Counts (vegetation/building/road/vehicle/other) | Geometry | Multiscale | GEAR-OV | GEAR road IoU |
|---|---|---:|---:|---:|---:|
| all20 | 20/20/20/20/20 | 50.5553 | 51.0405 | 50.0877 | 44.0687 |
| fixed15 | 15/15/15/15/15 | 48.6152 | 49.7035 | 48.1441 | 28.8562 |
| road20_others15 | 15/15/20/15/15 | 50.8292 | 51.8727 | 50.2937 | 37.1010 |
| road15_others20 | 20/20/15/20/20 | 47.9881 | 48.3253 | 47.3736 | 32.2023 |
| dynamic | 13/19/18/18/13 | 48.9932 | 49.8615 | 48.4134 | 32.2617 |
| random15_s0 | 15/15/15/15/15 | 50.1161 | 50.9653 | 49.8813 | 41.9186 |
| random15_s1 | 15/15/15/15/15 | 50.6112 | 51.2272 | 50.4999 | 47.3990 |
| random15_s2 | 15/15/15/15/15 | 47.1883 | 47.3825 | 46.8469 | 41.2646 |
| randommatch_s0 | 13/19/18/18/13 | 48.8598 | 49.6878 | 48.7174 | 49.0178 |
| randommatch_s1 | 13/19/18/18/13 | 49.7785 | 50.0912 | 49.0233 | 42.5851 |
| randommatch_s2 | 13/19/18/18/13 | 49.1739 | 49.6099 | 48.9395 | 46.7822 |

The three random15 GEAR-OV mIoUs average 49.0760 (range 46.8469–50.4999).
The three random vocabularies matched to dynamic's class counts average
48.8934 (range 48.7174–49.0233). Dynamic achieves 48.4134: **1.6743 points
below all20 and 0.4800 below the matched-random mean**. Thus the proposed
image-only variable-count rule is rejected as a performance-improving selector
on this test. It protects `road`, `street`, and `wide road`, but it still drops
`streets` and `roadway`, and changes the competing classes as well. Its road
IoU of 32.2617 does not recover the all20 value of 44.0687.

The 2×2 factorial is more informative than the selector outcome:

- From fixed15, restoring only the five removed road aliases raises GEAR-OV
  mIoU by **2.1496** and road IoU by **8.2448**. Conversely, starting from
  all20 and pruning only road aliases lowers mIoU by **2.7141** and road IoU
  by **11.8664**. The removed road evidence was valuable in this setting.
- Restoring only the non-road classes from fixed15 lowers overall mIoU by
  0.7705, although road IoU rises by 3.3461. Starting from all20, pruning
  non-road aliases raises mIoU by 0.2060 but lowers road IoU by 6.9677.
  Class competition therefore also matters: aggregate gain can conceal a
  road loss.
- At least one random fixed-15 vocabulary beats all20 (50.4999 versus
  50.0877), so deleting aliases *can* help. But the three seeds span 3.6530
  mIoU points, and the mean is 1.0117 below all20. This is no evidence for
  a reliable general rule that fewer words are better.

GEAR-OV per-class IoU (vegetation/building/road/vehicle/other):

| Arm | Vegetation | Building | Road | Vehicle | Other |
|---|---:|---:|---:|---:|---:|
| all20 | 83.1009 | 83.3091 | 44.0687 | 9.5261 | 30.4338 |
| fixed15 | 81.6410 | 77.2727 | 28.8562 | 18.5963 | 34.3545 |
| road20_others15 | 81.5423 | 77.2868 | 37.1010 | 21.2566 | 34.2819 |
| road15_others20 | 83.3028 | 83.1845 | 32.2023 | 7.4695 | 30.7090 |
| dynamic | 82.2928 | 77.1757 | 32.2617 | 16.5368 | 33.8002 |

## Interpretation and next decision

The evidence argues against both simple extremes: not all 20 words are
individually necessary (some 15-word draws do better), but blindly removing
five or using this dynamic rule often removes useful evidence. UDD5's fixed15
road collapse is substantially explained by deleting road aliases, while
non-road alias changes and their interaction influence road IoU too. An alias
selector must evaluate its **marginal contribution under the full competing
vocabulary**, not only class-local uniqueness or agreement with native
DINO.text. The present dynamic rule should not be used as the final module.

These are descriptive comparisons on one 40-image dataset. The rule was
developed after seeing a labeled UDD5 audit, and no confidence intervals or
untouched dataset establish generalization. A next preregistered study should
test a cross-class counterfactual, with matched random controls, on a dataset
not used to design it; avoid optimizing this UDD5 score by repeatedly changing
thresholds.
