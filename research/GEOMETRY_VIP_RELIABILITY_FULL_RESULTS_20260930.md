# Geometry / VIP branch-selection feasibility: full VDD and Potsdam

2026-09-30. Frozen image-only rules, same 20 aliases per class. Full VDD 80 images and
Potsdam 504 RGB tiles from 14 parent scenes. No target-mask parameter selection.
These repeatedly consulted datasets are exploratory development, not untouched test sets.

The strong branch profiles intentionally differ: Geometry/Multiscale uses its existing
512/128 three-view RS-template protocol; VIP20 uses pinned proxy/ImageNet/448/336/112
with native scoring and no background threshold. This is not a head-only factorial.
Both runtime copies load the same frozen weights; the copies preserve native precision.

## Full mIoU

| Method | VDD | Delta vs Multiscale | Potsdam | Delta vs Multiscale |
|---|---:|---:|---:|---:|
| Geometry | 38.8511 | -2.3724 | 40.6892 | +0.2459 |
| Multiscale | 41.2235 | +0.0000 | 40.4433 | +0.0000 |
| VIP20 | 51.0697 | +9.8462 | 42.3519 | +1.9086 |
| MeanProb50 | 53.8316 | +12.6081 | 42.4761 | +2.0328 |
| MaxConfidence | 51.7483 | +10.5248 | 42.9481 | +2.5048 |
| LeaveFamilyOut | 52.9364 | +11.7129 | 41.1135 | +0.6702 |
| GroundedLeaveFamilyOut | 52.9117 | +11.6882 | 40.9863 | +0.5430 |

## Control reproduction

- vdd: exact historical confusion equality {'Geometry': True, 'Multiscale': True, 'VIP20': True}.
- potsdam: exact historical confusion equality {'Geometry': True, 'Multiscale': True, 'VIP20': True}.

## Complementarity and routing

Pattern order: both wrong, Multiscale only correct, VIP only correct, both correct.
The oracle is a GT-only pixel-accuracy audit, not a usable rule or an mIoU bound.

### vdd

Full-resolution pattern counts: [155462387, 94747617, 191999636, 517790360].
GT-only any-correct accuracy: 83.8060%.

| Selector | Beneficial retention | Harmful rejection | XOR route accuracy | Fixes vs Multiscale | Harms vs Multiscale |
|---|---:|---:|---:|---:|---:|
| MaxConfidence | 99.71% | 2.09% | 67.45% | 191434945 | 92766744 |
| LeaveFamilyOut | 73.61% | 68.09% | 71.78% | 141323136 | 30238359 |
| GroundedLeaveFamilyOut | 71.55% | 72.77% | 71.95% | 137371044 | 25795875 |

| Method | IoU by class |
|---|---|
| Geometry | other=15.2481; wall=22.9675; road=48.5684; vegetation=70.025; vehicle=9.4839; roof=65.8483; water=39.8168 |
| Multiscale | other=14.419; wall=24.3485; road=51.0981; vegetation=72.0733; vehicle=9.2742; roof=70.6678; water=46.6838 |
| VIP20 | other=33.932; wall=36.7687; road=38.3941; vegetation=61.3136; vehicle=23.0906; roof=80.7481; water=83.2407 |
| MeanProb50 | other=34.39; wall=44.8855; road=40.3079; vegetation=64.586; vehicle=25.0886; roof=82.396; water=85.1669 |
| MaxConfidence | other=33.8846; wall=38.9787; road=38.5282; vegetation=61.4553; vehicle=25.2201; roof=80.9246; water=83.2469 |
| LeaveFamilyOut | other=30.2871; wall=42.3392; road=43.9492; vegetation=69.1901; vehicle=18.211; roof=81.7557; water=84.8225 |
| GroundedLeaveFamilyOut | other=29.7377; wall=42.071; road=44.4346; vegetation=69.8664; vehicle=17.4844; roof=81.8488; water=84.9387 |

| Method / class | Predicted area % | Precision % | Recall % |
|---|---:|---:|---:|
| Multiscale / other | 19.2946 | 24.8882 | 25.5277 |
| Multiscale / wall | 10.9216 | 25.3817 | 85.6770 |
| Multiscale / road | 9.5199 | 53.3172 | 92.4681 |
| Multiscale / vegetation | 28.0057 | 95.3002 | 74.7294 |
| Multiscale / vehicle | 5.1689 | 9.3419 | 92.7493 |
| Multiscale / roof | 18.9573 | 87.3662 | 78.7114 |
| Multiscale / water | 8.1320 | 91.2630 | 48.8679 |
| VIP20 / other | 18.7371 | 50.7707 | 50.5706 |
| VIP20 / wall | 1.9072 | 72.4919 | 42.7306 |
| VIP20 / road | 12.5020 | 39.9233 | 90.9283 |
| VIP20 / vegetation | 23.5706 | 95.6011 | 63.0936 |
| VIP20 / vehicle | 1.3234 | 26.1390 | 66.4422 |
| VIP20 / roof | 25.2720 | 81.8709 | 98.3299 |
| VIP20 / water | 16.6877 | 86.7682 | 95.3435 |
| MeanProb50 / other | 17.9336 | 52.4316 | 49.9855 |
| MeanProb50 / wall | 2.3850 | 73.0069 | 53.8168 |
| MeanProb50 / road | 12.3979 | 41.4477 | 93.6136 |
| MeanProb50 / vegetation | 24.4846 | 96.4817 | 66.1439 |
| MeanProb50 / vehicle | 1.5402 | 26.8362 | 79.3925 |
| MeanProb50 / roof | 24.9467 | 83.2774 | 98.7319 |
| MeanProb50 / water | 16.3119 | 88.8168 | 95.3968 |
| GroundedLeaveFamilyOut / other | 16.7595 | 48.6489 | 43.3428 |
| GroundedLeaveFamilyOut / wall | 4.8355 | 49.4271 | 73.8688 |
| GroundedLeaveFamilyOut / road | 11.4266 | 45.5434 | 94.8057 |
| GroundedLeaveFamilyOut / vegetation | 26.2621 | 97.0647 | 71.3744 |
| GroundedLeaveFamilyOut / vehicle | 2.6500 | 17.8061 | 90.6337 |
| GroundedLeaveFamilyOut / roof | 22.9576 | 86.2624 | 94.1166 |
| GroundedLeaveFamilyOut / water | 15.1087 | 92.0937 | 91.6197 |

Patch diagnostics: `{"broad_crops": 160, "disagreement_patches": 2670082, "grounded_proposed": 1637421, "identifiable_disagreements": 2670082, "lfo_proposed": 1725545, "neighbor_supported_disagreements": 2670082, "tiles": 7040, "valid_patches": 7208960}`.
Parallel wall 516.65s; aggregate GPU 2050.86s;
peak allocated CUDA 5788.23 MiB (combined evaluation, two frozen runtime copies).

### potsdam

Full-resolution pattern counts: [131535650, 34602300, 50253983, 287608067].
GT-only any-correct accuracy: 73.9017%.

| Selector | Beneficial retention | Harmful rejection | XOR route accuracy | Fixes vs Multiscale | Harms vs Multiscale |
|---|---:|---:|---:|---:|---:|
| MaxConfidence | 99.26% | 11.09% | 63.31% | 49883535 | 30765818 |
| LeaveFamilyOut | 40.23% | 70.26% | 52.48% | 20218884 | 10290173 |
| GroundedLeaveFamilyOut | 32.45% | 77.04% | 50.63% | 16306907 | 7946357 |

| Method | IoU by class |
|---|---|
| Geometry | impervious surface=53.0394; building=79.545; low vegetation=39.927; tree=52.0633; car=10.8915; clutter=8.6688 |
| Multiscale | impervious surface=51.6213; building=79.6181; low vegetation=40.8567; tree=52.3325; car=10.0975; clutter=8.1333 |
| VIP20 | impervious surface=58.7167; building=76.2799; low vegetation=30.1486; tree=53.5234; car=29.3108; clutter=6.1321 |
| MeanProb50 | impervious surface=58.3783; building=77.2131; low vegetation=32.388; tree=54.9129; car=25.6274; clutter=6.337 |
| MaxConfidence | impervious surface=59.9111; building=76.3562; low vegetation=32.9852; tree=53.7713; car=28.4196; clutter=6.2455 |
| LeaveFamilyOut | impervious surface=55.6569; building=78.1581; low vegetation=36.9407; tree=54.0406; car=14.6861; clutter=7.1983 |
| GroundedLeaveFamilyOut | impervious surface=55.2146; building=78.7116; low vegetation=37.4954; tree=53.6146; car=13.5883; clutter=7.2932 |

| Method / class | Predicted area % | Precision % | Recall % |
|---|---:|---:|---:|
| Multiscale / impervious surface | 26.5898 | 74.3427 | 62.8116 |
| Multiscale / building | 26.8998 | 83.7416 | 94.1757 |
| Multiscale / low vegetation | 9.6145 | 92.3872 | 42.2802 |
| Multiscale / tree | 12.2235 | 82.2486 | 58.9959 |
| Multiscale / car | 19.3362 | 10.1034 | 99.4256 |
| Multiscale / clutter | 5.3363 | 13.9974 | 16.2577 |
| VIP20 / impervious surface | 34.7991 | 70.4514 | 77.9013 |
| VIP20 / building | 29.1697 | 78.7555 | 96.0422 |
| VIP20 / low vegetation | 7.1874 | 90.8753 | 31.0898 |
| VIP20 / tree | 12.4069 | 82.7494 | 60.2455 |
| VIP20 / car | 6.4431 | 29.5796 | 96.9933 |
| VIP20 / clutter | 9.9939 | 8.4340 | 18.3460 |
| MeanProb50 / impervious surface | 34.3285 | 70.6519 | 77.0667 |
| MeanProb50 / building | 28.7965 | 79.7623 | 96.0254 |
| MeanProb50 / low vegetation | 7.6610 | 91.5536 | 33.3856 |
| MeanProb50 / tree | 12.6322 | 83.2675 | 61.7239 |
| MeanProb50 / car | 7.5655 | 25.6977 | 98.9438 |
| MeanProb50 / clutter | 9.0163 | 8.9961 | 17.6544 |
| GroundedLeaveFamilyOut / impervious surface | 29.6464 | 73.3356 | 69.0836 |
| GroundedLeaveFamilyOut / building | 27.5248 | 82.3188 | 94.7265 |
| GroundedLeaveFamilyOut / low vegetation | 8.7869 | 92.4714 | 38.6760 |
| GroundedLeaveFamilyOut / tree | 12.5037 | 82.4700 | 60.5106 |
| GroundedLeaveFamilyOut / car | 14.3543 | 13.6003 | 99.3550 |
| GroundedLeaveFamilyOut / clutter | 7.1839 | 11.1446 | 17.4261 |

Patch diagnostics: `{"broad_crops": 2016, "disagreement_patches": 1227921, "grounded_proposed": 508372, "identifiable_disagreements": 1227921, "lfo_proposed": 612293, "neighbor_supported_disagreements": 1227921, "tiles": 4536, "valid_patches": 4644864}`.
Parallel wall 275.36s; aggregate GPU 1093.01s;
peak allocated CUDA 5663.68 MiB (combined evaluation, two frozen runtime copies).

## Predeclared mechanism decision

GroundedLeaveFamilyOut exceeds Multiscale, VIP20, fixed equal-probability fusion and
max-confidence selection on both development datasets: **False**.

This decision does not establish statistical significance, eight-dataset transfer or novelty.
Failure does not justify retuning this selector on these masks or renaming it a successful final model.
All counterfactual, area and per-class counts are audit-only; they did not enter prediction.
