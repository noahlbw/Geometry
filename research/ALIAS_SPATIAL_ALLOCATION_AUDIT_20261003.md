# Frozen Spatial Allocation: Verified64-Window Diagnostic

64 fixed unique development image IDs, eight/domain, ONE top-left512 window/image. Same original whole-image wide observations; frozen checkpoints, public20 aliases/class and reconstruction. No encoder loaded or rerun, no label-based fitting. Scores persisted before masks. LoveDA P/D share images; means count D once. LandCover.ai substitutes for unlabeled iSAID; corrected IRRG is retained. These are not full-dataset mIoUs.

## Decision

```json
{
  "primary_above_all_feasible_allocation_controls": true,
  "mean_differences_pp": {
    "CapacityMean_Exact": 0.12308279376722453,
    "SpatialShuffle0_Exact": 0.12041220326656799,
    "SpatialShuffle1_Exact": 0.1413374726345893,
    "SpatialShuffle2_Exact": 0.14156853273951242
  },
  "interpretation": "Positional evidence in this cached development diagnostic only.",
  "previous_promotion_gate_still_failed": true,
  "no_full_rollout": true
}
```

This is a descriptive budget-matched diagnostic, not significance or a new promotion gate. RawMean is unrestricted if it violates protected alias capacity. CapacityMean and feasible shuffles retain each class suppression total; shuffles also preserve its exact action spectrum. Capacity constraints condition the permutation distribution, so it is not uniform over all permutations.

## Fixed-Class Paired Metrics

The original coupling union-positive class set stays fixed; a scored class disappearing after intervention remains IoU0.

| Dataset/protocol | Geometry | Anchored_Exact | TargetContext_Exact | RawMean_Exact | CapacityMean_Exact | SpatialShuffle0_Exact | SpatialShuffle1_Exact | SpatialShuffle2_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 54.455821 | 54.356319 | 54.356319 | 54.393054 | 54.369978 | 54.345758 |
| potsdam/potsdam | 35.703546 | 38.920258 | 38.949938 | 38.967256 | 38.967256 | 38.984888 | 38.973787 | 38.976389 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.877049 | 28.417010 | 28.417010 | 28.395929 | 28.372816 | 28.436761 |
| oem/oem | 39.820583 | 39.023152 | 39.076361 | 39.110295 | 39.110295 | 39.099851 | 39.104438 | 39.112740 |
| loveda/P | 49.539928 | 50.706552 | 50.681958 | 49.860320 | 49.860320 | 49.982791 | 48.983293 | 49.911530 |
| loveda/D | 33.877920 | 30.736030 | 31.446522 | 30.941112 | 30.941112 | 30.940464 | 30.841322 | 30.823110 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.755697 | 51.765443 | 51.765443 | 51.735691 | 51.762476 | 51.743279 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.929129 | 66.941659 | 66.941659 | 66.949762 | 66.925295 | 66.939707 |
| flair1/flair1 | 35.605867 | 33.608404 | 33.466656 | 33.473416 | 33.473416 | 33.494236 | 33.476361 | 33.446881 |
| Equal-domain mean | 40.531033 | 42.867949 | 43.119647 | 42.996564 | 42.996564 | 42.999234 | 42.978309 | 42.978078 |

## Historical Standard Metrics

| Dataset/protocol | Geometry | Anchored_Exact | TargetContext_Exact | RawMean_Exact | CapacityMean_Exact | SpatialShuffle0_Exact | SpatialShuffle1_Exact | SpatialShuffle2_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 54.455821 | 54.356319 | 54.356319 | 54.393054 | 54.369978 | 54.345758 |
| potsdam/potsdam | 35.703546 | 38.920258 | 38.949938 | 38.967256 | 38.967256 | 38.984888 | 38.973787 | 38.976389 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.877049 | 28.417010 | 28.417010 | 28.395929 | 28.372816 | 28.436761 |
| oem/oem | 39.820583 | 39.023152 | 39.076361 | 39.110295 | 39.110295 | 39.099851 | 39.104438 | 39.112740 |
| loveda/P | 49.539928 | 50.706552 | 50.681958 | 49.860320 | 49.860320 | 49.982791 | 48.983293 | 49.911530 |
| loveda/D | 33.877920 | 30.736030 | 31.446522 | 30.941112 | 30.941112 | 30.940464 | 30.841322 | 30.823110 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.755697 | 51.765443 | 51.765443 | 51.735691 | 51.762476 | 51.743279 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.929129 | 66.941659 | 66.941659 | 66.949762 | 66.925295 | 66.939707 |
| flair1/flair1 | 38.842764 | 33.608404 | 33.466656 | 33.473416 | 33.473416 | 33.494236 | 33.476361 | 33.446881 |
| Equal-domain mean | 40.935645 | 42.867949 | 43.119647 | 42.996564 | 42.996564 | 42.999234 | 42.978309 | 42.978078 |

## Numerical Allocation Checks

| Dataset | Replay max error | Cap enlargement max | Budget max error | Shuffle spectrum max error | RawMean cap excess max | RawMean violating donor/classes | Shuffle changed fraction mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 0.000e+00 | 0.000e+00 | 5.294e-13 | 0.000e+00 | 0.000000 | 0 | 0.890776 |
| potsdam | 0.000e+00 | 0.000e+00 | 1.002e-12 | 0.000e+00 | 0.000000 | 0 | 0.904453 |
| udd5 | 0.000e+00 | 0.000e+00 | 1.357e-12 | 0.000e+00 | 0.000000 | 0 | 0.856901 |
| oem | 0.000e+00 | 0.000e+00 | 1.492e-12 | 0.000e+00 | 0.000000 | 0 | 0.915227 |
| loveda | 0.000e+00 | 0.000e+00 | 1.158e-12 | 0.000e+00 | 0.000000 | 0 | 0.890443 |
| vaihingen | 0.000e+00 | 0.000e+00 | 1.265e-12 | 0.000e+00 | 0.000000 | 0 | 0.904012 |
| landcoverai | 0.000e+00 | 0.000e+00 | 1.144e-12 | 0.000e+00 | 0.000000 | 0 | 0.843416 |
| flair1 | 4.441e-16 | 0.000e+00 | 1.606e-12 | 0.000e+00 | 0.000000 | 0 | 0.939073 |

## Correction Transitions

| Dataset/protocol | Method | Beneficial pixels | Harmful pixels | Wrong-to-wrong pixels |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | Geometry | 200243 | 664465 | 229565 |
| vdd/vdd | TargetContext_Exact | 10515 | 3627 | 6316 |
| vdd/vdd | RawMean_Exact | 10340 | 6454 | 7061 |
| vdd/vdd | CapacityMean_Exact | 10340 | 6454 | 7061 |
| vdd/vdd | SpatialShuffle0_Exact | 10572 | 6160 | 7153 |
| vdd/vdd | SpatialShuffle1_Exact | 10655 | 6403 | 6982 |
| vdd/vdd | SpatialShuffle2_Exact | 10368 | 6364 | 7184 |
| potsdam/potsdam | Geometry | 98825 | 248015 | 200337 |
| potsdam/potsdam | TargetContext_Exact | 7137 | 6569 | 4902 |
| potsdam/potsdam | RawMean_Exact | 5367 | 4289 | 3855 |
| potsdam/potsdam | CapacityMean_Exact | 5367 | 4289 | 3855 |
| potsdam/potsdam | SpatialShuffle0_Exact | 5879 | 4395 | 4027 |
| potsdam/potsdam | SpatialShuffle1_Exact | 5683 | 4515 | 3986 |
| potsdam/potsdam | SpatialShuffle2_Exact | 5695 | 4425 | 3945 |
| udd5/udd5 | Geometry | 29933 | 87090 | 153751 |
| udd5/udd5 | TargetContext_Exact | 9608 | 928 | 11452 |
| udd5/udd5 | RawMean_Exact | 6983 | 1582 | 11247 |
| udd5/udd5 | CapacityMean_Exact | 6983 | 1582 | 11247 |
| udd5/udd5 | SpatialShuffle0_Exact | 6944 | 1647 | 11783 |
| udd5/udd5 | SpatialShuffle1_Exact | 6812 | 1718 | 11364 |
| udd5/udd5 | SpatialShuffle2_Exact | 6961 | 1475 | 11336 |
| oem/oem | Geometry | 215866 | 181426 | 84092 |
| oem/oem | TargetContext_Exact | 12763 | 5169 | 1835 |
| oem/oem | RawMean_Exact | 13265 | 5020 | 2672 |
| oem/oem | CapacityMean_Exact | 13265 | 5020 | 2672 |
| oem/oem | SpatialShuffle0_Exact | 12957 | 5214 | 2755 |
| oem/oem | SpatialShuffle1_Exact | 13147 | 5064 | 2922 |
| oem/oem | SpatialShuffle2_Exact | 13239 | 4973 | 2618 |
| loveda/P | Geometry | 78545 | 16195 | 37158 |
| loveda/P | TargetContext_Exact | 655 | 1800 | 1441 |
| loveda/P | RawMean_Exact | 382 | 2621 | 2238 |
| loveda/P | CapacityMean_Exact | 382 | 2621 | 2238 |
| loveda/P | SpatialShuffle0_Exact | 382 | 2625 | 2180 |
| loveda/P | SpatialShuffle1_Exact | 424 | 2704 | 2221 |
| loveda/P | SpatialShuffle2_Exact | 368 | 2529 | 2145 |
| loveda/D | Geometry | 453985 | 129960 | 264121 |
| loveda/D | TargetContext_Exact | 34796 | 2679 | 5644 |
| loveda/D | RawMean_Exact | 24272 | 3121 | 7005 |
| loveda/D | CapacityMean_Exact | 24272 | 3121 | 7005 |
| loveda/D | SpatialShuffle0_Exact | 25524 | 3178 | 7321 |
| loveda/D | SpatialShuffle1_Exact | 24049 | 3323 | 7450 |
| loveda/D | SpatialShuffle2_Exact | 23509 | 3113 | 7203 |
| vaihingen/vaihingen | Geometry | 97388 | 179006 | 102887 |
| vaihingen/vaihingen | TargetContext_Exact | 4201 | 6990 | 4122 |
| vaihingen/vaihingen | RawMean_Exact | 3892 | 6351 | 3661 |
| vaihingen/vaihingen | CapacityMean_Exact | 3892 | 6351 | 3661 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | 3636 | 6644 | 3671 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | 3664 | 6035 | 3438 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | 3713 | 6615 | 3867 |
| landcoverai/landcoverai | Geometry | 37157 | 113587 | 11811 |
| landcoverai/landcoverai | TargetContext_Exact | 4522 | 2635 | 41 |
| landcoverai/landcoverai | RawMean_Exact | 4261 | 2714 | 33 |
| landcoverai/landcoverai | CapacityMean_Exact | 4261 | 2714 | 33 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | 4224 | 2631 | 46 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | 4071 | 2757 | 49 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | 4101 | 2689 | 33 |
| flair1/flair1 | Geometry | 134726 | 98118 | 101041 |
| flair1/flair1 | TargetContext_Exact | 6233 | 14117 | 7190 |
| flair1/flair1 | RawMean_Exact | 5434 | 10579 | 7183 |
| flair1/flair1 | CapacityMean_Exact | 5434 | 10579 | 7183 |
| flair1/flair1 | SpatialShuffle0_Exact | 5781 | 11213 | 7555 |
| flair1/flair1 | SpatialShuffle1_Exact | 5547 | 10590 | 7348 |
| flair1/flair1 | SpatialShuffle2_Exact | 5237 | 11237 | 7160 |

## Per-Class Outcomes

Each row includes IoU, precision, recall and prediction area on evaluated valid pixels. None are used to choose a method per domain.

| Dataset/protocol | Method | Class | IoU % | Delta vs original pp | Precision % | Recall % | Area % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -47.4915 | 70.4934 | 12.9511 | 9.3800 |
| vdd/vdd | Geometry | wall | 19.8084 | -39.8732 | 20.3031 | 89.0471 | 34.8907 |
| vdd/vdd | Geometry | road | 24.9744 | 1.7494 | 29.4846 | 62.0157 | 6.9810 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 25.6066 | 89.0479 | 76.2942 | 19.7359 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -46.4204 | 3.8880 | 100.0000 | 9.2105 |
| vdd/vdd | Geometry | roof | 63.9822 | -22.2785 | 64.6648 | 98.3769 | 9.1667 |
| vdd/vdd | Geometry | water | 74.5988 | 21.7649 | 75.8754 | 97.7943 | 10.6353 |
| vdd/vdd | Anchored_Exact | other | 59.7766 | 0.0000 | 81.1541 | 69.4120 | 43.6684 |
| vdd/vdd | Anchored_Exact | wall | 59.6816 | 0.0000 | 65.8225 | 86.4811 | 10.4520 |
| vdd/vdd | Anchored_Exact | road | 23.2250 | 0.0000 | 23.3759 | 97.2962 | 13.8146 |
| vdd/vdd | Anchored_Exact | vegetation | 44.1426 | 0.0000 | 94.9106 | 45.2128 | 10.9733 |
| vdd/vdd | Anchored_Exact | vehicle | 50.3084 | 0.0000 | 50.3286 | 99.9201 | 0.7110 |
| vdd/vdd | Anchored_Exact | roof | 86.2607 | 0.0000 | 86.7114 | 99.4009 | 6.9072 |
| vdd/vdd | Anchored_Exact | water | 52.8339 | 0.0000 | 55.7407 | 91.0164 | 13.4736 |
| vdd/vdd | TargetContext_Exact | other | 60.2012 | 0.4246 | 80.9440 | 70.1422 | 44.2423 |
| vdd/vdd | TargetContext_Exact | wall | 59.9532 | 0.2716 | 66.3858 | 86.0867 | 10.3160 |
| vdd/vdd | TargetContext_Exact | road | 23.1865 | -0.0385 | 23.3132 | 97.7099 | 13.9107 |
| vdd/vdd | TargetContext_Exact | vegetation | 44.2118 | 0.0692 | 95.3073 | 45.1956 | 10.9234 |
| vdd/vdd | TargetContext_Exact | vehicle | 52.6445 | 2.3361 | 52.7000 | 99.8003 | 0.6782 |
| vdd/vdd | TargetContext_Exact | roof | 86.8556 | 0.5949 | 87.3512 | 99.3511 | 6.8532 |
| vdd/vdd | TargetContext_Exact | water | 54.1380 | 1.3041 | 57.2870 | 90.7823 | 13.0762 |
| vdd/vdd | RawMean_Exact | other | 60.2173 | 0.4407 | 80.8750 | 70.2160 | 44.3267 |
| vdd/vdd | RawMean_Exact | wall | 60.4605 | 0.7789 | 66.6708 | 86.6501 | 10.3392 |
| vdd/vdd | RawMean_Exact | road | 22.8689 | -0.3561 | 23.0023 | 97.5260 | 14.0721 |
| vdd/vdd | RawMean_Exact | vegetation | 43.8407 | -0.3019 | 95.2232 | 44.8265 | 10.8438 |
| vdd/vdd | RawMean_Exact | vehicle | 52.4036 | 2.0952 | 52.4402 | 99.8668 | 0.6820 |
| vdd/vdd | RawMean_Exact | roof | 87.5185 | 1.2578 | 88.0098 | 99.3661 | 6.8029 |
| vdd/vdd | RawMean_Exact | water | 53.1849 | 0.3510 | 56.8707 | 89.1377 | 12.9333 |
| vdd/vdd | CapacityMean_Exact | other | 60.2173 | 0.4407 | 80.8750 | 70.2160 | 44.3267 |
| vdd/vdd | CapacityMean_Exact | wall | 60.4605 | 0.7789 | 66.6708 | 86.6501 | 10.3392 |
| vdd/vdd | CapacityMean_Exact | road | 22.8689 | -0.3561 | 23.0023 | 97.5260 | 14.0721 |
| vdd/vdd | CapacityMean_Exact | vegetation | 43.8407 | -0.3019 | 95.2232 | 44.8265 | 10.8438 |
| vdd/vdd | CapacityMean_Exact | vehicle | 52.4036 | 2.0952 | 52.4402 | 99.8668 | 0.6820 |
| vdd/vdd | CapacityMean_Exact | roof | 87.5185 | 1.2578 | 88.0098 | 99.3661 | 6.8029 |
| vdd/vdd | CapacityMean_Exact | water | 53.1849 | 0.3510 | 56.8707 | 89.1377 | 12.9333 |
| vdd/vdd | SpatialShuffle0_Exact | other | 60.2395 | 0.4629 | 80.8808 | 70.2418 | 44.3398 |
| vdd/vdd | SpatialShuffle0_Exact | wall | 60.4829 | 0.8013 | 66.6945 | 86.6561 | 10.3362 |
| vdd/vdd | SpatialShuffle0_Exact | road | 22.8892 | -0.3358 | 23.0251 | 97.4858 | 14.0524 |
| vdd/vdd | SpatialShuffle0_Exact | vegetation | 43.8488 | -0.2938 | 95.2093 | 44.8381 | 10.8482 |
| vdd/vdd | SpatialShuffle0_Exact | vehicle | 52.5427 | 2.2343 | 52.5759 | 99.8802 | 0.6803 |
| vdd/vdd | SpatialShuffle0_Exact | roof | 87.4611 | 1.2004 | 87.9524 | 99.3653 | 6.8073 |
| vdd/vdd | SpatialShuffle0_Exact | water | 53.2872 | 0.4533 | 56.9377 | 89.2602 | 12.9358 |
| vdd/vdd | SpatialShuffle1_Exact | other | 60.2362 | 0.4596 | 80.8751 | 70.2416 | 44.3428 |
| vdd/vdd | SpatialShuffle1_Exact | wall | 60.4800 | 0.7984 | 66.7059 | 86.6309 | 10.3314 |
| vdd/vdd | SpatialShuffle1_Exact | road | 22.8895 | -0.3355 | 23.0245 | 97.5016 | 14.0550 |
| vdd/vdd | SpatialShuffle1_Exact | vegetation | 43.8519 | -0.2907 | 95.2395 | 44.8346 | 10.8439 |
| vdd/vdd | SpatialShuffle1_Exact | vehicle | 52.4736 | 2.1652 | 52.5140 | 99.8535 | 0.6809 |
| vdd/vdd | SpatialShuffle1_Exact | roof | 87.4299 | 1.1692 | 87.9215 | 99.3645 | 6.8096 |
| vdd/vdd | SpatialShuffle1_Exact | water | 53.2288 | 0.3949 | 56.8962 | 89.1984 | 12.9363 |
| vdd/vdd | SpatialShuffle2_Exact | other | 60.2288 | 0.4522 | 80.8873 | 70.2224 | 44.3240 |
| vdd/vdd | SpatialShuffle2_Exact | wall | 60.4459 | 0.7643 | 66.6637 | 86.6321 | 10.3381 |
| vdd/vdd | SpatialShuffle2_Exact | road | 22.8538 | -0.3712 | 22.9884 | 97.5016 | 14.0771 |
| vdd/vdd | SpatialShuffle2_Exact | vegetation | 43.8660 | -0.2766 | 95.2754 | 44.8414 | 10.8415 |
| vdd/vdd | SpatialShuffle2_Exact | vehicle | 52.3597 | 2.0513 | 52.3963 | 99.8668 | 0.6825 |
| vdd/vdd | SpatialShuffle2_Exact | roof | 87.4497 | 1.1890 | 87.9397 | 99.3669 | 6.8084 |
| vdd/vdd | SpatialShuffle2_Exact | water | 53.2164 | 0.3825 | 56.9012 | 89.1516 | 12.9284 |
| potsdam/potsdam | Geometry | impervious surface | 48.6703 | -17.2274 | 84.3816 | 53.4888 | 25.9112 |
| potsdam/potsdam | Geometry | building | 72.8222 | 1.7693 | 79.5088 | 89.6472 | 18.1787 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 12.1298 | 77.4031 | 28.7716 | 6.4458 |
| potsdam/potsdam | Geometry | tree | 49.3365 | -6.1699 | 84.7437 | 54.1457 | 11.2731 |
| potsdam/potsdam | Geometry | car | 11.6958 | -12.8420 | 11.6965 | 99.9455 | 30.6388 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 3.0400 | 7.7773 | 13.2570 | 7.5523 |
| potsdam/potsdam | Anchored_Exact | impervious surface | 65.8977 | 0.0000 | 85.6706 | 74.0609 | 35.3370 |
| potsdam/potsdam | Anchored_Exact | building | 71.0529 | 0.0000 | 74.9985 | 93.1063 | 20.0156 |
| potsdam/potsdam | Anchored_Exact | low vegetation | 14.4124 | 0.0000 | 79.9195 | 14.9539 | 3.2447 |
| potsdam/potsdam | Anchored_Exact | tree | 55.5064 | 0.0000 | 91.9038 | 58.3601 | 11.2039 |
| potsdam/potsdam | Anchored_Exact | car | 24.5378 | 0.0000 | 24.5936 | 99.0837 | 14.4459 |
| potsdam/potsdam | Anchored_Exact | clutter | 2.1143 | 0.0000 | 2.6529 | 9.4321 | 15.7528 |
| potsdam/potsdam | TargetContext_Exact | impervious surface | 65.2643 | -0.6334 | 85.3578 | 73.4919 | 35.1941 |
| potsdam/potsdam | TargetContext_Exact | building | 70.8956 | -0.1573 | 74.7569 | 93.2092 | 20.1025 |
| potsdam/potsdam | TargetContext_Exact | low vegetation | 15.4050 | 0.9926 | 81.3387 | 15.9694 | 3.4046 |
| potsdam/potsdam | TargetContext_Exact | tree | 55.8624 | 0.3560 | 91.9003 | 58.7552 | 11.2802 |
| potsdam/potsdam | TargetContext_Exact | car | 24.1387 | -0.3991 | 24.1886 | 99.1542 | 14.6983 |
| potsdam/potsdam | TargetContext_Exact | clutter | 2.1336 | 0.0193 | 2.6932 | 9.3126 | 15.3204 |
| potsdam/potsdam | RawMean_Exact | impervious surface | 65.4170 | -0.4807 | 85.4104 | 73.6465 | 35.2464 |
| potsdam/potsdam | RawMean_Exact | building | 71.0398 | -0.0131 | 74.9236 | 93.1995 | 20.0557 |
| potsdam/potsdam | RawMean_Exact | low vegetation | 15.1303 | 0.7179 | 80.6640 | 15.6997 | 3.3751 |
| potsdam/potsdam | RawMean_Exact | tree | 55.9406 | 0.4342 | 91.9121 | 58.8369 | 11.2944 |
| potsdam/potsdam | RawMean_Exact | car | 24.1537 | -0.3841 | 24.2094 | 99.0558 | 14.6710 |
| potsdam/potsdam | RawMean_Exact | clutter | 2.1222 | 0.0079 | 2.6777 | 9.2814 | 15.3574 |
| potsdam/potsdam | CapacityMean_Exact | impervious surface | 65.4170 | -0.4807 | 85.4104 | 73.6465 | 35.2464 |
| potsdam/potsdam | CapacityMean_Exact | building | 71.0398 | -0.0131 | 74.9236 | 93.1995 | 20.0557 |
| potsdam/potsdam | CapacityMean_Exact | low vegetation | 15.1303 | 0.7179 | 80.6640 | 15.6997 | 3.3751 |
| potsdam/potsdam | CapacityMean_Exact | tree | 55.9406 | 0.4342 | 91.9121 | 58.8369 | 11.2944 |
| potsdam/potsdam | CapacityMean_Exact | car | 24.1537 | -0.3841 | 24.2094 | 99.0558 | 14.6710 |
| potsdam/potsdam | CapacityMean_Exact | clutter | 2.1222 | 0.0079 | 2.6777 | 9.2814 | 15.3574 |
| potsdam/potsdam | SpatialShuffle0_Exact | impervious surface | 65.4140 | -0.4837 | 85.4321 | 73.6266 | 35.2279 |
| potsdam/potsdam | SpatialShuffle0_Exact | building | 71.0245 | -0.0284 | 74.8920 | 93.2220 | 20.0690 |
| potsdam/potsdam | SpatialShuffle0_Exact | low vegetation | 15.2103 | 0.7979 | 80.6544 | 15.7863 | 3.3941 |
| potsdam/potsdam | SpatialShuffle0_Exact | tree | 55.9915 | 0.4851 | 91.9281 | 58.8866 | 11.3020 |
| potsdam/potsdam | SpatialShuffle0_Exact | car | 24.1472 | -0.3906 | 24.2016 | 99.0771 | 14.6789 |
| potsdam/potsdam | SpatialShuffle0_Exact | clutter | 2.1219 | 0.0076 | 2.6784 | 9.2663 | 15.3282 |
| potsdam/potsdam | SpatialShuffle1_Exact | impervious surface | 65.4163 | -0.4814 | 85.4147 | 73.6424 | 35.2427 |
| potsdam/potsdam | SpatialShuffle1_Exact | building | 71.0464 | -0.0065 | 74.9322 | 93.1974 | 20.0529 |
| potsdam/potsdam | SpatialShuffle1_Exact | low vegetation | 15.2045 | 0.7921 | 80.7764 | 15.7753 | 3.3866 |
| potsdam/potsdam | SpatialShuffle1_Exact | tree | 55.9039 | 0.3975 | 91.9239 | 58.7914 | 11.2843 |
| potsdam/potsdam | SpatialShuffle1_Exact | car | 24.1440 | -0.3938 | 24.1994 | 99.0611 | 14.6779 |
| potsdam/potsdam | SpatialShuffle1_Exact | clutter | 2.1277 | 0.0134 | 2.6845 | 9.3040 | 15.3557 |
| potsdam/potsdam | SpatialShuffle2_Exact | impervious surface | 65.4184 | -0.4793 | 85.4274 | 73.6357 | 35.2342 |
| potsdam/potsdam | SpatialShuffle2_Exact | building | 71.0360 | -0.0169 | 74.9239 | 93.1924 | 20.0541 |
| potsdam/potsdam | SpatialShuffle2_Exact | low vegetation | 15.1586 | 0.7462 | 80.7075 | 15.7285 | 3.3794 |
| potsdam/potsdam | SpatialShuffle2_Exact | tree | 55.9810 | 0.4746 | 91.8987 | 58.8871 | 11.3057 |
| potsdam/potsdam | SpatialShuffle2_Exact | car | 24.1371 | -0.4007 | 24.1924 | 99.0611 | 14.6821 |
| potsdam/potsdam | SpatialShuffle2_Exact | clutter | 2.1272 | 0.0129 | 2.6843 | 9.2965 | 15.3445 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | 13.9675 | 89.6338 | 64.7597 | 2.2305 |
| udd5/udd5 | Geometry | building | 83.4755 | -0.0116 | 86.9865 | 95.3877 | 78.8720 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -1.6538 | 3.7107 | 99.9773 | 11.3160 |
| udd5/udd5 | Geometry | other | 5.0713 | -0.6763 | 23.0553 | 6.1044 | 5.3797 |
| udd5/udd5 | Anchored_Exact | vegetation | 46.2799 | 0.0000 | 93.8774 | 47.7203 | 1.5693 |
| udd5/udd5 | Anchored_Exact | building | 83.4871 | 0.0000 | 83.6594 | 99.7539 | 85.7625 |
| udd5/udd5 | Anchored_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 |
| udd5/udd5 | Anchored_Exact | vehicle | 5.3645 | 0.0000 | 5.3645 | 99.9773 | 7.8274 |
| udd5/udd5 | Anchored_Exact | other | 5.7476 | 0.0000 | 29.7284 | 6.6513 | 4.5458 |
| udd5/udd5 | TargetContext_Exact | vegetation | 47.6083 | 1.3284 | 93.1707 | 49.3297 | 1.6346 |
| udd5/udd5 | TargetContext_Exact | building | 83.5377 | 0.0506 | 83.7470 | 99.7017 | 85.6280 |
| udd5/udd5 | TargetContext_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2585 |
| udd5/udd5 | TargetContext_Exact | vehicle | 5.9193 | 0.5548 | 5.9194 | 99.9773 | 7.0937 |
| udd5/udd5 | TargetContext_Exact | other | 7.3200 | 1.5724 | 32.5541 | 8.6285 | 5.3853 |
| udd5/udd5 | RawMean_Exact | vegetation | 45.7073 | -0.5726 | 94.2165 | 47.0268 | 1.5409 |
| udd5/udd5 | RawMean_Exact | building | 83.5955 | 0.1084 | 83.8182 | 99.6832 | 85.5394 |
| udd5/udd5 | RawMean_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2576 |
| udd5/udd5 | RawMean_Exact | vehicle | 5.7914 | 0.4269 | 5.7915 | 99.9773 | 7.2504 |
| udd5/udd5 | RawMean_Exact | other | 6.9908 | 1.2432 | 31.0659 | 8.2744 | 5.4117 |
| udd5/udd5 | CapacityMean_Exact | vegetation | 45.7073 | -0.5726 | 94.2165 | 47.0268 | 1.5409 |
| udd5/udd5 | CapacityMean_Exact | building | 83.5955 | 0.1084 | 83.8182 | 99.6832 | 85.5394 |
| udd5/udd5 | CapacityMean_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2576 |
| udd5/udd5 | CapacityMean_Exact | vehicle | 5.7914 | 0.4269 | 5.7915 | 99.9773 | 7.2504 |
| udd5/udd5 | CapacityMean_Exact | other | 6.9908 | 1.2432 | 31.0659 | 8.2744 | 5.4117 |
| udd5/udd5 | SpatialShuffle0_Exact | vegetation | 45.6058 | -0.6741 | 94.2129 | 46.9202 | 1.5375 |
| udd5/udd5 | SpatialShuffle0_Exact | building | 83.6029 | 0.1158 | 83.8248 | 99.6843 | 85.5335 |
| udd5/udd5 | SpatialShuffle0_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2568 |
| udd5/udd5 | SpatialShuffle0_Exact | vehicle | 5.7954 | 0.4309 | 5.7954 | 99.9773 | 7.2454 |
| udd5/udd5 | SpatialShuffle0_Exact | other | 6.9756 | 1.2280 | 30.9348 | 8.2624 | 5.4267 |
| udd5/udd5 | SpatialShuffle1_Exact | vegetation | 45.5240 | -0.7559 | 94.2629 | 46.8213 | 1.5335 |
| udd5/udd5 | SpatialShuffle1_Exact | building | 83.5965 | 0.1094 | 83.8191 | 99.6833 | 85.5385 |
| udd5/udd5 | SpatialShuffle1_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2618 |
| udd5/udd5 | SpatialShuffle1_Exact | vehicle | 5.7909 | 0.4264 | 5.7909 | 99.9773 | 7.2510 |
| udd5/udd5 | SpatialShuffle1_Exact | other | 6.9527 | 1.2051 | 30.8918 | 8.2333 | 5.4152 |
| udd5/udd5 | SpatialShuffle2_Exact | vegetation | 45.8120 | -0.4679 | 94.1781 | 47.1472 | 1.5455 |
| udd5/udd5 | SpatialShuffle2_Exact | building | 83.6086 | 0.1215 | 83.8299 | 99.6853 | 85.5292 |
| udd5/udd5 | SpatialShuffle2_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2528 |
| udd5/udd5 | SpatialShuffle2_Exact | vehicle | 5.7742 | 0.4097 | 5.7743 | 99.9773 | 7.2719 |
| udd5/udd5 | SpatialShuffle2_Exact | other | 6.9890 | 1.2414 | 31.1086 | 8.2687 | 5.4006 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 |
| oem/oem | Geometry | rangeland | 41.5417 | -7.9844 | 54.4153 | 63.7145 | 17.0320 |
| oem/oem | Geometry | developed space | 25.6980 | 1.9311 | 65.5490 | 29.7108 | 8.8828 |
| oem/oem | Geometry | road | 52.3963 | -2.1566 | 60.6188 | 79.4359 | 8.2389 |
| oem/oem | Geometry | tree | 65.2556 | 9.5951 | 84.8180 | 73.8857 | 23.0134 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 |
| oem/oem | Geometry | agriculture land | 71.2323 | -10.4177 | 95.7971 | 73.5302 | 16.2828 |
| oem/oem | Geometry | building | 62.4408 | 15.4121 | 64.4677 | 95.2061 | 17.5919 |
| oem/oem | Anchored_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 |
| oem/oem | Anchored_Exact | rangeland | 49.5261 | 0.0000 | 70.5348 | 62.4456 | 12.8780 |
| oem/oem | Anchored_Exact | developed space | 23.7669 | 0.0000 | 38.9362 | 37.8898 | 19.0710 |
| oem/oem | Anchored_Exact | road | 54.5529 | 0.0000 | 68.0810 | 73.3007 | 6.7693 |
| oem/oem | Anchored_Exact | tree | 55.6605 | 0.0000 | 91.7802 | 58.5807 | 16.8622 |
| oem/oem | Anchored_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | Anchored_Exact | agriculture land | 81.6500 | 0.0000 | 82.0066 | 99.4703 | 25.7312 |
| oem/oem | Anchored_Exact | building | 47.0287 | 0.0000 | 68.3392 | 60.1297 | 10.4811 |
| oem/oem | TargetContext_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.7257 |
| oem/oem | TargetContext_Exact | rangeland | 48.8609 | -0.6652 | 70.5025 | 61.4162 | 12.6715 |
| oem/oem | TargetContext_Exact | developed space | 25.4214 | 1.6545 | 40.4478 | 40.6278 | 19.6849 |
| oem/oem | TargetContext_Exact | road | 54.4536 | -0.0993 | 67.9967 | 73.2190 | 6.7701 |
| oem/oem | TargetContext_Exact | tree | 55.7997 | 0.1392 | 91.5938 | 58.8115 | 16.9631 |
| oem/oem | TargetContext_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | TargetContext_Exact | agriculture land | 81.4126 | -0.2374 | 81.7680 | 99.4689 | 25.8059 |
| oem/oem | TargetContext_Exact | building | 46.6627 | -0.3660 | 68.3329 | 59.5374 | 10.3789 |
| oem/oem | RawMean_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6978 |
| oem/oem | RawMean_Exact | rangeland | 49.0126 | -0.5135 | 70.5154 | 61.6461 | 12.7166 |
| oem/oem | RawMean_Exact | developed space | 25.4688 | 1.7019 | 40.5115 | 40.6846 | 19.6814 |
| oem/oem | RawMean_Exact | road | 54.6129 | 0.0600 | 68.3328 | 73.1185 | 6.7275 |
| oem/oem | RawMean_Exact | tree | 55.7284 | 0.0679 | 91.6267 | 58.7188 | 16.9303 |
| oem/oem | RawMean_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | RawMean_Exact | agriculture land | 81.2982 | -0.3518 | 81.6538 | 99.4671 | 25.8415 |
| oem/oem | RawMean_Exact | building | 46.7614 | -0.2673 | 68.3400 | 59.6928 | 10.4049 |
| oem/oem | CapacityMean_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6978 |
| oem/oem | CapacityMean_Exact | rangeland | 49.0126 | -0.5135 | 70.5154 | 61.6461 | 12.7166 |
| oem/oem | CapacityMean_Exact | developed space | 25.4688 | 1.7019 | 40.5115 | 40.6846 | 19.6814 |
| oem/oem | CapacityMean_Exact | road | 54.6129 | 0.0600 | 68.3328 | 73.1185 | 6.7275 |
| oem/oem | CapacityMean_Exact | tree | 55.7284 | 0.0679 | 91.6267 | 58.7188 | 16.9303 |
| oem/oem | CapacityMean_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | CapacityMean_Exact | agriculture land | 81.2982 | -0.3518 | 81.6538 | 99.4671 | 25.8415 |
| oem/oem | CapacityMean_Exact | building | 46.7614 | -0.2673 | 68.3400 | 59.6928 | 10.4049 |
| oem/oem | SpatialShuffle0_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.7249 |
| oem/oem | SpatialShuffle0_Exact | rangeland | 48.9948 | -0.5313 | 70.5174 | 61.6165 | 12.7101 |
| oem/oem | SpatialShuffle0_Exact | developed space | 25.4062 | 1.6393 | 40.4511 | 40.5856 | 19.6628 |
| oem/oem | SpatialShuffle0_Exact | road | 54.6166 | 0.0637 | 68.3365 | 73.1208 | 6.7274 |
| oem/oem | SpatialShuffle0_Exact | tree | 55.7346 | 0.0741 | 91.6321 | 58.7234 | 16.9306 |
| oem/oem | SpatialShuffle0_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | SpatialShuffle0_Exact | agriculture land | 81.2863 | -0.3637 | 81.6442 | 99.4636 | 25.8437 |
| oem/oem | SpatialShuffle0_Exact | building | 46.7603 | -0.2684 | 68.3544 | 59.6801 | 10.4004 |
| oem/oem | SpatialShuffle1_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.7034 |
| oem/oem | SpatialShuffle1_Exact | rangeland | 48.9921 | -0.5340 | 70.4960 | 61.6286 | 12.7165 |
| oem/oem | SpatialShuffle1_Exact | developed space | 25.4635 | 1.6966 | 40.5156 | 40.6668 | 19.6708 |
| oem/oem | SpatialShuffle1_Exact | road | 54.6381 | 0.0852 | 68.3737 | 73.1169 | 6.7234 |
| oem/oem | SpatialShuffle1_Exact | tree | 55.7304 | 0.0699 | 91.6262 | 58.7212 | 16.9311 |
| oem/oem | SpatialShuffle1_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | SpatialShuffle1_Exact | agriculture land | 81.2699 | -0.3801 | 81.6275 | 99.4638 | 25.8490 |
| oem/oem | SpatialShuffle1_Exact | building | 46.7414 | -0.2873 | 68.3167 | 59.6780 | 10.4058 |
| oem/oem | SpatialShuffle2_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6899 |
| oem/oem | SpatialShuffle2_Exact | rangeland | 49.0222 | -0.5039 | 70.5204 | 61.6576 | 12.7181 |
| oem/oem | SpatialShuffle2_Exact | developed space | 25.4812 | 1.7143 | 40.5020 | 40.7258 | 19.7059 |
| oem/oem | SpatialShuffle2_Exact | road | 54.6363 | 0.0834 | 68.3640 | 73.1247 | 6.7250 |
| oem/oem | SpatialShuffle2_Exact | tree | 55.7142 | 0.0537 | 91.6406 | 58.6973 | 16.9215 |
| oem/oem | SpatialShuffle2_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | SpatialShuffle2_Exact | agriculture land | 81.3116 | -0.3384 | 81.6676 | 99.4668 | 25.8371 |
| oem/oem | SpatialShuffle2_Exact | building | 46.7363 | -0.2924 | 68.3236 | 59.6644 | 10.4024 |
| loveda/P | Geometry | building | 7.6311 | -53.2168 | 7.6311 | 100.0000 | 0.3472 |
| loveda/P | Geometry | road | 63.0522 | 9.2969 | 64.4734 | 96.6222 | 12.4720 |
| loveda/P | Geometry | water | 63.2131 | 0.8040 | 68.9718 | 88.3328 | 18.8451 |
| loveda/P | Geometry | barren | 1.9144 | -2.4071 | 34.2731 | 1.9874 | 0.6429 |
| loveda/P | Geometry | tree | 72.0056 | 32.0503 | 88.8915 | 79.1256 | 14.6560 |
| loveda/P | Geometry | farm | 89.4231 | 6.4729 | 91.1653 | 97.9076 | 53.0368 |
| loveda/P | Anchored_Exact | building | 60.8479 | 0.0000 | 72.1893 | 79.4788 | 0.0292 |
| loveda/P | Anchored_Exact | road | 53.7553 | 0.0000 | 54.7280 | 96.7995 | 14.7198 |
| loveda/P | Anchored_Exact | water | 62.4091 | 0.0000 | 68.1253 | 88.1486 | 19.0394 |
| loveda/P | Anchored_Exact | barren | 4.3215 | 0.0000 | 89.7378 | 4.3430 | 0.5366 |
| loveda/P | Anchored_Exact | tree | 39.9553 | 0.0000 | 99.3680 | 40.0571 | 6.6373 |
| loveda/P | Anchored_Exact | farm | 82.9502 | 0.0000 | 83.2671 | 99.5433 | 59.0377 |
| loveda/P | TargetContext_Exact | building | 61.7571 | 0.9092 | 74.9216 | 77.8502 | 0.0275 |
| loveda/P | TargetContext_Exact | road | 53.1125 | -0.6428 | 54.0606 | 96.8037 | 14.9022 |
| loveda/P | TargetContext_Exact | water | 62.1478 | -0.2613 | 68.2053 | 87.4963 | 18.8764 |
| loveda/P | TargetContext_Exact | barren | 3.9695 | -0.3520 | 90.3493 | 3.9864 | 0.4892 |
| loveda/P | TargetContext_Exact | tree | 40.1305 | 0.1752 | 99.3515 | 40.2359 | 6.6680 |
| loveda/P | TargetContext_Exact | farm | 82.9743 | 0.0241 | 83.2810 | 99.5582 | 59.0366 |
| loveda/P | RawMean_Exact | building | 57.3643 | -3.4836 | 73.5099 | 72.3127 | 0.0261 |
| loveda/P | RawMean_Exact | road | 53.1271 | -0.6282 | 54.0679 | 96.8286 | 14.9040 |
| loveda/P | RawMean_Exact | water | 62.5134 | 0.1043 | 68.5613 | 87.6342 | 18.8080 |
| loveda/P | RawMean_Exact | barren | 4.0530 | -0.2685 | 90.0172 | 4.0713 | 0.5015 |
| loveda/P | RawMean_Exact | tree | 39.3918 | -0.5635 | 99.4493 | 39.4779 | 6.5360 |
| loveda/P | RawMean_Exact | farm | 82.7123 | -0.2379 | 83.0169 | 99.5584 | 59.2245 |
| loveda/P | CapacityMean_Exact | building | 57.3643 | -3.4836 | 73.5099 | 72.3127 | 0.0261 |
| loveda/P | CapacityMean_Exact | road | 53.1271 | -0.6282 | 54.0679 | 96.8286 | 14.9040 |
| loveda/P | CapacityMean_Exact | water | 62.5134 | 0.1043 | 68.5613 | 87.6342 | 18.8080 |
| loveda/P | CapacityMean_Exact | barren | 4.0530 | -0.2685 | 90.0172 | 4.0713 | 0.5015 |
| loveda/P | CapacityMean_Exact | tree | 39.3918 | -0.5635 | 99.4493 | 39.4779 | 6.5360 |
| loveda/P | CapacityMean_Exact | farm | 82.7123 | -0.2379 | 83.0169 | 99.5584 | 59.2245 |
| loveda/P | SpatialShuffle0_Exact | building | 58.0977 | -2.7502 | 73.3766 | 73.6156 | 0.0266 |
| loveda/P | SpatialShuffle0_Exact | road | 53.0933 | -0.6620 | 54.0339 | 96.8254 | 14.9129 |
| loveda/P | SpatialShuffle0_Exact | water | 62.4609 | 0.0518 | 68.5208 | 87.5972 | 18.8112 |
| loveda/P | SpatialShuffle0_Exact | barren | 4.1356 | -0.1859 | 90.0759 | 4.1546 | 0.5114 |
| loveda/P | SpatialShuffle0_Exact | tree | 39.3733 | -0.5820 | 99.4478 | 39.4595 | 6.5330 |
| loveda/P | SpatialShuffle0_Exact | farm | 82.7359 | -0.2143 | 83.0423 | 99.5559 | 59.2050 |
| loveda/P | SpatialShuffle1_Exact | building | 52.1411 | -8.7068 | 69.6970 | 67.4267 | 0.0256 |
| loveda/P | SpatialShuffle1_Exact | road | 53.1344 | -0.6209 | 54.0784 | 96.8192 | 14.8997 |
| loveda/P | SpatialShuffle1_Exact | water | 62.5123 | 0.1032 | 68.5470 | 87.6553 | 18.8164 |
| loveda/P | SpatialShuffle1_Exact | barren | 4.0038 | -0.3177 | 89.7360 | 4.0222 | 0.4970 |
| loveda/P | SpatialShuffle1_Exact | tree | 39.3949 | -0.5604 | 99.4428 | 39.4821 | 6.5371 |
| loveda/P | SpatialShuffle1_Exact | farm | 82.7133 | -0.2369 | 83.0177 | 99.5587 | 59.2242 |
| loveda/P | SpatialShuffle2_Exact | building | 57.6531 | -3.1948 | 72.6688 | 73.6156 | 0.0268 |
| loveda/P | SpatialShuffle2_Exact | road | 53.1452 | -0.6101 | 54.0893 | 96.8203 | 14.8968 |
| loveda/P | SpatialShuffle2_Exact | water | 62.4722 | 0.0631 | 68.5289 | 87.6060 | 18.8108 |
| loveda/P | SpatialShuffle2_Exact | barren | 3.9564 | -0.3651 | 89.9401 | 3.9740 | 0.4899 |
| loveda/P | SpatialShuffle2_Exact | tree | 39.5177 | -0.4376 | 99.4367 | 39.6063 | 6.5581 |
| loveda/P | SpatialShuffle2_Exact | farm | 82.7247 | -0.2255 | 83.0282 | 99.5601 | 59.2176 |
| loveda/D | Geometry | background | 39.0474 | 31.7250 | 75.8250 | 44.5998 | 26.3232 |
| loveda/D | Geometry | building | 4.8538 | -26.4283 | 4.8538 | 100.0000 | 0.3016 |
| loveda/D | Geometry | road | 45.0691 | 0.9421 | 45.8485 | 96.3650 | 9.6637 |
| loveda/D | Geometry | water | 53.8049 | -0.6330 | 58.0149 | 88.1157 | 12.3473 |
| loveda/D | Geometry | barren | 0.0293 | -3.2121 | 0.8665 | 0.0304 | 0.2146 |
| loveda/D | Geometry | tree | 37.0607 | 1.1933 | 42.3596 | 74.7642 | 16.0550 |
| loveda/D | Geometry | farm | 57.2802 | 18.4061 | 64.7327 | 83.2647 | 35.0945 |
| loveda/D | Anchored_Exact | background | 7.3224 | 0.0000 | 75.1732 | 7.5039 | 4.4672 |
| loveda/D | Anchored_Exact | building | 31.2821 | 0.0000 | 34.0307 | 79.4788 | 0.0342 |
| loveda/D | Anchored_Exact | road | 44.1270 | 0.0000 | 44.7837 | 96.7840 | 9.9365 |
| loveda/D | Anchored_Exact | water | 54.4379 | 0.0000 | 58.7421 | 88.1369 | 12.1974 |
| loveda/D | Anchored_Exact | barren | 3.2414 | 0.0000 | 82.3610 | 3.2640 | 0.2428 |
| loveda/D | Anchored_Exact | tree | 35.8674 | 0.0000 | 97.1427 | 36.2498 | 3.3944 |
| loveda/D | Anchored_Exact | farm | 38.8741 | 0.0000 | 38.9455 | 99.5311 | 69.7275 |
| loveda/D | TargetContext_Exact | background | 10.6467 | 3.3243 | 83.3341 | 10.8783 | 5.8419 |
| loveda/D | TargetContext_Exact | building | 32.3848 | 1.1027 | 35.6716 | 77.8502 | 0.0319 |
| loveda/D | TargetContext_Exact | road | 43.4899 | -0.6371 | 44.1278 | 96.7829 | 10.0841 |
| loveda/D | TargetContext_Exact | water | 54.1036 | -0.3343 | 58.6444 | 87.4805 | 12.1267 |
| loveda/D | TargetContext_Exact | barren | 2.9271 | -0.3143 | 81.6965 | 2.9464 | 0.2209 |
| loveda/D | TargetContext_Exact | tree | 36.8241 | 0.9567 | 97.1307 | 37.2291 | 3.4865 |
| loveda/D | TargetContext_Exact | farm | 39.7495 | 0.8754 | 39.8210 | 99.5507 | 68.2078 |
| loveda/D | RawMean_Exact | background | 9.5612 | 2.2388 | 82.3646 | 9.7610 | 5.3036 |
| loveda/D | RawMean_Exact | building | 30.4843 | -0.7978 | 35.1396 | 69.7068 | 0.0290 |
| loveda/D | RawMean_Exact | road | 43.3209 | -0.8061 | 43.9497 | 96.8026 | 10.1270 |
| loveda/D | RawMean_Exact | water | 54.3444 | -0.0935 | 58.8650 | 87.6183 | 12.1003 |
| loveda/D | RawMean_Exact | barren | 3.0201 | -0.2213 | 82.2798 | 3.0398 | 0.2263 |
| loveda/D | RawMean_Exact | tree | 36.4358 | 0.5684 | 97.3540 | 36.8003 | 3.4385 |
| loveda/D | RawMean_Exact | farm | 39.4211 | 0.5470 | 39.4917 | 99.5487 | 68.7752 |
| loveda/D | CapacityMean_Exact | background | 9.5612 | 2.2388 | 82.3646 | 9.7610 | 5.3036 |
| loveda/D | CapacityMean_Exact | building | 30.4843 | -0.7978 | 35.1396 | 69.7068 | 0.0290 |
| loveda/D | CapacityMean_Exact | road | 43.3209 | -0.8061 | 43.9497 | 96.8026 | 10.1270 |
| loveda/D | CapacityMean_Exact | water | 54.3444 | -0.0935 | 58.8650 | 87.6183 | 12.1003 |
| loveda/D | CapacityMean_Exact | barren | 3.0201 | -0.2213 | 82.2798 | 3.0398 | 0.2263 |
| loveda/D | CapacityMean_Exact | tree | 36.4358 | 0.5684 | 97.3540 | 36.8003 | 3.4385 |
| loveda/D | CapacityMean_Exact | farm | 39.4211 | 0.5470 | 39.4917 | 99.5487 | 68.7752 |
| loveda/D | SpatialShuffle0_Exact | background | 9.7100 | 2.3876 | 82.4323 | 9.9152 | 5.3830 |
| loveda/D | SpatialShuffle0_Exact | building | 30.4161 | -0.8660 | 35.2159 | 69.0554 | 0.0287 |
| loveda/D | SpatialShuffle0_Exact | road | 43.3077 | -0.8193 | 43.9359 | 96.8037 | 10.1303 |
| loveda/D | SpatialShuffle0_Exact | water | 54.3257 | -0.1122 | 58.8524 | 87.5978 | 12.1001 |
| loveda/D | SpatialShuffle0_Exact | barren | 3.0617 | -0.1797 | 82.4105 | 3.0819 | 0.2291 |
| loveda/D | SpatialShuffle0_Exact | tree | 36.3004 | 0.4330 | 97.3780 | 36.6587 | 3.4244 |
| loveda/D | SpatialShuffle0_Exact | farm | 39.4617 | 0.5876 | 39.5324 | 99.5487 | 68.7045 |
| loveda/D | SpatialShuffle1_Exact | background | 9.5312 | 2.2088 | 82.5295 | 9.7274 | 5.2748 |
| loveda/D | SpatialShuffle1_Exact | building | 29.9301 | -1.3520 | 34.4051 | 69.7068 | 0.0297 |
| loveda/D | SpatialShuffle1_Exact | road | 43.3108 | -0.8162 | 43.9394 | 96.8026 | 10.1294 |
| loveda/D | SpatialShuffle1_Exact | water | 54.3693 | -0.0686 | 58.8772 | 87.6559 | 12.1030 |
| loveda/D | SpatialShuffle1_Exact | barren | 2.9752 | -0.2662 | 82.0256 | 2.9947 | 0.2236 |
| loveda/D | SpatialShuffle1_Exact | tree | 36.3686 | 0.5012 | 97.3252 | 36.7358 | 3.4335 |
| loveda/D | SpatialShuffle1_Exact | farm | 39.4042 | 0.5301 | 39.4745 | 99.5500 | 68.8060 |
| loveda/D | SpatialShuffle2_Exact | background | 9.5252 | 2.2028 | 82.0165 | 9.7284 | 5.3083 |
| loveda/D | SpatialShuffle2_Exact | building | 29.9013 | -1.3808 | 34.5277 | 69.0554 | 0.0293 |
| loveda/D | SpatialShuffle2_Exact | road | 43.2977 | -0.8293 | 43.9254 | 96.8047 | 10.1328 |
| loveda/D | SpatialShuffle2_Exact | water | 54.3609 | -0.0770 | 58.8965 | 87.5914 | 12.0901 |
| loveda/D | SpatialShuffle2_Exact | barren | 3.0697 | -0.1717 | 82.6187 | 3.0897 | 0.2291 |
| loveda/D | SpatialShuffle2_Exact | tree | 36.1973 | 0.3299 | 97.3721 | 36.5544 | 3.4149 |
| loveda/D | SpatialShuffle2_Exact | farm | 39.4097 | 0.5356 | 39.4802 | 99.5491 | 68.7955 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -15.2111 | 77.7719 | 47.8510 | 16.8072 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 7.0164 | 73.5287 | 98.5836 | 27.6605 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 9.2961 | 92.4475 | 55.4111 | 17.7225 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.3793 | 82.7688 | 80.0845 | 19.9693 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -14.7902 | 10.3220 | 99.7520 | 17.8406 |
| vaihingen/vaihingen | Anchored_Exact | impervious surface | 57.3052 | 0.0000 | 73.8225 | 71.9196 | 26.6125 |
| vaihingen/vaihingen | Anchored_Exact | building | 65.7437 | 0.0000 | 65.8760 | 99.6954 | 31.2219 |
| vaihingen/vaihingen | Anchored_Exact | low vegetation | 43.7153 | 0.0000 | 96.1772 | 44.4883 | 13.6772 |
| vaihingen/vaihingen | Anchored_Exact | tree | 67.2612 | 0.0000 | 79.0950 | 81.8036 | 21.3454 |
| vaihingen/vaihingen | Anchored_Exact | car | 25.1095 | 0.0000 | 25.2570 | 97.7270 | 7.1430 |
| vaihingen/vaihingen | TargetContext_Exact | impervious surface | 57.1078 | -0.1974 | 72.9878 | 72.4122 | 27.1012 |
| vaihingen/vaihingen | TargetContext_Exact | building | 65.8598 | 0.1161 | 65.9960 | 99.6877 | 31.1628 |
| vaihingen/vaihingen | TargetContext_Exact | low vegetation | 42.7248 | -0.9905 | 96.3505 | 43.4276 | 13.3271 |
| vaihingen/vaihingen | TargetContext_Exact | tree | 67.2409 | -0.0203 | 78.8506 | 82.0365 | 21.4725 |
| vaihingen/vaihingen | TargetContext_Exact | car | 25.8452 | 0.7357 | 26.0031 | 97.7037 | 6.9364 |
| vaihingen/vaihingen | RawMean_Exact | impervious surface | 57.1817 | -0.1235 | 73.0204 | 72.4989 | 27.1216 |
| vaihingen/vaihingen | RawMean_Exact | building | 65.7799 | 0.0362 | 65.9135 | 99.6928 | 31.2034 |
| vaihingen/vaihingen | RawMean_Exact | low vegetation | 42.8597 | -0.8556 | 96.3714 | 43.5628 | 13.3657 |
| vaihingen/vaihingen | RawMean_Exact | tree | 67.2377 | -0.0235 | 79.0670 | 81.7988 | 21.3517 |
| vaihingen/vaihingen | RawMean_Exact | car | 25.7682 | 0.6587 | 25.9249 | 97.7089 | 6.9577 |
| vaihingen/vaihingen | CapacityMean_Exact | impervious surface | 57.1817 | -0.1235 | 73.0204 | 72.4989 | 27.1216 |
| vaihingen/vaihingen | CapacityMean_Exact | building | 65.7799 | 0.0362 | 65.9135 | 99.6928 | 31.2034 |
| vaihingen/vaihingen | CapacityMean_Exact | low vegetation | 42.8597 | -0.8556 | 96.3714 | 43.5628 | 13.3657 |
| vaihingen/vaihingen | CapacityMean_Exact | tree | 67.2377 | -0.0235 | 79.0670 | 81.7988 | 21.3517 |
| vaihingen/vaihingen | CapacityMean_Exact | car | 25.7682 | 0.6587 | 25.9249 | 97.7089 | 6.9577 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | impervious surface | 57.1355 | -0.1697 | 72.9899 | 72.4548 | 27.1164 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | building | 65.7595 | 0.0158 | 65.8929 | 99.6931 | 31.2132 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | low vegetation | 42.8239 | -0.8914 | 96.3678 | 43.5265 | 13.3550 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | tree | 67.2242 | -0.0370 | 79.0650 | 81.7810 | 21.3476 |
| vaihingen/vaihingen | SpatialShuffle0_Exact | car | 25.7353 | 0.6258 | 25.8907 | 97.7218 | 6.9678 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | impervious surface | 57.2053 | -0.0999 | 73.0957 | 72.4628 | 27.0802 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | building | 65.7639 | 0.0202 | 65.8973 | 99.6931 | 31.2111 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | low vegetation | 42.9074 | -0.8079 | 96.3655 | 43.6132 | 13.3820 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | tree | 67.2357 | -0.0255 | 79.0690 | 81.7937 | 21.3498 |
| vaihingen/vaihingen | SpatialShuffle1_Exact | car | 25.7001 | 0.5906 | 25.8553 | 97.7166 | 6.9770 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | impervious surface | 57.1409 | -0.1643 | 72.9818 | 72.4713 | 27.1256 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | building | 65.7690 | 0.0253 | 65.9023 | 99.6935 | 31.2089 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | low vegetation | 42.8265 | -0.8888 | 96.3659 | 43.5295 | 13.3562 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | tree | 67.2167 | -0.0445 | 79.0567 | 81.7789 | 21.3493 |
| vaihingen/vaihingen | SpatialShuffle2_Exact | car | 25.7632 | 0.6537 | 25.9191 | 97.7192 | 6.9600 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -4.8387 | 96.6505 | 85.0437 | 59.6226 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.5110 | 35.0436 | 99.2919 | 4.2927 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -0.2388 | 86.4992 | 89.2841 | 22.0960 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9543 | 93.6635 | 100.0000 | 8.8309 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -11.4629 | 15.6288 | 77.0019 | 5.1578 |
| landcoverai/landcoverai | Anchored_Exact | background | 87.4477 | 0.0000 | 93.8549 | 92.7587 | 66.9686 |
| landcoverai/landcoverai | Anchored_Exact | building | 44.4672 | 0.0000 | 44.8124 | 98.2973 | 3.3233 |
| landcoverai/landcoverai | Anchored_Exact | woodland | 78.6026 | 0.0000 | 94.9565 | 82.0272 | 18.4920 |
| landcoverai/landcoverai | Anchored_Exact | water | 97.6178 | 0.0000 | 97.6178 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | Anchored_Exact | road | 26.3947 | 0.0000 | 28.8528 | 75.5990 | 2.7429 |
| landcoverai/landcoverai | TargetContext_Exact | background | 87.5369 | 0.0892 | 94.1455 | 92.5764 | 66.6306 |
| landcoverai/landcoverai | TargetContext_Exact | building | 44.1597 | -0.3075 | 44.4956 | 98.3193 | 3.3477 |
| landcoverai/landcoverai | TargetContext_Exact | woodland | 79.4007 | 0.7981 | 94.7969 | 83.0186 | 18.7470 |
| landcoverai/landcoverai | TargetContext_Exact | water | 97.6244 | 0.0066 | 97.6244 | 100.0000 | 8.4726 |
| landcoverai/landcoverai | TargetContext_Exact | road | 25.9240 | -0.4707 | 28.2785 | 75.6901 | 2.8020 |
| landcoverai/landcoverai | RawMean_Exact | background | 87.5247 | 0.0770 | 94.0569 | 92.6485 | 66.7454 |
| landcoverai/landcoverai | RawMean_Exact | building | 44.3051 | -0.1621 | 44.6452 | 98.3099 | 3.3362 |
| landcoverai/landcoverai | RawMean_Exact | woodland | 79.1811 | 0.5785 | 94.8752 | 82.7190 | 18.6640 |
| landcoverai/landcoverai | RawMean_Exact | water | 97.6667 | 0.0489 | 97.6667 | 100.0000 | 8.4689 |
| landcoverai/landcoverai | RawMean_Exact | road | 26.0306 | -0.3641 | 28.4164 | 75.6126 | 2.7855 |
| landcoverai/landcoverai | CapacityMean_Exact | background | 87.5247 | 0.0770 | 94.0569 | 92.6485 | 66.7454 |
| landcoverai/landcoverai | CapacityMean_Exact | building | 44.3051 | -0.1621 | 44.6452 | 98.3099 | 3.3362 |
| landcoverai/landcoverai | CapacityMean_Exact | woodland | 79.1811 | 0.5785 | 94.8752 | 82.7190 | 18.6640 |
| landcoverai/landcoverai | CapacityMean_Exact | water | 97.6667 | 0.0489 | 97.6667 | 100.0000 | 8.4689 |
| landcoverai/landcoverai | CapacityMean_Exact | road | 26.0306 | -0.3641 | 28.4164 | 75.6126 | 2.7855 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | background | 87.5280 | 0.0803 | 94.0560 | 92.6531 | 66.7493 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | building | 44.3194 | -0.1478 | 44.6629 | 98.2941 | 3.3343 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | woodland | 79.1705 | 0.5679 | 94.8653 | 82.7150 | 18.6650 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | water | 97.6629 | 0.0451 | 97.6629 | 100.0000 | 8.4692 |
| landcoverai/landcoverai | SpatialShuffle0_Exact | road | 26.0680 | -0.3267 | 28.4583 | 75.6309 | 2.7821 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | background | 87.5098 | 0.0621 | 94.0489 | 92.6395 | 66.7445 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | building | 44.2877 | -0.1795 | 44.6268 | 98.3130 | 3.3377 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | woodland | 79.1504 | 0.5478 | 94.8622 | 82.6954 | 18.6612 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | water | 97.6651 | 0.0473 | 97.6651 | 100.0000 | 8.4691 |
| landcoverai/landcoverai | SpatialShuffle1_Exact | road | 26.0135 | -0.3812 | 28.3960 | 75.6126 | 2.7875 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | background | 87.5173 | 0.0696 | 94.0449 | 92.6519 | 66.7563 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | building | 44.3465 | -0.1207 | 44.6878 | 98.3067 | 3.3329 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | woodland | 79.1417 | 0.5391 | 94.8726 | 82.6780 | 18.6553 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | water | 97.6695 | 0.0517 | 97.6695 | 100.0000 | 8.4687 |
| landcoverai/landcoverai | SpatialShuffle2_Exact | road | 26.0236 | -0.3711 | 28.4067 | 75.6218 | 2.7868 |
| flair1/flair1 | Geometry | building | 49.6979 | -7.0776 | 50.7258 | 96.0825 | 13.5991 |
| flair1/flair1 | Geometry | pervious surface | 57.0621 | 9.7868 | 92.1299 | 59.9861 | 11.1728 |
| flair1/flair1 | Geometry | impervious surface | 52.6509 | 0.6651 | 62.6245 | 76.7765 | 20.0846 |
| flair1/flair1 | Geometry | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4411 |
| flair1/flair1 | Geometry | water | 73.2328 | 11.7763 | 77.1432 | 93.5263 | 5.3781 |
| flair1/flair1 | Geometry | coniferous | 43.0233 | -0.5049 | 61.7114 | 58.6897 | 0.5346 |
| flair1/flair1 | Geometry | deciduous | 54.0229 | -5.5008 | 78.9723 | 63.0995 | 13.6004 |
| flair1/flair1 | Geometry | brushwood | 18.6136 | 6.2849 | 23.4212 | 47.5559 | 9.9866 |
| flair1/flair1 | Geometry | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | Geometry | herbaceous vegetation | 60.2329 | 2.7604 | 94.3169 | 62.5013 | 21.2289 |
| flair1/flair1 | Geometry | agricultural land | 18.7339 | 5.7793 | 21.6468 | 58.1977 | 0.8198 |
| flair1/flair1 | Geometry | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1540 |
| flair1/flair1 | Anchored_Exact | building | 56.7755 | 0.0000 | 58.0396 | 96.3057 | 11.9131 |
| flair1/flair1 | Anchored_Exact | pervious surface | 47.2753 | 0.0000 | 91.6998 | 49.3887 | 9.2421 |
| flair1/flair1 | Anchored_Exact | impervious surface | 51.9858 | 0.0000 | 59.7781 | 79.9522 | 21.9112 |
| flair1/flair1 | Anchored_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 |
| flair1/flair1 | Anchored_Exact | water | 61.4565 | 0.0000 | 63.2502 | 95.5889 | 6.7041 |
| flair1/flair1 | Anchored_Exact | coniferous | 43.5282 | 0.0000 | 65.7321 | 56.3052 | 0.4815 |
| flair1/flair1 | Anchored_Exact | deciduous | 59.5237 | 0.0000 | 78.8015 | 70.8722 | 15.3089 |
| flair1/flair1 | Anchored_Exact | brushwood | 12.3287 | 0.0000 | 26.2213 | 18.8771 | 3.5408 |
| flair1/flair1 | Anchored_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | Anchored_Exact | herbaceous vegetation | 57.4725 | 0.0000 | 90.8252 | 61.0148 | 21.5207 |
| flair1/flair1 | Anchored_Exact | agricultural land | 12.9546 | 0.0000 | 13.9145 | 65.2534 | 1.4300 |
| flair1/flair1 | Anchored_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 |
| flair1/flair1 | TargetContext_Exact | building | 56.4909 | -0.2846 | 57.6383 | 96.5961 | 12.0322 |
| flair1/flair1 | TargetContext_Exact | pervious surface | 47.0237 | -0.2516 | 92.1297 | 48.9917 | 9.1250 |
| flair1/flair1 | TargetContext_Exact | impervious surface | 51.6353 | -0.3505 | 59.2782 | 80.0192 | 22.1145 |
| flair1/flair1 | TargetContext_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3173 |
| flair1/flair1 | TargetContext_Exact | water | 62.0656 | 0.6091 | 63.8889 | 95.6039 | 6.6381 |
| flair1/flair1 | TargetContext_Exact | coniferous | 43.4192 | -0.1090 | 65.2673 | 56.4664 | 0.4863 |
| flair1/flair1 | TargetContext_Exact | deciduous | 60.0711 | 0.5474 | 78.3090 | 72.0616 | 15.6637 |
| flair1/flair1 | TargetContext_Exact | brushwood | 12.1635 | -0.1652 | 27.7179 | 17.8141 | 3.1610 |
| flair1/flair1 | TargetContext_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | TargetContext_Exact | herbaceous vegetation | 56.1852 | -1.2873 | 91.0616 | 59.4646 | 20.9195 |
| flair1/flair1 | TargetContext_Exact | agricultural land | 12.5455 | -0.4091 | 13.3752 | 66.9118 | 1.5254 |
| flair1/flair1 | TargetContext_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0154 |
| flair1/flair1 | RawMean_Exact | building | 56.5232 | -0.2523 | 57.7730 | 96.3137 | 11.9690 |
| flair1/flair1 | RawMean_Exact | pervious surface | 46.9456 | -0.3297 | 92.0049 | 48.9422 | 9.1282 |
| flair1/flair1 | RawMean_Exact | impervious surface | 51.6374 | -0.3484 | 59.1992 | 80.1688 | 22.1854 |
| flair1/flair1 | RawMean_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2708 |
| flair1/flair1 | RawMean_Exact | water | 61.8174 | 0.3609 | 63.6770 | 95.4889 | 6.6522 |
| flair1/flair1 | RawMean_Exact | coniferous | 43.1871 | -0.3411 | 64.9349 | 56.3221 | 0.4876 |
| flair1/flair1 | RawMean_Exact | deciduous | 59.9776 | 0.4539 | 78.2696 | 71.9604 | 15.6496 |
| flair1/flair1 | RawMean_Exact | brushwood | 12.2418 | -0.0869 | 27.8681 | 17.9198 | 3.1626 |
| flair1/flair1 | RawMean_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0012 |
| flair1/flair1 | RawMean_Exact | herbaceous vegetation | 56.6531 | -0.8194 | 91.1747 | 59.9401 | 21.0606 |
| flair1/flair1 | RawMean_Exact | agricultural land | 12.6977 | -0.2569 | 13.5413 | 67.0839 | 1.5106 |
| flair1/flair1 | RawMean_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9223 |
| flair1/flair1 | CapacityMean_Exact | building | 56.5232 | -0.2523 | 57.7730 | 96.3137 | 11.9690 |
| flair1/flair1 | CapacityMean_Exact | pervious surface | 46.9456 | -0.3297 | 92.0049 | 48.9422 | 9.1282 |
| flair1/flair1 | CapacityMean_Exact | impervious surface | 51.6374 | -0.3484 | 59.1992 | 80.1688 | 22.1854 |
| flair1/flair1 | CapacityMean_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2708 |
| flair1/flair1 | CapacityMean_Exact | water | 61.8174 | 0.3609 | 63.6770 | 95.4889 | 6.6522 |
| flair1/flair1 | CapacityMean_Exact | coniferous | 43.1871 | -0.3411 | 64.9349 | 56.3221 | 0.4876 |
| flair1/flair1 | CapacityMean_Exact | deciduous | 59.9776 | 0.4539 | 78.2696 | 71.9604 | 15.6496 |
| flair1/flair1 | CapacityMean_Exact | brushwood | 12.2418 | -0.0869 | 27.8681 | 17.9198 | 3.1626 |
| flair1/flair1 | CapacityMean_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0012 |
| flair1/flair1 | CapacityMean_Exact | herbaceous vegetation | 56.6531 | -0.8194 | 91.1747 | 59.9401 | 21.0606 |
| flair1/flair1 | CapacityMean_Exact | agricultural land | 12.6977 | -0.2569 | 13.5413 | 67.0839 | 1.5106 |
| flair1/flair1 | CapacityMean_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9223 |
| flair1/flair1 | SpatialShuffle0_Exact | building | 56.5205 | -0.2550 | 57.7721 | 96.3084 | 11.9685 |
| flair1/flair1 | SpatialShuffle0_Exact | pervious surface | 46.9232 | -0.3521 | 92.0050 | 48.9178 | 9.1236 |
| flair1/flair1 | SpatialShuffle0_Exact | impervious surface | 51.6178 | -0.3680 | 59.1715 | 80.1723 | 22.1968 |
| flair1/flair1 | SpatialShuffle0_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2630 |
| flair1/flair1 | SpatialShuffle0_Exact | water | 61.8548 | 0.3983 | 63.7129 | 95.4975 | 6.6490 |
| flair1/flair1 | SpatialShuffle0_Exact | coniferous | 43.3299 | -0.1983 | 65.2353 | 56.3391 | 0.4855 |
| flair1/flair1 | SpatialShuffle0_Exact | deciduous | 60.0349 | 0.5112 | 78.2952 | 72.0212 | 15.6577 |
| flair1/flair1 | SpatialShuffle0_Exact | brushwood | 12.3028 | -0.0259 | 27.9968 | 17.9974 | 3.1617 |
| flair1/flair1 | SpatialShuffle0_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| flair1/flair1 | SpatialShuffle0_Exact | herbaceous vegetation | 56.5754 | -0.8971 | 91.1566 | 59.8609 | 21.0369 |
| flair1/flair1 | SpatialShuffle0_Exact | agricultural land | 12.7715 | -0.1831 | 13.6112 | 67.4280 | 1.5105 |
| flair1/flair1 | SpatialShuffle0_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9457 |
| flair1/flair1 | SpatialShuffle1_Exact | building | 56.5432 | -0.2323 | 57.7862 | 96.3350 | 11.9689 |
| flair1/flair1 | SpatialShuffle1_Exact | pervious surface | 46.9148 | -0.3605 | 92.0144 | 48.9061 | 9.1205 |
| flair1/flair1 | SpatialShuffle1_Exact | impervious surface | 51.6427 | -0.3431 | 59.1963 | 80.1869 | 22.1915 |
| flair1/flair1 | SpatialShuffle1_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2699 |
| flair1/flair1 | SpatialShuffle1_Exact | water | 61.8014 | 0.3449 | 63.6587 | 95.4921 | 6.6543 |
| flair1/flair1 | SpatialShuffle1_Exact | coniferous | 43.1629 | -0.3653 | 64.9027 | 56.3052 | 0.4877 |
| flair1/flair1 | SpatialShuffle1_Exact | deciduous | 59.9610 | 0.4373 | 78.2908 | 71.9186 | 15.6362 |
| flair1/flair1 | SpatialShuffle1_Exact | brushwood | 12.2982 | -0.0305 | 27.9354 | 18.0129 | 3.1714 |
| flair1/flair1 | SpatialShuffle1_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0014 |
| flair1/flair1 | SpatialShuffle1_Exact | herbaceous vegetation | 56.6839 | -0.7886 | 91.1992 | 59.9639 | 21.0633 |
| flair1/flair1 | SpatialShuffle1_Exact | agricultural land | 12.7082 | -0.2464 | 13.5344 | 67.5532 | 1.5219 |
| flair1/flair1 | SpatialShuffle1_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9129 |
| flair1/flair1 | SpatialShuffle2_Exact | building | 56.5041 | -0.2714 | 57.7657 | 96.2785 | 11.9662 |
| flair1/flair1 | SpatialShuffle2_Exact | pervious surface | 46.8781 | -0.3972 | 91.9744 | 48.8775 | 9.1191 |
| flair1/flair1 | SpatialShuffle2_Exact | impervious surface | 51.6191 | -0.3667 | 59.1857 | 80.1493 | 22.1851 |
| flair1/flair1 | SpatialShuffle2_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2853 |
| flair1/flair1 | SpatialShuffle2_Exact | water | 61.7654 | 0.3089 | 63.6180 | 95.4975 | 6.6589 |
| flair1/flair1 | SpatialShuffle2_Exact | coniferous | 43.1947 | -0.3335 | 64.9633 | 56.3136 | 0.4873 |
| flair1/flair1 | SpatialShuffle2_Exact | deciduous | 59.9621 | 0.4384 | 78.2640 | 71.9428 | 15.6468 |
| flair1/flair1 | SpatialShuffle2_Exact | brushwood | 12.1589 | -0.1698 | 27.7058 | 17.8092 | 3.1615 |
| flair1/flair1 | SpatialShuffle2_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | SpatialShuffle2_Exact | herbaceous vegetation | 56.6076 | -0.8649 | 91.1729 | 59.8899 | 21.0434 |
| flair1/flair1 | SpatialShuffle2_Exact | agricultural land | 12.6727 | -0.2819 | 13.5098 | 67.1621 | 1.5159 |
| flair1/flair1 | SpatialShuffle2_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9289 |

## Cached Diagnostic Cost

Suite wall: 16.0097s. Nine mathematical tests and one mask-free cached smoke pass. No new encoder forwards. Timing/memory below belong to cached reconstruction/evaluation, NOT deployment inference. The previous source cost of64-150.5 extra forwards/window remains unresolved.

| Dataset | Worker seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 6.3904 | 34.5898 |
| potsdam | 4.1988 | 32.5234 |
| udd5 | 4.4221 | 30.4570 |
| oem | 5.1065 | 36.6562 |
| loveda | 7.6634 | 35.2930 |
| vaihingen | 3.6107 | 30.4570 |
| landcoverai | 3.5072 | 30.4570 |
| flair1 | 6.6469 | 44.9219 |

All64 unique sample keys, actual vocabulary SHA/groups/classes, three retained per-image confusion controls, confusion sums and transition endpoints are verified. Frozen source caches remain remote, merged/per-image results local.
