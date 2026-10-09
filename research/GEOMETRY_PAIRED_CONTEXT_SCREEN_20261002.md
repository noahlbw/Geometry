# Geometry-supported paired contextual readout: verified screen

Fixed96 complete-image development evaluation: UDD5 full40; seven other domains8 each. These are not full eight-dataset results. Frozen all20 aliases and weights; labels evaluation-only. Corrected IRRG Vaihingen. VIP source explicitly borrowed; all domains informed development.

Promotion passed: `False`.

The candidate improves seven of eight primary domain entries versus Geometry,
but most gains are small; UDD5 loses0.100498pp. Equal-domain improvement is
0.130781pp versus Geometry, while it trails Anchored_VIP by2.097430pp. It exceeds
the same-information half-context control by only0.025356pp and shuffled
writeback by0.046594pp. These point estimates do not establish a strong coupling
mechanism or an independent cross-domain result.

The paired difference fails to retain the useful absolute semantic observation:
on VDD water IoU is35.4850 versus Anchored_VIP92.4595; Potsdam car11.8530 versus
24.2128. Full40 UDD5 road remains44.6720 versus original Geometry45.7856, while
building improves85.5504 versus83.9035. The complete and reference readouts share
backbone, view and text, so subtracting them cannot retain every advantage that
this shared information offers over local Geometry. The observations establish
failure of this fixed difference-readout candidate, not semantic correctness of
either branch or impossibility of every conditional readout.

Eight new CPU tests passed in the remote evaluation environment. Local syntax
compilation passed; local unit-test import was unavailable because its Python
environment lacks OpenCV, an existing adapter dependency. The mask-free real
checkpoint smoke gave exact full-feature, full-support and full-profile replay
(all maximum errors0); all weights remained frozen. Unique coverage, transitions
and original Geometry per-image confusions were verified across all eight runs.
The report formatter initially failed on absent-class null IoUs; this was repaired
without changing predictions or rerunning inference. All experiment sessions are
terminal and physical GPUs4-7 are idle. Original best results remain unchanged.

| Dataset/protocol | Geometry | BroadVIP | MeanProb_VIP | MeanLogit_VIP | Anchored_VIP | MeanLogit_Context | ShuffledContextWriteback | Geometry_PairedContext |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vdd/vdd | 31.9462 | 51.1203 | 55.1355 | 56.1762 | 57.0012 | 32.3571 | 32.5212 | 32.4305 |
| potsdam/potsdam | 40.3528 | 40.5048 | 40.8641 | 40.8725 | 41.7806 | 40.6632 | 40.5191 | 40.6508 |
| udd5/udd5 | 50.5553 | 45.2814 | 48.5016 | 48.8694 | 49.2117 | 50.3498 | 50.4052 | 50.4548 |
| oem/oem | 39.3543 | 28.1181 | 31.0853 | 31.8024 | 31.9533 | 39.4488 | 39.4162 | 39.4547 |
| loveda/P | 62.8254 | 56.3776 | 61.9651 | 62.7434 | 62.4863 | 62.9502 | 63.0009 | 62.9356 |
| loveda/D | 38.4558 | 33.2861 | 35.9632 | 36.6573 | 36.5600 | 38.4413 | 38.4569 | 38.4653 |
| vaihingen/vaihingen | 50.2325 | 47.6306 | 49.8264 | 50.0312 | 51.4491 | 50.4605 | 50.2465 | 50.4525 |
| landcoverai/landcoverai | 60.9049 | 65.8638 | 66.1395 | 66.4263 | 66.9060 | 60.9044 | 60.9069 | 60.9090 |
| flair1/flair1 | 38.8428 | 29.7067 | 31.6911 | 32.3648 | 33.6084 | 38.8629 | 38.8460 | 38.8732 |
| Equal-domain mean, LoveDA D once | 43.830569 | 42.688973 | 44.900830 | 45.400028 | 46.058780 | 43.935994 | 43.914756 | 43.961350 |

All five full40 UDD5 Geometry/VIP/fusion controls exactly reproduce their historical confusion matrices.

## Prospective checks

```json
{
  "mean_vs_Geometry": true,
  "mean_vs_Anchored_VIP": false,
  "mean_vs_MeanLogit_VIP": false,
  "mean_vs_MeanProb_VIP": false,
  "mean_vs_MeanLogit_Context": true,
  "retain_potsdam_potsdam": true,
  "potsdam_vs_Geometry": true,
  "potsdam_vs_Anchored_VIP": false,
  "retain_oem_oem": true,
  "retain_loveda_P": true,
  "retain_loveda_D": true,
  "retain_flair1_flair1": true,
  "retain_vaihingen_vaihingen": true,
  "retain_landcoverai_landcoverai": true,
  "retain_vdd_vdd": true,
  "vdd_vs_Geometry": true,
  "vdd_vs_Anchored_VIP": false,
  "retain_udd5_udd5": true
}
```

## Per-class metrics

### vdd/vdd

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| other | 25.3949 | 53.4471 | 25.6464 | 25.6390 | 0.2441 | -27.8081 |
| wall | 11.8982 | 57.7209 | 12.8356 | 12.8204 | 0.9222 | -44.9005 |
| road | 24.1970 | 25.5594 | 21.5006 | 21.5505 | -2.6465 | -4.0089 |
| vegetation | 62.0257 | 60.0299 | 63.4430 | 63.6202 | 1.5945 | 3.5903 |
| vehicle | 5.6965 | 25.3443 | 5.5562 | 5.6292 | -0.0673 | -19.7151 |
| roof | 60.7577 | 84.4472 | 62.0870 | 62.2690 | 1.5113 | -22.1782 |
| water | 33.6534 | 92.4595 | 35.4308 | 35.4850 | 1.8316 | -56.9745 |

Diagnostics:
```json
{
  "tiles": 704,
  "wide_crops": 0.022727272727272728,
  "reference_crops": 1.5454545454545454,
  "mapped_queries": 28.636363636363637,
  "reference_empty_rows": 0.0,
  "reference_rows": 1363.090909090909,
  "reference_support_fraction_sum": 0.03707391495250208,
  "mean_absolute_innovation": 0.05207896652725388,
  "changed_patch_fraction": 0.10445057262073863,
  "solver_relative_residual": 8.090384540486313e-07,
  "solver_iterations": 8.639204545454545,
  "energy_before": 73.45471986831927,
  "energy_after": 37.686265243175015,
  "mean_absolute_context_delta": 0.11260220231964592
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 30699885 | 5340270 |
| MeanLogit_Context | 3834348 | 2720702 |
| Geometry_PairedContext | 3739695 | 2539171 |

### potsdam/potsdam

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 51.6875 | 62.1577 | 51.9130 | 52.0174 | 0.3299 | -10.1403 |
| building | 78.2026 | 75.5787 | 78.1862 | 78.2980 | 0.0954 | 2.7193 |
| low vegetation | 33.4221 | 21.2306 | 33.4487 | 33.2251 | -0.1970 | 11.9945 |
| tree | 62.5813 | 64.8712 | 63.8549 | 63.8355 | 1.2542 | -1.0357 |
| car | 11.6582 | 24.2128 | 11.8622 | 11.8530 | 0.1948 | -12.3598 |
| clutter | 4.5652 | 2.6325 | 4.7145 | 4.6759 | 0.1107 | 2.0434 |

Diagnostics:
```json
{
  "tiles": 72,
  "wide_crops": 0.4444444444444444,
  "reference_crops": 4.0,
  "mapped_queries": 592.1111111111111,
  "reference_empty_rows": 0.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 0.4519174804704057,
  "mean_absolute_innovation": 0.02048657875921991,
  "changed_patch_fraction": 0.030042860243055556,
  "solver_relative_residual": 4.490644648234997e-07,
  "solver_iterations": 9.0,
  "energy_before": 11.218367084860802,
  "energy_after": 6.150005391902393,
  "mean_absolute_context_delta": 0.051757734475864306
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 720293 | 420742 |
| MeanLogit_Context | 109450 | 63157 |
| Geometry_PairedContext | 83164 | 39491 |

### udd5/udd5

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| vegetation | 82.4937 | 68.3846 | 81.4454 | 81.6210 | -0.8727 | 13.2364 |
| building | 83.9035 | 82.5264 | 85.2996 | 85.5504 | 1.6469 | 3.0240 |
| road | 45.7856 | 39.8083 | 44.6557 | 44.6720 | -1.1136 | 4.8637 |
| vehicle | 10.0092 | 20.4213 | 9.5818 | 9.5506 | -0.4586 | -10.8707 |
| other | 30.5846 | 34.9177 | 30.7662 | 30.8800 | 0.2954 | -4.0377 |

Diagnostics:
```json
{
  "tiles": 3232,
  "wide_crops": 0.024752475247524754,
  "reference_crops": 1.5643564356435644,
  "mapped_queries": 28.771039603960396,
  "reference_empty_rows": 0.0,
  "reference_rows": 1379.7623762376238,
  "reference_support_fraction_sum": 0.03672925563728538,
  "mean_absolute_innovation": 0.08245709387177491,
  "changed_patch_fraction": 0.0936067788907797,
  "solver_relative_residual": 7.871779274619859e-07,
  "solver_iterations": 8.601485148514852,
  "energy_before": 120.60110260413425,
  "energy_after": 61.354089746380794,
  "mean_absolute_context_delta": 0.17497528569270293
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 29576997 | 37113904 |
| MeanLogit_Context | 15468355 | 14712667 |
| Geometry_PairedContext | 15116104 | 13905234 |

### oem/oem

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rangeland | 47.7780 | 37.5959 | 47.8551 | 47.8651 | 0.0871 | 10.2692 |
| developed space | 28.5294 | 24.0531 | 28.6647 | 28.6236 | 0.0942 | 4.5705 |
| road | 40.8291 | 42.1205 | 40.8503 | 40.8230 | -0.0061 | -1.2975 |
| tree | 56.0657 | 38.7562 | 56.1613 | 56.1647 | 0.0990 | 17.4085 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | 79.1353 | 68.8612 | 79.4709 | 79.5302 | 0.3949 | 10.6690 |
| building | 62.4971 | 44.2393 | 62.5884 | 62.6307 | 0.1336 | 18.3914 |

Diagnostics:
```json
{
  "tiles": 67,
  "wide_crops": 0.47761194029850745,
  "reference_crops": 4.0,
  "mapped_queries": 564.8358208955224,
  "reference_empty_rows": 0.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 0.5591125379302608,
  "mean_absolute_innovation": 0.008602310710774696,
  "changed_patch_fraction": 0.010319496268656716,
  "solver_relative_residual": 4.886555489268977e-07,
  "solver_iterations": 8.985074626865671,
  "energy_before": 3.895235294655987,
  "energy_after": 2.267152070132112,
  "mean_absolute_context_delta": 0.024348434474129828
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 459329 | 1251366 |
| MeanLogit_Context | 28946 | 18892 |
| Geometry_PairedContext | 22169 | 12498 |

### loveda/P

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| building | 64.4987 | 79.5953 | 64.4812 | 64.5007 | 0.0020 | -15.0946 |
| road | 66.9348 | 68.0988 | 66.9139 | 66.9158 | -0.0190 | -1.1830 |
| water | 81.1851 | 79.3816 | 81.2976 | 81.2963 | 0.1112 | 1.9147 |
| barren | 24.3403 | 23.8204 | 24.7652 | 24.5947 | 0.2544 | 0.7743 |
| tree | 53.7307 | 36.7931 | 53.8491 | 53.8784 | 0.1477 | 17.0853 |
| farm | 86.2626 | 87.2288 | 86.3943 | 86.4276 | 0.1650 | -0.8012 |

Diagnostics:
```json
{
  "tiles": 72,
  "wide_crops": 0.4444444444444444,
  "reference_crops": 4.0,
  "mapped_queries": 513.7777777777778,
  "reference_empty_rows": 441.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 0.45984441683524185,
  "mean_absolute_innovation": 0.007293647120731193,
  "changed_patch_fraction": 0.009209526909722222,
  "solver_relative_residual": 4.427051752347527e-07,
  "solver_iterations": 8.444444444444445,
  "energy_before": 2.0422309241056005,
  "energy_after": 1.1696825135305113,
  "mean_absolute_context_delta": 0.021257031576548496
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 268850 | 259754 |
| MeanLogit_Context | 13788 | 8777 |
| Geometry_PairedContext | 10818 | 5237 |

### loveda/D

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| background | 34.1983 | 6.4379 | 34.0262 | 34.0757 | -0.1226 | 27.6378 |
| building | 31.1862 | 36.3815 | 31.1478 | 31.2097 | 0.0235 | -5.1718 |
| road | 45.4818 | 56.4727 | 45.4766 | 45.4560 | -0.0258 | -11.0167 |
| water | 61.9193 | 62.7476 | 61.9719 | 62.0055 | 0.0862 | -0.7421 |
| barren | 14.5296 | 20.0167 | 14.4640 | 14.4425 | -0.0871 | -5.5742 |
| tree | 26.8904 | 31.2898 | 26.9054 | 26.9319 | 0.0415 | -4.3579 |
| farm | 54.9851 | 42.5737 | 55.0970 | 55.1357 | 0.1506 | 12.5620 |

Diagnostics:
```json
{
  "tiles": 72,
  "wide_crops": 0.4444444444444444,
  "reference_crops": 4.0,
  "mapped_queries": 513.7777777777778,
  "reference_empty_rows": 441.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 0.45984441683524185,
  "mean_absolute_innovation": 0.007182511881764084,
  "changed_patch_fraction": 0.010945638020833334,
  "solver_relative_residual": 4.4691593675199533e-07,
  "solver_iterations": 8.444444444444445,
  "energy_before": 2.3177570931373137,
  "energy_after": 1.3302451698659752,
  "mean_absolute_context_delta": 0.021049242295501043
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 833269 | 1635494 |
| MeanLogit_Context | 31002 | 32021 |
| Geometry_PairedContext | 24755 | 22749 |

### vaihingen/vaihingen

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 49.2002 | 60.0516 | 49.7745 | 49.8090 | 0.6088 | -10.2426 |
| building | 74.6610 | 68.2387 | 74.5968 | 74.6239 | -0.0371 | 6.3852 |
| low vegetation | 46.7115 | 35.3912 | 47.2313 | 47.0467 | 0.3352 | 11.6555 |
| tree | 71.8329 | 71.8529 | 71.8086 | 71.9056 | 0.0727 | 0.0527 |
| car | 8.7566 | 21.7111 | 8.8913 | 8.8775 | 0.1209 | -12.8336 |

Diagnostics:
```json
{
  "tiles": 72,
  "wide_crops": 0.4444444444444444,
  "reference_crops": 4.0,
  "mapped_queries": 592.1111111111111,
  "reference_empty_rows": 0.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 0.4839208492388328,
  "mean_absolute_innovation": 0.020958054771957297,
  "changed_patch_fraction": 0.023451063368055556,
  "solver_relative_residual": 4.087603764604511e-07,
  "solver_iterations": 9.0,
  "energy_before": 9.213097676634789,
  "energy_after": 5.0809276252985,
  "mean_absolute_context_delta": 0.0540307388227019
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 647944 | 382120 |
| MeanLogit_Context | 92831 | 67062 |
| Geometry_PairedContext | 71153 | 46440 |

### landcoverai/landcoverai

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| background | 82.6090 | 87.4477 | 82.6090 | 82.6181 | 0.0091 | -4.8296 |
| building | 34.9562 | 44.4672 | 34.9311 | 34.9411 | -0.0151 | -9.5261 |
| woodland | 78.3638 | 78.6026 | 78.3799 | 78.3906 | 0.0268 | -0.2120 |
| water | 93.6635 | 97.6178 | 93.6615 | 93.6625 | -0.0010 | -3.9553 |
| road | 14.9318 | 26.3947 | 14.9405 | 14.9330 | 0.0012 | -11.4617 |

Diagnostics:
```json
{
  "tiles": 8,
  "wide_crops": 4.0,
  "reference_crops": 4.0,
  "mapped_queries": 1764.0,
  "reference_empty_rows": 1984.5,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 1.4302721321582794,
  "mean_absolute_innovation": 0.00028080975243938155,
  "changed_patch_fraction": 0.0008544921875,
  "solver_relative_residual": 2.2388542930684707e-07,
  "solver_iterations": 4.0,
  "energy_before": 0.008369743241928518,
  "energy_after": 0.005281051795464009,
  "mean_absolute_context_delta": 0.0010352726676501334
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 113587 | 37157 |
| MeanLogit_Context | 804 | 751 |
| Geometry_PairedContext | 437 | 298 |

### flair1/flair1

| Class | Geometry | Anchored_VIP | MeanLogit_Context | Geometry_PairedContext | Delta vs Geometry | Delta vs Anchored |
|---|---:|---:|---:|---:|---:|---:|
| building | 49.6979 | 56.7755 | 49.6801 | 49.7418 | 0.0439 | -7.0337 |
| pervious surface | 57.0621 | 47.2756 | 57.0127 | 57.0229 | -0.0392 | 9.7473 |
| impervious surface | 52.6509 | 51.9858 | 52.6457 | 52.6297 | -0.0212 | 0.6439 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| water | 73.2328 | 61.4565 | 73.2175 | 73.2119 | -0.0209 | 11.7554 |
| coniferous | 43.0233 | 43.5282 | 43.3883 | 43.3779 | 0.3546 | -0.1503 |
| deciduous | 54.0229 | 59.5237 | 54.1048 | 54.1425 | 0.1196 | -5.3812 |
| brushwood | 18.6136 | 12.3287 | 18.6022 | 18.6065 | -0.0071 | 6.2778 |
| vineyard | -- | 0.0000 | -- | -- | -- | -- |
| herbaceous vegetation | 60.2329 | 57.4725 | 60.1374 | 60.1464 | -0.0865 | 2.6739 |
| agricultural land | 18.7339 | 12.9546 | 18.7029 | 18.7255 | -0.0084 | 5.7709 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Diagnostics:
```json
{
  "tiles": 8,
  "wide_crops": 4.0,
  "reference_crops": 4.0,
  "mapped_queries": 1764.0,
  "reference_empty_rows": 441.0,
  "reference_rows": 3528.0,
  "reference_support_fraction_sum": 1.3180201761424541,
  "mean_absolute_innovation": 0.0022551467000084813,
  "changed_patch_fraction": 0.00244140625,
  "solver_relative_residual": 4.1385476023947376e-07,
  "solver_iterations": 7.0,
  "energy_before": 0.9128388793906197,
  "energy_after": 0.5390076521798619,
  "mean_absolute_context_delta": 0.006945372559130192
}
```

| Method | Beneficial changed pixels | Harmful changed pixels |
|---|---:|---:|
| Anchored_VIP | 98118 | 134725 |
| MeanLogit_Context | 2439 | 2994 |
| Geometry_PairedContext | 1523 | 1689 |

## Runtime

Suite elapsed seconds: 581.3291001319885. Each dataset duration below covers all eight arms, not independent primary latency.

| Dataset | Parallel wall seconds | Peak allocated CUDA MiB |
|---|---:|---:|
| vdd | 132.4116 | 5836.0381 |
| potsdam | 10.5998 | 5559.6035 |
| udd5 | 505.9067 | 5638.8965 |
| oem | 10.7148 | 5574.7998 |
| loveda | 15.7410 | 5607.5718 |
| vaihingen | 9.8502 | 5551.4766 |
| landcoverai | 2.4541 | 5446.4287 |
| flair1 | 3.1366 | 5490.0068 |

This fixed candidate failed its prospective promotion gate. No full rollout or label-based parameter adjustment was performed. Original Geometry and historical Anchored_VIP remain preserved.
