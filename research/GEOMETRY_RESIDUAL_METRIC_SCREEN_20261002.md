# Geometry residual semantic metric: verified screen

Same96 COMPLETE images: UDD5 full40, other seven domains8 each. Not full eight-dataset totals. Corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. LoveDA D enters the mean once; P separately. Every domain is development data.

Original frozen Geometry is unchanged. One original fine512 view; no extra encoder, VIP observer or context prediction in primary. All20 aliases and original RS/LME/assembly remain. The union text span is uncentered, only numerically rank-reduced; image covariance changes the joint visual/text cosine metric, not which aliases are retained. Window metric adaptation is not adaptation-free or a learned network. No labels select aliases, rank, metric strength or routes.

17 focused tests pass locally/remotely. Mask-free real-checkpoint smoke and independent local result checks verify exact zero-covariance identity, identity-Geometry residual, frozen head, unchanged all20/checkpoints/unique complete IDs and exact per-image Geometry/SCLIP/VIPProxy controls. Per-image confusion sums and transition endpoints reconstruct every result.

Mahalanobis metrics, whitening and graph residuals are established. The specific bounded coupling is a hypothesis, not a verified priority claim. Geometry residuals can contain true class boundaries and detail; metric eigenvalue bounds are not semantic reliability bounds. A constant wrong semantic mode is not removed by this construction.

## mIoU

| Dataset/protocol | Geometry | SCLIP_Two | VIPProxy_Two | UniformMetric | SpatialMetric | ShuffledMetric | MeanLogit_UniformMetric | Geometry_ResidualMetric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.9462 | 30.6800 | 31.1056 | 31.9572 | 32.0320 | 31.9901 | 31.9960 | 32.0031 |
| potsdam/potsdam | 40.3528 | 44.0525 | 42.7377 | 37.9896 | 38.5296 | 39.0740 | 39.3002 | 40.1153 |
| udd5/udd5 | 50.5553 | 50.2837 | 49.9071 | 49.4964 | 49.7702 | 50.0113 | 50.1013 | 50.4920 |
| oem/oem | 39.3543 | 37.3120 | 38.9899 | 39.5549 | 39.5588 | 39.4496 | 39.4926 | 39.3479 |
| loveda/P | 62.8254 | 68.2854 | 68.5692 | 60.9196 | 61.2726 | 61.7483 | 62.0079 | 62.4960 |
| loveda/D | 38.4558 | 36.5043 | 35.6670 | 38.1470 | 38.3059 | 38.2216 | 38.3683 | 38.3209 |
| vaihingen/vaihingen | 50.2325 | 52.0872 | 50.5393 | 48.7875 | 48.8210 | 49.1124 | 49.6045 | 50.1763 |
| landcoverai/landcoverai | 60.9049 | 61.7170 | 59.3073 | 59.9807 | 59.8894 | 60.0053 | 60.7584 | 60.4775 |
| flair1/flair1 | 38.8428 | 37.6225 | 39.3436 | 37.8558 | 38.1923 | 38.1866 | 38.3848 | 38.8444 |
| Equal-domain mean | 43.830569 | 43.782393 | 43.449702 | 42.971138 | 43.137402 | 43.256352 | 43.500753 | 43.722164 |

Fixed gate passed: False. Mean primary delta versus Geometry: -0.108405pp; versus global metric: +0.751026pp; versus same-source fusion: +0.221411pp.

Failed checks: mean_vs_Geometry, mean_vs_SCLIP_Two, retain_potsdam_potsdam, potsdam_vs_Geometry, retain_udd5_udd5, retain_oem_oem, retain_loveda_P, retain_loveda_D, retain_vaihingen_vaihingen, retain_landcoverai_landcoverai

## Foreground Metrics

| Dataset/protocol | Metric | Geometry | UniformMetric | Primary |
| --- | --- | ---: | ---: | ---: |
| loveda/D | foreground_mean_iou_percent | 39.1654 | 38.4194 | 38.9808 |
| udd5/udd5 | non_residual_mean_iou_percent | 55.5480 | 54.2411 | 55.4811 |
| landcoverai/landcoverai | non_residual_mean_iou_percent | 55.4788 | 54.2682 | 55.0477 |

## Mechanism Verdict

Primary beats Geometry on 2/8 domain scores: vdd, flair1. LoveDA P is a separate protocol. Better performance than global/spatial covariance can mean less damage to useful semantic variation; it is not an improvement over original Geometry.

The covariance sees R=(I-G)YB, not a ground-truth noise field. A true class boundary or small-object detail can enter R, while a coherent wrong response can be explained by G and remain outside R. A constant descriptor field gives zero residual and exact unchanged predictions. Thus this construction cannot itself identify a constant wrong class mode. The result does not establish a transferable source of competing-class truth.

The metric contracts visual and text directions, but their subsequent cosine normalization can increase or decrease individual alias scores; there is no classwise suppression guarantee. No global mean is removed from the deployed descriptor and no rejected aliases are counted. The covariance bound prevents an ill-conditioned metric, not false-positive semantic changes.

| Domain/protocol | Corrected | Corrupted | Net corrections | Primary minus Geometry | Primary minus uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 747374 | 550036 | +197338 | +0.0569 | +0.0459 |
| potsdam/potsdam | 19029 | 47029 | -28000 | -0.2376 | +2.1257 |
| udd5/udd5 | 1392919 | 1621181 | -228262 | -0.0633 | +0.9955 |
| oem/oem | 36985 | 25134 | +11851 | -0.0064 | -0.2070 |
| loveda/P | 16520 | 14792 | +1728 | -0.3294 | +1.5764 |
| loveda/D | 59804 | 58118 | +1686 | -0.1350 | +0.1738 |
| vaihingen/vaihingen | 22292 | 30397 | -8105 | -0.0562 | +1.3888 |
| landcoverai/landcoverai | 8556 | 17300 | -8744 | -0.4273 | +0.4968 |
| flair1/flair1 | 7747 | 8014 | -267 | +0.0017 | +0.9887 |

The prospective gate fails. Reject promotion of this fixed residual metric; do not tune its covariance strength or infer a universal ceiling. The next action must change the information or its semantic interpretation, not relabel a weaker control as the baseline.

## Per-class Coverage And Competition

### vdd/vdd

Corrected/corrupted/wrong-to-wrong: 747374/550036/820092.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| other | 25.3949 | 25.8394 | 25.3567 | -0.0382 | 39.601/41.347 | 28.102 | -223713 | -817672 |
| wall | 11.8982 | 11.2994 | 11.6165 | -0.2817 | 11.698/94.331 | 15.719 | -32 | +359456 |
| road | 24.1970 | 24.7576 | 23.4963 | -0.7007 | 24.034/91.312 | 5.240 | -4709 | +129559 |
| vegetation | 62.0257 | 60.8142 | 61.7629 | -0.2628 | 84.993/69.323 | 17.617 | -111957 | -81893 |
| vehicle | 5.6965 | 5.3947 | 5.5236 | -0.1729 | 5.524/99.955 | 4.275 | -3 | +124532 |
| roof | 60.7577 | 61.9110 | 61.9659 | 1.2082 | 87.831/67.786 | 21.474 | +413526 | +99541 |
| water | 33.6534 | 33.6844 | 34.2999 | 0.6465 | 93.317/35.164 | 7.573 | +124226 | -10861 |

Internal diagnostics:
```json
{
  "tiles": 704,
  "text_span_rank": 140.0,
  "text_span_max_error": 1.7881393432617188e-07,
  "covariance_trace": 0.0157620598196941,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.005202982015444749
}
```

### potsdam/potsdam

Corrected/corrupted/wrong-to-wrong: 19029/47029/41115.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| impervious surface | 51.6875 | 50.8204 | 51.4373 | -0.2502 | 74.982/62.094 | 29.987 | -2433 | +12224 |
| building | 78.2026 | 76.3835 | 78.3176 | 0.1150 | 80.744/96.304 | 14.554 | +762 | -790 |
| low vegetation | 33.4221 | 30.1807 | 33.1261 | -0.2960 | 73.213/37.695 | 9.876 | -9021 | -11526 |
| tree | 62.5813 | 55.4805 | 61.9259 | -0.6554 | 90.810/66.066 | 17.424 | -15925 | -4041 |
| car | 11.6582 | 10.9257 | 11.4718 | -0.1864 | 11.485/99.003 | 20.194 | -43 | +25500 |
| clutter | 4.5652 | 4.1467 | 4.4129 | -0.1523 | 7.470/9.733 | 7.965 | -1340 | +6633 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "text_span_rank": 120.0,
  "text_span_max_error": 1.7881393432617188e-07,
  "covariance_trace": 0.01763819243448476,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.0041471480508334935
}
```

### udd5/udd5

Corrected/corrupted/wrong-to-wrong: 1392919/1621181/1129374.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| vegetation | 82.4937 | 80.7264 | 82.2202 | -0.2735 | 96.577/84.688 | 25.973 | -469891 | -124572 |
| building | 83.9035 | 83.7202 | 84.0726 | 0.1691 | 88.352/94.553 | 42.014 | +386425 | +69216 |
| road | 45.7856 | 43.3530 | 45.8691 | 0.0835 | 70.536/56.740 | 10.787 | +128954 | +148522 |
| vehicle | 10.0092 | 9.1647 | 9.7627 | -0.2465 | 9.785/97.741 | 7.996 | -255 | +865605 |
| other | 30.5846 | 30.5178 | 30.5353 | -0.0493 | 53.295/41.691 | 13.230 | -273495 | -730509 |

Internal diagnostics:
```json
{
  "tiles": 3232,
  "text_span_rank": 100.0,
  "text_span_max_error": 1.1920928955078125e-07,
  "covariance_trace": 0.01375902653326013,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.005887298681225748
}
```

### oem/oem

Corrected/corrupted/wrong-to-wrong: 36985/25134/21087.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 10.111 | +0 | -24127 |
| rangeland | 47.7780 | 47.1656 | 47.6636 | -0.1144 | 65.328/63.804 | 18.700 | -3483 | -2580 |
| developed space | 28.5294 | 31.9339 | 29.2125 | 0.6831 | 62.222/35.511 | 12.783 | +13718 | -1960 |
| road | 40.8291 | 40.6960 | 40.5322 | -0.2969 | 45.200/79.694 | 8.045 | +1940 | +9760 |
| tree | 56.0657 | 54.6311 | 56.1647 | 0.0990 | 87.417/61.105 | 17.775 | +2574 | +840 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.067 | +0 | +184 |
| agriculture land | 79.1353 | 79.5911 | 79.0928 | -0.0425 | 91.037/85.772 | 15.790 | -5202 | -5824 |
| building | 62.4971 | 62.4217 | 62.1173 | -0.3798 | 65.041/93.253 | 16.728 | +2304 | +11856 |

Internal diagnostics:
```json
{
  "tiles": 67,
  "text_span_rank": 160.0,
  "text_span_max_error": 1.7881393432617188e-07,
  "covariance_trace": 0.024560058572843893,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.0038903382356598308
}
```

### loveda/P

Corrected/corrupted/wrong-to-wrong: 16520/14792/8846.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| building | 64.4987 | 60.1591 | 63.1267 | -1.3720 | 63.944/98.015 | 1.002 | +10 | +1004 |
| road | 66.9348 | 64.2317 | 65.7550 | -1.1798 | 66.413/98.515 | 10.562 | +244 | +8973 |
| water | 81.1851 | 80.1610 | 81.5803 | 0.3952 | 87.179/92.702 | 22.216 | +392 | -4808 |
| barren | 24.3403 | 23.5997 | 24.2844 | -0.0559 | 63.856/28.154 | 3.223 | -240 | -95 |
| tree | 53.7307 | 51.6819 | 53.7517 | 0.0210 | 64.904/75.777 | 12.922 | -2481 | -4897 |
| farm | 86.2626 | 85.6841 | 86.4779 | 0.2153 | 95.417/90.225 | 50.074 | +3803 | -1905 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "text_span_rank": 142.0,
  "text_span_max_error": 1.4901161193847656e-07,
  "covariance_trace": 0.018108628871333268,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.004470396634941507
}
```

### loveda/D

Corrected/corrupted/wrong-to-wrong: 59804/58118/28285.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| background | 34.1983 | 36.5129 | 34.3610 | 0.1627 | 65.729/41.861 | 28.427 | +9223 | +5583 |
| building | 31.1862 | 31.0642 | 30.8841 | -0.3021 | 31.552/93.588 | 1.074 | -23 | +805 |
| road | 45.4818 | 44.5791 | 44.3670 | -1.1148 | 44.690/98.395 | 8.680 | +194 | +18147 |
| water | 61.9193 | 61.4296 | 62.7224 | 0.8031 | 66.598/91.509 | 15.894 | +4807 | -10335 |
| barren | 14.5296 | 12.5391 | 14.2643 | -0.2653 | 59.135/15.824 | 1.083 | -1113 | -881 |
| tree | 26.8904 | 27.1903 | 26.4533 | -0.4371 | 29.587/71.407 | 14.789 | -2357 | +13463 |
| farm | 54.9851 | 53.7140 | 55.1940 | 0.2089 | 70.258/72.022 | 30.054 | -9045 | -28468 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "text_span_rank": 142.0,
  "text_span_max_error": 1.4901161193847656e-07,
  "covariance_trace": 0.018108628871333268,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.004470396634941507
}
```

### vaihingen/vaihingen

Corrected/corrupted/wrong-to-wrong: 22292/30397/20339.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| impervious surface | 49.2002 | 47.2739 | 49.2830 | 0.0828 | 81.999/55.262 | 20.633 | +6879 | +9386 |
| building | 74.6610 | 72.8861 | 74.8879 | 0.2269 | 75.723/98.548 | 28.050 | +255 | -6506 |
| low vegetation | 46.7115 | 44.3859 | 46.2884 | -0.4231 | 92.002/48.229 | 13.342 | -8890 | +56 |
| tree | 71.8329 | 71.2283 | 71.7392 | -0.0937 | 82.794/84.309 | 21.280 | -6373 | -6326 |
| car | 8.7566 | 8.1632 | 8.6827 | -0.0739 | 8.698/97.995 | 16.695 | +24 | +11495 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "text_span_rank": 100.0,
  "text_span_max_error": 1.4901161193847656e-07,
  "covariance_trace": 0.016347663890984323,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.003859229165957206
}
```

### landcoverai/landcoverai

Corrected/corrupted/wrong-to-wrong: 8556/17300/2153.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| background | 82.6090 | 82.8309 | 82.1968 | -0.4122 | 96.129/85.010 | 59.922 | -474 | +6760 |
| building | 34.9562 | 32.1851 | 34.7186 | -0.2376 | 34.806/99.282 | 4.322 | -3 | +609 |
| woodland | 78.3638 | 78.6980 | 77.9039 | -0.4599 | 87.722/87.438 | 21.338 | -8286 | -7617 |
| water | 93.6635 | 91.8902 | 93.7359 | 0.0724 | 93.736/100.000 | 8.824 | +0 | -143 |
| road | 14.9318 | 14.2994 | 13.8324 | -1.0994 | 14.426/77.088 | 5.594 | +19 | +9135 |

Internal diagnostics:
```json
{
  "tiles": 8,
  "text_span_rank": 100.0,
  "text_span_max_error": 1.4901161193847656e-07,
  "covariance_trace": 0.012556188856251538,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.004860709508648142
}
```

### flair1/flair1

Corrected/corrupted/wrong-to-wrong: 7747/8014/4940.

| Class | Geometry IoU | Uniform metric IoU | Primary IoU | Delta | Primary P/R | Primary area% | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| building | 49.6979 | 47.3768 | 49.9017 | 0.2038 | 50.866/96.340 | 13.598 | +388 | -411 |
| pervious surface | 57.0621 | 58.1265 | 57.1448 | 0.0827 | 92.531/59.908 | 11.110 | -281 | -1039 |
| impervious surface | 52.6509 | 51.4352 | 52.4751 | -0.1758 | 62.215/77.022 | 20.281 | +843 | +3284 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 3.452 | +0 | +238 |
| water | 73.2328 | 70.6744 | 73.1398 | -0.0930 | 76.890/93.749 | 5.409 | +207 | +434 |
| coniferous | 43.0233 | 44.4969 | 43.0721 | 0.0488 | 61.718/58.775 | 0.535 | +10 | +5 |
| deciduous | 54.0229 | 47.9247 | 53.8298 | -0.1931 | 79.352/62.598 | 13.428 | -1791 | -1832 |
| brushwood | 18.6136 | 17.3603 | 19.1306 | 0.5170 | 23.948/48.745 | 10.011 | +1226 | -711 |
| vineyard | NA | NA | NA | NA | 0.000/0.000 | 0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 59.4590 | 60.1138 | -0.1191 | 94.320/62.372 | 21.184 | -869 | -65 |
| agricultural land | 18.7339 | 19.5594 | 18.4808 | -0.2531 | 21.310/58.198 | 0.833 | +0 | +272 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.158 | +0 | +92 |

Internal diagnostics:
```json
{
  "tiles": 8,
  "text_span_rank": 240.0,
  "text_span_max_error": 2.384185791015625e-07,
  "covariance_trace": 0.036947072483599186,
  "metric_condition_upper_bound": 2.0,
  "mean_absolute_alias_change": 0.004087447159690782
}
```

## Cost And Decision

Parallel suite wall: 646.1968s. Worker timings include all eight arms, not standalone deployment.

| Dataset | Eight-arm seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 162.4630 | 3821.4365 |
| potsdam | 10.4858 | 3818.1782 |
| udd5 | 628.0714 | 3818.7856 |
| oem | 10.8752 | 3820.9834 |
| loveda | 17.4840 | 3822.0474 |
| vaihingen | 8.5855 | 3818.1294 |
| landcoverai | 1.6617 | 3717.1128 |
| flair1 | 2.5774 | 3719.8481 |

### Independent Window Readout

Single512-window original Geometry preparation plus all20 alias scoring and class LME. Primary additionally computes its residual covariance, Cholesky factor and joint metric cosines. One resident model, no comparator heads/covariances. Excludes text encoding/span construction, dense image assembly/transfer, decoding and initialization; not optimized or whole-image latency.

One resident model; three warmups/20 synchronized trials; same fixed OEM image, no masks.

| Arm | Median ms | Peak allocated MiB |
| --- | ---: | ---: |
| Geometry | 26.7870 | 3717.6416 |
| Geometry_ResidualMetric | 28.9636 | 3717.6416 |

As-implemented window readout median ratio: 1.0813x. Not a whole-image ratio; text-span initialization is excluded and must not be described as free.

Do not select Uniform/Spatial/Shuffle controls as per-domain winners. No retrospective covariance/rank/strength search, no automatic full rollout. A failed gate rejects this fixed metric without proving a training-free ceiling; a pass still needs unchanged full-domain results, independent validation and a verified novelty boundary. Preserve Geometry/historical best models.
