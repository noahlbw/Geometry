# Geometry and attention-profile compatibility: results

Signature `geometry-native-role-compatibility-v1-20261002`. Phase screen; status complete. Verified 8/8; pending: none.

## Complete candidate

The unchanged Geometry prior is conditioned on Hellinger distances between frozen native attention profiles in each semantic head. The distance scale is the Geometry-weighted mean for that head. Equal/uninformative profiles return original Geometry exactly. Both blocks keep original query patch/prefix mass, special pathways, residual, MLP and LN. There are no incoming donor quotas, alias deletions, teacher, fitted gate or domain routing.

Hellinger kernels, attention-profile similarity, Gibbs/KL updates and adaptive scales are established mechanisms, not claimed mathematical inventions. Profile agreement does not certify semantic correctness, and precision-only scaling can expose small noisy differences.

## Coverage and metrics

The screen has full40 UDD5 plus eight fixed samples for each other domain,96 images total. Partial screens are not full benchmarks. All domains are development data. Corrected IRRG Vaihingen is used; LandCover.ai replaces unlabeled iSAID. Matched VIPProxy_Two is not official VIP.

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | RoleOnly | MeanLogit_RoleCompatible | Geometry_RoleCompatible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 31.4204 | 31.8081 | 31.6467 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 33.4961 | 40.2742 | 40.1809 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 47.6229 | 50.3800 | 50.1738 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 39.0201 | 39.4383 | 39.5040 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 58.2303 | 62.2012 | 61.5712 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 37.6927 | 38.3867 | 38.2467 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 45.4784 | 50.2444 | 50.2190 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 62.0113 | 60.8913 | 60.8735 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 34.1744 | 38.5991 | 38.3479 |

## Frozen decision

Equal-domain mean counts LoveDA D once.

| Method | Mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| RoleOnly | 41.364535 |
| MeanLogit_RoleCompatible | 43.752746 |
| Geometry_RoleCompatible | 43.649058 |

Promotion gate passed: False.

| Check | Passed |
| --- | --- |
| mean_vs_Geometry | False |
| mean_vs_SCLIP_Two | False |
| mean_vs_VIPProxy_Two | True |
| mean_vs_RoleOnly | True |
| mean_vs_MeanLogit_RoleCompatible | False |
| retain_potsdam_potsdam | False |
| focus_potsdam_vs_Geometry | False |
| focus_potsdam_vs_VIPProxy_Two | False |
| focus_potsdam_vs_MeanLogit_RoleCompatible | False |
| retain_oem_oem | True |
| retain_loveda_P | False |
| retain_loveda_D | False |
| retain_flair1_flair1 | False |
| retain_landcoverai_landcoverai | False |
| retain_vaihingen_vaihingen | False |
| retain_vdd_vdd | False |
| focus_vdd_vs_Geometry | False |
| focus_vdd_vs_VIPProxy_Two | True |
| focus_vdd_vs_MeanLogit_RoleCompatible | False |
| retain_udd5_udd5 | False |

A failed prospective gate prevents full rollout. No outcome-based role-scale or domain tuning.

## vdd/vdd competition

Primary delta vs Geometry -0.299501; vs VIPProxy +0.541070; vs RoleOnly +0.226300; vs fixed mean-logit -0.161383.

Beneficial 758174; harmful 1127751. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| other | 25.3949 | 26.1403 | 0.7454 | 38.928/42.213 | 39.004/44.215 | 29.186/30.512 | +517441 | +754695 |
| wall | 11.8982 | 12.1192 | 0.2210 | 11.984/94.332 | 12.209/94.294 | 15.344/15.055 | -725 | -276631 |
| road | 24.1970 | 24.3293 | 0.1323 | 24.741/91.668 | 24.946/90.770 | 5.110/5.019 | -11886 | -76128 |
| vegetation | 62.0257 | 61.4662 | -0.5595 | 84.684/69.863 | 85.208/68.808 | 17.819/17.442 | -218706 | -143231 |
| vehicle | 5.6965 | 5.4128 | -0.2837 | 5.697/99.956 | 5.413/99.933 | 4.145/4.361 | -52 | +207591 |
| roof | 60.7577 | 59.2943 | -1.4634 | 88.015/66.238 | 88.096/64.459 | 20.940/20.359 | -475179 | -82652 |
| water | 33.6534 | 32.7647 | -0.8887 | 93.059/34.520 | 93.067/33.584 | 7.455/7.252 | -180470 | -14067 |

## potsdam/potsdam competition

Primary delta vs Geometry -0.171941; vs VIPProxy -2.556850; vs RoleOnly +6.684804; vs fixed mean-logit -0.093287.

Beneficial 57935; harmful 72230. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 51.8288 | 0.1413 | 75.391/62.178 | 75.253/62.477 | 29.864/30.063 | +8670 | +7228 |
| building | 78.2026 | 77.9851 | -0.2175 | 80.677/96.226 | 80.610/95.992 | 14.555/14.531 | -2291 | +413 |
| low vegetation | 33.4221 | 33.3307 | -0.0914 | 72.470/38.283 | 72.702/38.099 | 10.133/10.052 | -2820 | -3640 |
| tree | 62.5813 | 61.6546 | -0.9267 | 90.654/66.897 | 90.521/65.910 | 17.673/17.438 | -18923 | +90 |
| car | 11.6582 | 11.6737 | 0.0155 | 11.672/99.026 | 11.687/99.022 | 19.876/19.849 | -8 | -2173 |
| clutter | 4.5652 | 4.6124 | 0.0472 | 7.745/10.007 | 7.750/10.228 | 7.899/8.067 | +1077 | +12377 |

## udd5/udd5 competition

Primary delta vs Geometry -0.381517; vs VIPProxy +0.266669; vs RoleOnly +2.550899; vs fixed mean-logit -0.206202.

Beneficial 1782201; harmful 3996036. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 81.9446 | -0.5491 | 96.486/85.049 | 96.551/84.416 | 26.108/25.896 | -825162 | -106805 |
| building | 83.9035 | 83.3305 | -0.5730 | 88.360/94.329 | 88.602/93.336 | 41.911/41.356 | -1715987 | -723886 |
| road | 45.7856 | 45.0029 | -0.7827 | 70.678/56.522 | 70.546/55.415 | 10.724/10.534 | -652952 | -184252 |
| vehicle | 10.0092 | 9.9783 | -0.0309 | 10.032/97.748 | 10.001/97.806 | 7.799/7.829 | +2066 | +127089 |
| other | 30.5846 | 30.6126 | 0.0280 | 52.854/42.059 | 50.992/43.374 | 13.458/14.385 | +978200 | +3101689 |

Geometry non_residual_mean_iou_percent: 55.5480.

Geometry_RoleCompatible non_residual_mean_iou_percent: 55.0641.

## oem/oem competition

Primary delta vs Geometry +0.149624; vs VIPProxy +0.514047; vs RoleOnly +0.483890; vs fixed mean-logit +0.065676.

Beneficial 74021; harmful 50684. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 10.425/10.121 | +0 | -23369 |
| rangeland | 47.7780 | 47.6839 | -0.0941 | 65.294/64.041 | 65.002/64.155 | 18.779/18.897 | +1671 | +7394 |
| developed space | 28.5294 | 30.1524 | 1.6230 | 61.562/34.713 | 62.636/36.765 | 12.630/13.148 | +35289 | +4419 |
| road | 40.8291 | 41.1731 | 0.3440 | 45.753/79.140 | 45.970/79.780 | 7.893/7.919 | +2242 | -228 |
| tree | 56.0657 | 56.2382 | 0.1725 | 87.448/60.973 | 87.513/61.145 | 17.731/17.767 | +3355 | -544 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.065/0.070 | +0 | +378 |
| agriculture land | 79.1353 | 77.9342 | -1.2011 | 90.641/86.177 | 90.482/84.894 | 15.934/15.724 | -16503 | +417 |
| building | 62.4971 | 62.8498 | 0.3527 | 65.584/92.995 | 66.126/92.692 | 16.544/16.355 | -2717 | -11804 |

## loveda/P competition

Primary delta vs Geometry -1.254152; vs VIPProxy -6.998006; vs RoleOnly +3.340919; vs fixed mean-logit -0.630010.

Beneficial 9533; harmful 59005. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| building | 64.4987 | 62.9950 | -1.5037 | 65.367/97.982 | 63.788/98.065 | 0.980/1.005 | +25 | +1125 |
| road | 66.9348 | 65.8339 | -1.1009 | 67.652/98.440 | 66.563/98.363 | 10.361/10.522 | -252 | +7634 |
| water | 81.1851 | 80.9148 | -0.2703 | 86.764/92.661 | 87.051/91.986 | 22.313/22.077 | -6455 | -4330 |
| barren | 24.3403 | 24.1982 | -0.1421 | 63.873/28.226 | 61.174/28.589 | 3.230/3.416 | +1214 | +7295 |
| tree | 53.7307 | 50.8289 | -2.9018 | 64.518/76.267 | 60.278/76.429 | 13.083/14.033 | +821 | +42673 |
| farm | 86.2626 | 84.6565 | -1.6061 | 95.330/90.069 | 95.446/88.220 | 50.033/48.946 | -44825 | -4925 |

## loveda/D competition

Primary delta vs Geometry -0.209136; vs VIPProxy +2.579702; vs RoleOnly +0.553962; vs fixed mean-logit -0.139986.

Beneficial 124993; harmful 130833. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| background | 34.1983 | 35.4658 | 1.2675 | 65.750/41.611 | 64.399/44.115 | 28.248/30.577 | +92422 | +100102 |
| building | 31.1862 | 30.7333 | -0.4529 | 31.858/93.665 | 31.335/94.116 | 1.064/1.087 | +135 | +1764 |
| road | 45.4818 | 45.0684 | -0.4134 | 45.835/98.335 | 45.430/98.266 | 8.458/8.527 | -224 | +5968 |
| water | 61.9193 | 61.6112 | -0.3081 | 65.955/91.007 | 66.409/89.504 | 15.961/15.590 | -14370 | -16293 |
| barren | 14.5296 | 14.1076 | -0.4220 | 59.063/16.157 | 57.451/15.754 | 1.107/1.110 | -1349 | +1570 |
| tree | 26.8904 | 27.2599 | 0.3695 | 30.053/71.873 | 30.606/71.376 | 14.654/14.290 | -2516 | -27590 |
| farm | 54.9851 | 53.4807 | -1.5044 | 69.572/72.395 | 70.294/69.097 | 30.508/28.820 | -79938 | -59681 |

Geometry foreground_mean_iou_percent: 39.1654.

Geometry_RoleCompatible foreground_mean_iou_percent: 38.7102.

## vaihingen/vaihingen competition

Primary delta vs Geometry -0.013456; vs VIPProxy -0.320323; vs RoleOnly +4.740580; vs fixed mean-logit -0.025407.

Beneficial 64572; harmful 60728. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 49.5798 | 0.3796 | 82.397/54.979 | 81.292/55.965 | 20.428/21.077 | +24002 | +27591 |
| building | 74.6610 | 74.6259 | -0.0351 | 75.500/98.533 | 75.648/98.222 | 28.129/27.985 | -5326 | -6073 |
| low vegetation | 46.7115 | 46.5947 | -0.1168 | 92.073/48.669 | 91.721/48.640 | 13.453/13.497 | -573 | +4053 |
| tree | 71.8329 | 71.3869 | -0.4460 | 82.551/84.692 | 82.787/83.830 | 21.440/21.161 | -14327 | -7834 |
| car | 8.7566 | 8.9077 | 0.1511 | 8.773/97.975 | 8.924/98.033 | 16.550/16.279 | +68 | -21581 |

## landcoverai/landcoverai competition

Primary delta vs Geometry -0.031321; vs VIPProxy +1.566205; vs RoleOnly -1.137747; vs fixed mean-logit -0.017735.

Beneficial 7810; harmful 8405. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| background | 82.6090 | 82.5833 | -0.0257 | 96.651/85.044 | 96.482/85.147 | 59.623/59.800 | +1474 | +2240 |
| building | 34.9562 | 35.0928 | 0.1366 | 35.044/99.292 | 35.178/99.311 | 4.293/4.277 | +6 | -334 |
| woodland | 78.3638 | 78.3341 | -0.0297 | 86.499/89.284 | 86.899/88.823 | 22.096/21.881 | -2068 | -2446 |
| water | 93.6635 | 93.5514 | -0.1121 | 93.664/100.000 | 93.551/100.000 | 8.831/8.841 | +0 | +222 |
| road | 14.9318 | 14.8062 | -0.1256 | 15.629/77.002 | 15.493/76.970 | 5.158/5.201 | -7 | +913 |

Geometry foreground_mean_iou_percent: 55.4788.

Geometry non_residual_mean_iou_percent: 55.4788.

Geometry_RoleCompatible foreground_mean_iou_percent: 55.4461.

Geometry_RoleCompatible non_residual_mean_iou_percent: 55.4461.

## flair1/flair1 competition

Primary delta vs Geometry -0.494844; vs VIPProxy -0.995672; vs RoleOnly +4.173491; vs fixed mean-logit -0.251185.

Beneficial 11268; harmful 23816. These pixel totals do not define mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| building | 49.6979 | 49.2900 | -0.4079 | 50.726/96.082 | 50.337/95.950 | 13.599/13.685 | -199 | +2004 |
| pervious surface | 57.0621 | 55.9692 | -1.0929 | 92.130/59.986 | 92.097/58.793 | 11.173/10.954 | -4293 | -286 |
| impervious surface | 52.6509 | 52.2318 | -0.4191 | 62.624/76.777 | 62.052/76.746 | 20.085/20.262 | -105 | +3817 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 3.441/3.509 | +0 | +1424 |
| water | 73.2328 | 72.6047 | -0.6281 | 77.143/93.526 | 76.328/93.705 | 5.378/5.446 | +166 | +1256 |
| coniferous | 43.0233 | 43.1287 | 0.1054 | 61.711/58.690 | 62.993/57.765 | 0.535/0.515 | -109 | -292 |
| deciduous | 54.0229 | 52.0045 | -2.0184 | 78.972/63.099 | 79.024/60.333 | 13.600/12.996 | -9872 | -2807 |
| brushwood | 18.6136 | 17.7942 | -0.8194 | 23.421/47.556 | 22.356/46.583 | 9.987/10.249 | -1003 | +6493 |
| vineyard | undefined | undefined | undefined | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 60.6319 | 0.3990 | 94.317/62.501 | 94.323/62.928 | 21.229/21.372 | +2867 | +143 |
| agricultural land | 18.7339 | 18.1720 | -0.5619 | 21.647/58.198 | 20.900/58.198 | 0.820/0.849 | +0 | +614 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.154/0.163 | +0 | +182 |

## Cost and diagnostics

Joint evaluator timings include all controls; summed durations are GPU allocation time, not active kernel time.

| Dataset | Max shard seconds | Sum shard seconds | Peak MiB | Role scale | Relation L1 displacement | Row error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 156.300 | 156.300 | 4303.475 | 0.07640099 | 0.479743 | 0.00000043 |
| potsdam | 10.475 | 10.475 | 4301.602 | 0.08777316 | 0.500001 | 0.00000047 |
| udd5 | 160.830 | 632.500 | 4302.214 | 0.08581464 | 0.491232 | 0.00000045 |
| oem | 11.591 | 11.591 | 4304.625 | 0.09199507 | 0.502677 | 0.00000044 |
| loveda | 15.456 | 15.456 | 4304.025 | 0.08560851 | 0.494098 | 0.00000041 |
| vaihingen | 9.513 | 9.513 | 4302.214 | 0.09510940 | 0.510414 | 0.00000047 |
| landcoverai | 1.633 | 1.633 | 4293.449 | 0.07599715 | 0.486341 | 0.00000040 |
| flair1 | 2.193 | 2.193 | 4294.544 | 0.09054571 | 0.504134 | 0.00000045 |

Actual-checkpoint FP32/bf16 Geometry identity and cache errors are exactly0. No target masks were loaded for smoke; weights remain frozen.

Window medians: Geometry 0.024152s; primary 0.043669s (1.8081x). Two warmups/five synchronized single-window repeats, original prepare-image pipeline. No control heads in primary timing. Not full-image end-to-end latency.

The objective remains a useful globally fixed Geometry coupling with strong fair comparisons, independent validation and defensible provenance. Neither numerical invariants nor a changed architecture establish CVPR readiness or superiority to the complete official VIP system.
