# Geometry and native donor-budget coupling: results

Signature `geometry-native-donor-marginal-v1-20261002`. Phase screen; status complete. Verified 8/8; pending: none.

## Complete candidate

Original Geometry conditionals and native head-specific donor marginals jointly define a KL-projected attention flow in both frozen blocks. Write (T-Q)V onto the exact original Geometry attended path. Keep all20 aliases, residual/MLP/prefix pathways and original 512/128/Hann assembly. No additional teacher, class quotas, fitted threshold or winner routing.

Sinkhorn/balanced attention is established, not the claimed innovation. This is a fixed native-role coupling hypothesis. The actual smoke shows native quotas are more concentrated than Geometry's incoming mass; do not interpret it as automatic hub suppression. The flow invariant is not a final semantic-correctness guarantee.

## Metrics

Screen: full40 UDD5 plus eight fixed images per other domain,96 images total. Partial-domain screens are not full benchmarks. All domains are development data. VIPProxy_Two is a matched operator adaptation, not official VIP. LandCover.ai replaces unlabeled iSAID.

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | UniformDonorBudget | MeanLogit_DonorBudget | Geometry_DonorBudget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 31.8403 | 30.2612 | 27.0574 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 38.8706 | 33.5217 | 25.1387 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 49.9567 | 47.3342 | 39.4870 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 39.0446 | 38.5939 | 35.0790 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 61.2926 | 56.4750 | 45.6732 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 37.7608 | 35.1782 | 27.9732 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 49.0445 | 46.7584 | 40.6944 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 60.3820 | 58.4938 | 51.6082 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 38.2480 | 33.8732 | 27.8610 |

## Frozen decision

Equal-domain mean counts LoveDA D once.

| Method | Mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| UniformDonorBudget | 43.143425 |
| MeanLogit_DonorBudget | 40.501816 |
| Geometry_DonorBudget | 34.362349 |

Promotion gate passed: False.

| Check | Passed |
| --- | --- |
| mean_vs_Geometry | False |
| mean_vs_VIPProxy_Two | False |
| mean_vs_UniformDonorBudget | False |
| mean_vs_MeanLogit_DonorBudget | False |
| retain_potsdam_potsdam | False |
| focus_potsdam_vs_Geometry | False |
| focus_potsdam_vs_MeanLogit_DonorBudget | False |
| retain_oem_oem | False |
| retain_loveda_P | False |
| retain_loveda_D | False |
| retain_flair1_flair1 | False |
| retain_landcoverai_landcoverai | False |
| retain_vaihingen_vaihingen | False |
| retain_vdd_vdd | False |
| focus_vdd_vs_Geometry | False |
| focus_vdd_vs_MeanLogit_DonorBudget | False |
| retain_udd5_udd5 | False |

No full rollout when the frozen gate fails. No post-result budget or coefficient tuning.

## vdd/vdd changes

Primary delta vs Geometry -4.888766; vs VIPProxy -4.048195; vs uniform budget -4.782852; vs fixed mean-logit -3.203792.

Beneficial 3347149; harmful 9916736. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| other | 25.3949 | 18.7912 | -6.6037 | 38.928/42.213 | 37.277/27.480 | 29.186/19.842 | -3806717 | -5164262 |
| wall | 11.8982 | 9.2572 | -2.6410 | 11.984/94.332 | 9.311/94.178 | 15.344/19.718 | -2897 | +4201343 |
| road | 24.1970 | 17.9509 | -6.2461 | 24.741/91.668 | 18.935/77.543 | 5.110/5.648 | -187030 | +703540 |
| vegetation | 62.0257 | 43.2924 | -18.7333 | 84.684/69.863 | 80.112/48.505 | 17.819/13.078 | -4428554 | -123183 |
| vehicle | 5.6965 | 2.6751 | -3.0214 | 5.697/99.956 | 2.675/99.995 | 4.145/8.831 | +88 | +4498098 |
| roof | 60.7577 | 63.3675 | 2.6098 | 88.015/66.238 | 86.202/70.520 | 20.940/22.763 | +1143960 | +605936 |
| water | 33.6534 | 34.0675 | 0.4141 | 93.059/34.520 | 75.866/38.208 | 7.455/10.121 | +711563 | +1848115 |

## potsdam/potsdam changes

Primary delta vs Geometry -15.214153; vs VIPProxy -17.599062; vs uniform budget -13.731940; vs fixed mean-logit -8.382970.

Beneficial 73170; harmful 1745195. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 37.6204 | -14.0671 | 75.391/62.178 | 70.478/44.658 | 29.864/22.944 | -507540 | -46063 |
| building | 78.2026 | 62.1090 | -16.0936 | 80.677/96.226 | 66.082/91.175 | 14.555/16.836 | -49312 | +231858 |
| low vegetation | 33.4221 | 17.0446 | -16.3775 | 72.470/38.283 | 66.512/18.645 | 10.133/5.377 | -301354 | -79113 |
| tree | 62.5813 | 23.9259 | -38.6554 | 90.654/66.897 | 95.602/24.192 | 17.673/6.060 | -818217 | -110820 |
| car | 11.6582 | 6.3026 | -5.3556 | 11.672/99.026 | 6.307/98.887 | 19.876/36.731 | -260 | +1348631 |
| clutter | 4.5652 | 3.8296 | -0.7356 | 7.745/10.007 | 5.559/10.960 | 7.899/12.051 | +4658 | +327532 |

## udd5/udd5 changes

Primary delta vs Geometry -11.068323; vs VIPProxy -10.420138; vs uniform budget -10.469716; vs fixed mean-logit -7.847232.

Beneficial 9344883; harmful 56425739. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 57.8390 | -24.6547 | 96.486/85.049 | 96.970/58.903 | 26.108/17.992 | -34070060 | -1637889 |
| building | 83.9035 | 78.5927 | -5.3108 | 88.360/94.329 | 81.338/95.883 | 41.911/46.279 | +2683090 | +16535716 |
| road | 45.7856 | 30.8961 | -14.8895 | 70.678/56.522 | 49.355/45.238 | 10.724/12.291 | -6657326 | +13552481 |
| vehicle | 10.0092 | 5.3002 | -4.7090 | 10.032/97.748 | 5.303/99.031 | 7.799/14.949 | +45204 | +31409677 |
| other | 30.5846 | 24.8069 | -5.7777 | 52.854/42.059 | 59.474/29.853 | 13.458/8.489 | -9081764 | -12779129 |

Geometry non_residual_mean_iou_percent: 55.5480.

Geometry_DonorBudget non_residual_mean_iou_percent: 43.1570.

## oem/oem changes

Primary delta vs Geometry -4.275331; vs VIPProxy -3.910908; vs uniform budget -3.965563; vs fixed mean-logit -3.514926.

Beneficial 257720; harmful 671063. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 10.425/11.975 | +0 | +118948 |
| rangeland | 47.7780 | 41.4300 | -6.3480 | 65.294/64.041 | 65.699/52.865 | 18.779/15.406 | -164278 | -94649 |
| developed space | 28.5294 | 29.3823 | 0.8529 | 61.562/34.713 | 57.261/37.636 | 12.630/14.722 | +50263 | +110334 |
| road | 40.8291 | 28.6853 | -12.1438 | 45.753/79.140 | 30.502/82.804 | 7.893/12.387 | +12835 | +332180 |
| tree | 56.0657 | 49.3924 | -6.6733 | 87.448/60.973 | 89.711/52.358 | 17.731/14.842 | -168168 | -53630 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.065/0.140 | +0 | +5744 |
| agriculture land | 79.1353 | 73.0416 | -6.0937 | 90.641/86.177 | 92.464/77.665 | 15.934/14.077 | -109510 | -33040 |
| building | 62.4971 | 58.7003 | -3.7968 | 65.584/92.995 | 63.219/89.145 | 16.544/16.452 | -34485 | +27456 |

## loveda/P changes

Primary delta vs Geometry -17.152155; vs VIPProxy -22.896009; vs uniform budget -15.619379; vs fixed mean-logit -10.801762.

Beneficial 103966; harmful 576785. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| building | 64.4987 | 50.9289 | -13.5698 | 65.367/97.982 | 51.139/99.201 | 0.980/1.268 | +365 | +12831 |
| road | 66.9348 | 36.0224 | -30.9124 | 67.652/98.440 | 36.163/98.929 | 10.361/19.479 | +1594 | +415831 |
| water | 81.1851 | 66.5087 | -14.6764 | 86.764/92.661 | 70.015/92.997 | 22.313/27.751 | +3213 | +245730 |
| barren | 24.3403 | 16.7829 | -7.5574 | 63.873/28.226 | 65.007/18.450 | 3.230/2.074 | -32714 | -20188 |
| tree | 53.7307 | 24.3881 | -29.3426 | 64.518/76.267 | 69.374/27.331 | 13.083/4.360 | -247950 | -151381 |
| farm | 86.2626 | 79.4083 | -6.8543 | 95.330/90.069 | 96.270/81.929 | 50.033/45.067 | -197327 | -30004 |

## loveda/D changes

Primary delta vs Geometry -10.482605; vs VIPProxy -7.693767; vs uniform budget -9.787561; vs fixed mean-logit -7.204971.

Beneficial 504134; harmful 1262012. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| background | 34.1983 | 26.6798 | -7.5185 | 65.750/41.611 | 64.166/31.351 | 28.248/21.809 | -378684 | -153808 |
| building | 31.1862 | 25.3063 | -5.8799 | 31.858/93.665 | 25.369/99.034 | 1.064/1.413 | +1607 | +27236 |
| road | 45.4818 | 20.5241 | -24.9577 | 45.835/98.335 | 20.574/98.830 | 8.458/18.937 | +1613 | +864898 |
| water | 61.9193 | 44.7962 | -17.1231 | 65.955/91.007 | 46.414/92.783 | 15.961/23.123 | +16990 | +575277 |
| barren | 14.5296 | 12.4496 | -2.0800 | 59.063/16.157 | 68.523/13.205 | 1.107/0.780 | -9878 | -17175 |
| tree | 26.8904 | 16.3301 | -10.5603 | 30.053/71.873 | 31.390/25.394 | 14.654/4.957 | -235500 | -566355 |
| farm | 54.9851 | 49.7265 | -5.2586 | 69.572/72.395 | 66.809/66.041 | 30.508/28.982 | -154026 | +27805 |

Geometry foreground_mean_iou_percent: 39.1654.

Geometry_DonorBudget foreground_mean_iou_percent: 28.1888.

## vaihingen/vaihingen changes

Primary delta vs Geometry -9.538103; vs VIPProxy -9.844970; vs uniform budget -8.350158; vs fixed mean-logit -6.064017.

Beneficial 109659; harmful 1074090. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 29.0079 | -20.1923 | 82.397/54.979 | 75.239/32.069 | 20.428/13.049 | -557418 | -28994 |
| building | 74.6610 | 68.7470 | -5.9140 | 75.500/98.533 | 69.442/98.565 | 28.129/30.593 | +542 | +195258 |
| low vegetation | 46.7115 | 30.5561 | -16.1554 | 92.073/48.669 | 91.001/31.508 | 13.453/8.812 | -347097 | -21726 |
| tree | 71.8329 | 69.8439 | -1.9890 | 82.551/84.692 | 83.467/81.058 | 21.440/20.295 | -60359 | -30657 |
| car | 8.7566 | 5.3170 | -3.4396 | 8.773/97.975 | 5.323/97.891 | 16.550/27.251 | -99 | +850550 |

## landcoverai/landcoverai changes

Primary delta vs Geometry -9.296678; vs VIPProxy -7.699151; vs uniform budget -8.773797; vs fixed mean-logit -6.885612.

Beneficial 14194; harmful 185059. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| background | 82.6090 | 75.5466 | -7.0624 | 96.651/85.044 | 96.176/77.886 | 59.623/54.874 | -101709 | +2127 |
| building | 34.9562 | 22.2265 | -12.7297 | 35.044/99.292 | 22.261/99.308 | 4.293/6.759 | +5 | +51711 |
| woodland | 78.3638 | 68.0103 | -10.3535 | 86.499/89.284 | 89.722/73.756 | 22.096/17.598 | -69709 | -24631 |
| water | 93.6635 | 84.8507 | -8.8128 | 93.664/100.000 | 84.851/100.000 | 8.831/9.748 | +0 | +19235 |
| road | 14.9318 | 7.4067 | -7.5251 | 15.629/77.002 | 7.551/79.498 | 5.158/11.021 | +548 | +122423 |

Geometry foreground_mean_iou_percent: 55.4788.

Geometry non_residual_mean_iou_percent: 55.4788.

Geometry_DonorBudget foreground_mean_iou_percent: 45.6236.

Geometry_DonorBudget non_residual_mean_iou_percent: 45.6236.

## flair1/flair1 changes

Primary delta vs Geometry -10.981805; vs VIPProxy -11.482634; vs uniform budget -10.387021; vs fixed mean-logit -6.012216.

Beneficial 64846; harmful 311730. These totals measure pixel correctness, not the mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | Predicted area percent (old/new) | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| building | 49.6979 | 32.1352 | -17.5627 | 50.726/96.082 | 32.735/94.609 | 13.599/20.750 | -2218 | +152122 |
| pervious surface | 57.0621 | 67.5342 | 10.4721 | 92.130/59.986 | 88.472/74.050 | 11.173/14.362 | +50591 | +16274 |
| impervious surface | 52.6509 | 37.5995 | -15.0514 | 62.624/76.777 | 42.904/75.254 | 20.085/28.735 | -5227 | +186567 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 3.441/0.463 | +0 | -62429 |
| water | 73.2328 | 38.8682 | -34.3646 | 77.143/93.526 | 39.069/98.695 | 5.378/11.206 | +4806 | +117365 |
| coniferous | 43.0233 | 41.4340 | -1.5893 | 61.711/58.690 | 64.184/53.895 | 0.535/0.472 | -565 | -747 |
| deciduous | 54.0229 | 32.9685 | -21.0544 | 78.972/63.099 | 87.732/34.562 | 13.600/6.706 | -101829 | -42706 |
| brushwood | 18.6136 | 1.5993 | -17.0143 | 23.421/47.556 | 4.594/2.395 | 9.987/2.564 | -46563 | -109039 |
| vineyard | undefined | undefined | undefined | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 40.3634 | -19.8695 | 94.317/62.501 | 97.539/40.779 | 21.229/13.393 | -145879 | -18382 |
| agricultural land | 18.7339 | 13.9682 | -4.7657 | 21.647/58.198 | 15.526/58.198 | 0.820/1.143 | +0 | +6775 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.154/0.206 | +0 | +1084 |

## Diagnostics and cost

Evaluator timing includes all controls; aggregate durations are allocation time, not kernel-active GPU time.

| Dataset | Max shard seconds | Sum shard seconds | Peak MiB | Mean iterations | Marginal error | Incoming L1 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 202.392 | 202.392 | 3925.920 | 80.801 | 0.00000859 | 1.288617 |
| potsdam | 15.371 | 15.371 | 3923.073 | 84.333 | 0.00000883 | 1.312319 |
| udd5 | 225.194 | 885.241 | 3924.001 | 83.249 | 0.00000861 | 1.304611 |
| oem | 14.802 | 14.802 | 3925.805 | 71.194 | 0.00000818 | 1.245687 |
| loveda | 18.836 | 18.836 | 3926.156 | 71.806 | 0.00000859 | 1.266778 |
| vaihingen | 14.128 | 14.128 | 3923.383 | 75.194 | 0.00000855 | 1.338670 |
| landcoverai | 2.028 | 2.028 | 3914.864 | 72.000 | 0.00000875 | 1.216366 |
| flair1 | 2.626 | 2.626 | 3915.958 | 73.250 | 0.00000894 | 1.251445 |

## Native donor concentration

These averages summarize per-tile, per-block/head maximum donor fractions. They do not identify a semantic class or prove concentration alone caused the errors.

| Dataset/protocol | Geometry max donor percent | Native-budget max donor percent | Ratio | Harmful/beneficial changes |
| --- | ---: | ---: | ---: | ---: |
| vdd/vdd | 0.2174 | 8.5322 | 39.2499 | 2.9627 |
| potsdam/potsdam | 0.2231 | 8.9207 | 39.9924 | 23.8512 |
| udd5/udd5 | 0.2209 | 8.2382 | 37.2929 | 6.0381 |
| oem/oem | 0.2246 | 7.9228 | 35.2783 | 2.6038 |
| loveda/P | 0.2132 | 7.8061 | 36.6149 | 5.5478 |
| loveda/D | 0.2132 | 7.8061 | 36.6149 | 2.5033 |
| vaihingen/vaihingen | 0.2187 | 10.3226 | 47.1907 | 9.7948 |
| landcoverai/landcoverai | 0.2001 | 7.6525 | 38.2408 | 13.0378 |
| flair1/flair1 | 0.2274 | 7.4290 | 32.6740 | 4.8072 |

Interpretation: the prescribed native head roles are not verified semantic ownership. A converged marginal projection may still replace Geometry's useful spatial distribution with concentrated, semantically unsuitable evidence. Uniform budgets provide the matched control for whether marginal projection itself or the native budget is damaging. Neither uniform nor native budgets guarantee correct class competition; use the exact per-class TP/FP/area results above. No parameters are selected from these labeled audits.

FP32/bf16 actual-checkpoint Geometry-identity and native-cache errors are exactly0. Target masks were not loaded. The initial128-step numerical failure is retained; the equivalent FP32 matrix-vector solver was corrected before labeled evaluation.

Window medians: Geometry 0.024455s; primary 0.075483s (3.0866x). Three synchronized single-window repeats, original prepare-image pipeline. No control heads in primary timing. Not full-image end-to-end latency.

Completion, numerical constraints and a changed architecture do not prove an original CVPR contribution. The required outcome remains a useful globally fixed coupling, strong nearest-method comparisons, mechanism evidence, independent validation and transparent cost.
