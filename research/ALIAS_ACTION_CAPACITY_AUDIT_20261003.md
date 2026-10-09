# Alias Action Capacity: Verified Window Audit

64 unique fixed images, eight per domain, ONE top-left512 window/image. Wide view is from the original full image. Not complete-image or full-dataset mIoU. LoveDA P/D share images; means count D once. Frozen network, all20 aliases, same original views/salience/temperature/objective. Labels audit only; no selector fitting. Constructed stress, not fresh LLM words.

Pointwise envelopes permit different donor choices for different output locations. They are not one joint oracle prediction. LabelPolicy controls are feasible privileged-label policies, not optimal solutions or deployable methods.

## Paired Metric Correction

Historical standard mIoU averages classes with positive union separately for each prediction. An absent class can disappear from that denominator when its last false prediction is removed. For paired action comparisons below, freeze the Anchored_Exact union-positive class set separately per dataset/protocol/scenario, before any action; a scored zero-union class has IoU=0. All saved predictions and raw standard metrics remain unchanged. This corrects the comparison, not the model, and introduces no fitted parameter. Action AUC and counts in the main tables use the fixed set; raw-standard counterparts remain in audit_summary.json.

## clean

### Fixed-Class Paired Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 53.746954 | 53.739421 | 66.865322 | 67.041700 |
| potsdam/potsdam | 35.703546 | 38.920232 | 38.920258 | 39.141453 | 46.857408 | 46.921657 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.175811 | 28.170638 | 46.267441 | 46.350797 |
| oem/oem | 39.820583 | 39.023075 | 39.023152 | 39.157933 | 44.335737 | 44.520077 |
| loveda/P | 49.539928 | 50.706552 | 50.706552 | 50.935333 | 57.079016 | 57.740602 |
| loveda/D | 33.877920 | 30.736049 | 30.736030 | 30.729095 | 41.590882 | 42.256252 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.826962 | 51.588246 | 56.757742 | 56.979040 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.906020 | 66.922266 | 69.779926 | 69.949192 |
| flair1/flair1 | 35.605867 | 33.608426 | 33.608404 | 33.476562 | 38.885607 | 39.127864 |
| Equal-domain mean | 40.531033 | 42.867941 | 42.867949 | 42.865702 | 51.417508 | 51.643322 |

### Raw Standard Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 53.746954 | 53.739421 | 66.865322 | 67.041700 |
| potsdam/potsdam | 35.703546 | 38.920232 | 38.920258 | 39.141453 | 46.857408 | 46.921657 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.175811 | 28.170638 | 46.267441 | 46.350797 |
| oem/oem | 39.820583 | 39.023075 | 39.023152 | 39.157933 | 44.335737 | 44.520077 |
| loveda/P | 49.539928 | 50.706552 | 50.706552 | 50.935333 | 57.079016 | 57.740602 |
| loveda/D | 33.877920 | 30.736049 | 30.736030 | 30.729095 | 41.590882 | 42.256252 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.826962 | 51.588246 | 56.757742 | 56.979040 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.906020 | 66.922266 | 69.779926 | 69.949192 |
| flair1/flair1 | 38.842764 | 33.608426 | 33.608404 | 33.476562 | 38.885607 | 39.127864 |
| Equal-domain mean | 40.935645 | 42.867941 | 42.867949 | 42.865702 | 51.417508 | 51.643322 |

| Dataset/protocol | Wrong patch centres | Protected impossible | Unrestricted impossible | Protected possible | Source action AUC | Useful / harmful / neutral noncanonical actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vdd/vdd | 2556 | 1544 | 1533 | 1012 | 0.4732 | 61 / 41 / 31 |
| potsdam/potsdam | 3074 | 2076 | 2068 | 998 | 0.4905 | 48 / 66 / 0 |
| udd5/udd5 | 2050 | 1372 | 1370 | 678 | 0.4500 | 28 / 25 / 42 |
| oem/oem | 2692 | 1586 | 1560 | 1106 | 0.4815 | 59 / 70 / 23 |
| loveda/P | 1032 | 816 | 806 | 216 | 0.4647 | 37 / 57 / 20 |
| loveda/D | 4433 | 3095 | 3063 | 1338 | 0.3924 | 57 / 56 / 20 |
| vaihingen/vaihingen | 2232 | 1765 | 1706 | 467 | 0.4069 | 47 / 48 / 0 |
| landcoverai/landcoverai | 752 | 431 | 424 | 321 | 0.4021 | 59 / 29 / 7 |
| flair1/flair1 | 2790 | 1816 | 1768 | 974 | 0.2900 | 104 / 107 / 17 |

## wrong_parent

### Fixed-Class Paired Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 47.408779 | 47.408779 | 47.501507 | 60.421808 | 60.580068 |
| potsdam/potsdam | 35.703546 | 36.070708 | 36.070589 | 36.733768 | 46.079305 | 46.113952 |
| udd5/udd5 | 30.500979 | 27.616556 | 27.616556 | 27.923834 | 47.531653 | 47.535231 |
| oem/oem | 39.820583 | 40.053295 | 40.053288 | 40.175834 | 45.779213 | 45.961894 |
| loveda/P | 49.539928 | 49.989695 | 49.989695 | 49.356612 | 58.431692 | 59.965294 |
| loveda/D | 33.877920 | 30.333417 | 30.333514 | 30.086101 | 41.300419 | 42.200454 |
| vaihingen/vaihingen | 49.365065 | 53.060099 | 53.060153 | 52.655475 | 59.570662 | 59.851382 |
| landcoverai/landcoverai | 60.904853 | 68.123522 | 68.123522 | 68.091943 | 72.715311 | 73.121059 |
| flair1/flair1 | 38.842764 | 36.269320 | 36.269320 | 36.259007 | 43.995074 | 44.316505 |
| Equal-domain mean | 40.935645 | 42.366962 | 42.366965 | 42.428434 | 52.174181 | 52.460068 |

### Raw Standard Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 47.408779 | 47.408779 | 47.501507 | 60.421808 | 60.580068 |
| potsdam/potsdam | 35.703546 | 36.070708 | 36.070589 | 36.733768 | 46.079305 | 46.113952 |
| udd5/udd5 | 30.500979 | 27.616556 | 27.616556 | 27.923834 | 47.531653 | 47.535231 |
| oem/oem | 39.820583 | 40.053295 | 40.053288 | 40.175834 | 45.779213 | 45.961894 |
| loveda/P | 49.539928 | 49.989695 | 49.989695 | 49.356612 | 58.431692 | 59.965294 |
| loveda/D | 33.877920 | 30.333417 | 30.333514 | 30.086101 | 41.300419 | 42.200454 |
| vaihingen/vaihingen | 49.365065 | 53.060099 | 53.060153 | 52.655475 | 59.570662 | 59.851382 |
| landcoverai/landcoverai | 60.904853 | 68.123522 | 68.123522 | 68.091943 | 72.715311 | 73.121059 |
| flair1/flair1 | 38.842764 | 36.269320 | 36.269320 | 36.259007 | 43.995074 | 48.748156 |
| Equal-domain mean | 40.935645 | 42.366962 | 42.366965 | 42.428434 | 52.174181 | 53.014025 |

| Dataset/protocol | Wrong patch centres | Protected impossible | Unrestricted impossible | Protected possible | Source action AUC | Useful / harmful / neutral noncanonical actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vdd/vdd | 2852 | 1822 | 1796 | 1030 | 0.5171 | 63 / 45 / 25 |
| potsdam/potsdam | 3334 | 2125 | 2114 | 1209 | 0.5691 | 51 / 63 / 0 |
| udd5/udd5 | 2178 | 1562 | 1559 | 616 | 0.3897 | 35 / 18 / 42 |
| oem/oem | 2516 | 1414 | 1387 | 1102 | 0.5358 | 66 / 75 / 11 |
| loveda/P | 945 | 719 | 703 | 226 | 0.4646 | 39 / 58 / 17 |
| loveda/D | 4515 | 3093 | 3058 | 1422 | 0.4983 | 50 / 71 / 12 |
| vaihingen/vaihingen | 2182 | 1432 | 1365 | 750 | 0.3793 | 42 / 51 / 2 |
| landcoverai/landcoverai | 698 | 339 | 322 | 359 | 0.4435 | 64 / 30 / 1 |
| flair1/flair1 | 2769 | 1578 | 1539 | 1191 | 0.3471 | 96 / 105 / 27 |

## paraphrase

### Fixed-Class Paired Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 54.530706 | 54.530664 | 54.480856 | 69.536709 | 69.549230 |
| potsdam/potsdam | 35.703546 | 41.663326 | 41.663326 | 41.734168 | 50.330556 | 50.354656 |
| udd5/udd5 | 30.500979 | 31.169271 | 31.169271 | 31.157806 | 46.830565 | 46.831121 |
| oem/oem | 39.820583 | 39.619434 | 39.619434 | 39.728584 | 45.162035 | 45.332967 |
| loveda/P | 49.539928 | 48.735654 | 48.735654 | 48.764997 | 57.087992 | 57.481838 |
| loveda/D | 33.877920 | 29.514071 | 29.513994 | 29.500178 | 40.999885 | 41.355192 |
| vaihingen/vaihingen | 49.365065 | 52.518371 | 52.518481 | 52.305387 | 58.161528 | 58.258096 |
| landcoverai/landcoverai | 60.904853 | 65.629553 | 65.629553 | 65.631121 | 69.389701 | 69.444667 |
| flair1/flair1 | 35.605867 | 34.530135 | 34.530135 | 34.416924 | 39.940879 | 40.099932 |
| Equal-domain mean | 40.531033 | 43.646858 | 43.646857 | 43.619378 | 52.543982 | 52.653233 |

### Raw Standard Metrics

| Dataset/protocol | Geometry | Anchored_CG | Anchored_Exact | ClassRelativeReject_CG | LabelPolicy_Protected | LabelPolicy_Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 54.530706 | 54.530664 | 54.480856 | 69.536709 | 69.549230 |
| potsdam/potsdam | 35.703546 | 41.663326 | 41.663326 | 41.734168 | 50.330556 | 50.354656 |
| udd5/udd5 | 30.500979 | 31.169271 | 31.169271 | 31.157806 | 46.830565 | 46.831121 |
| oem/oem | 39.820583 | 39.619434 | 39.619434 | 39.728584 | 45.162035 | 45.332967 |
| loveda/P | 49.539928 | 48.735654 | 48.735654 | 48.764997 | 57.087992 | 57.481838 |
| loveda/D | 33.877920 | 29.514071 | 29.513994 | 29.500178 | 40.999885 | 41.355192 |
| vaihingen/vaihingen | 49.365065 | 52.518371 | 52.518481 | 52.305387 | 58.161528 | 58.258096 |
| landcoverai/landcoverai | 60.904853 | 65.629553 | 65.629553 | 65.631121 | 69.389701 | 69.444667 |
| flair1/flair1 | 38.842764 | 34.530135 | 34.530135 | 34.416924 | 43.571868 | 43.745380 |
| Equal-domain mean | 40.935645 | 43.646858 | 43.646857 | 43.619378 | 52.997856 | 53.108914 |

| Dataset/protocol | Wrong patch centres | Protected impossible | Unrestricted impossible | Protected possible | Source action AUC | Useful / harmful / neutral noncanonical actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vdd/vdd | 2557 | 1374 | 1370 | 1183 | 0.4793 | 59 / 43 / 31 |
| potsdam/potsdam | 2860 | 1784 | 1776 | 1076 | 0.4825 | 53 / 61 / 0 |
| udd5/udd5 | 1908 | 1253 | 1253 | 655 | 0.3416 | 32 / 22 / 41 |
| oem/oem | 2703 | 1520 | 1493 | 1183 | 0.4875 | 60 / 68 / 24 |
| loveda/P | 1055 | 802 | 794 | 253 | 0.4702 | 32 / 64 / 18 |
| loveda/D | 4515 | 3164 | 3147 | 1351 | 0.3934 | 62 / 54 / 17 |
| vaihingen/vaihingen | 2180 | 1557 | 1501 | 623 | 0.3701 | 48 / 47 / 0 |
| landcoverai/landcoverai | 794 | 424 | 418 | 370 | 0.3236 | 55 / 30 / 10 |
| flair1/flair1 | 2694 | 1684 | 1652 | 1010 | 0.3295 | 112 / 103 / 13 |

## Clean Fixed-Readout Action Examples

Each row attenuates ONE alias across all eight observed windows. These labeled audit findings are scene-dependent action utilities, not a blacklist or a selector to deploy.

| Dataset / class / alias | Raw delta pp | Fixed-set delta pp | Own TP change | Own FP change | Frozen risk | Largest rival FP reduction |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| udd5 / vehicle / cars seen from above | +2.261230 | +2.261230 | -1 | -77653 | 0.000000 | other: -47391 |
| potsdam / car / traffic vehicle | +0.640937 | +0.640937 | -27 | -20702 | 0.000000 | impervious surface: -20112 |
| vdd / vehicle / vehicles | +0.570833 | +0.570833 | -9 | -1107 | 0.000000 | other: -1077 |
| potsdam / impervious surface / road pavement | -0.772698 | -0.772698 | -26864 | -1183 | 0.103774 | tree: -982 |
| vaihingen / impervious surface / parking pavement | -1.209770 | -1.209770 | -22621 | -6103 | 0.076618 | low vegetation: -4598 |
| loveda / farm / farmland | +1.411040 | +1.411040 | -271 | -57521 | 0.000000 | background: -47711 |
| flair1 / vineyard / vineyards | +3.054150 | -0.001063 | +0 | -34 | 0.000000 | deciduous: -32 |

## Interpretation

Protected label policy improves all eight clean domains; fixed-set mean gain is 8.549559pp. This proves some feasible label-assisted choices help in these windows even with canonical protection. It is not an optimal upper bound or evidence that a label-free source can recover this gain. Unrestricted protection removal adds only 0.225814pp on clean. 

The current frozen source can assign zero risk to beneficial suppression of legitimate vehicle/farm phrases, while assigning positive risk to suppression that destroys pavement true coverage. All eight clean global-action AUCs are below0.5; this compares one frozen aggregated risk with window-global actions and is not a querywise impossibility result. Increasing its strength cannot reach its zero-risk harmful activations. A different image-conditioned semantic witness is needed.

Class-level envelopes below are grouped by TRUE class. Zero missed vehicle centres does not mean zero false vehicle activation; the latter belongs to other true-class rows. Pointwise possible locations cannot be combined into one achievable prediction without checking joint compatibility.

## Clean Class-Level Capacity

| Dataset/protocol/class | Wrong centres | Protected impossible | Unrestricted impossible | Protected possible |
| --- | ---: | ---: | ---: | ---: |
| vdd/vdd/other | 1347 | 934 | 933 | 413 |
| vdd/vdd/wall | 88 | 40 | 40 | 48 |
| vdd/vdd/road | 8 | 4 | 4 | 4 |
| vdd/vdd/vegetation | 1037 | 523 | 515 | 514 |
| vdd/vdd/vehicle | 0 | 0 | 0 | 0 |
| vdd/vdd/roof | 3 | 2 | 2 | 1 |
| vdd/vdd/water | 73 | 41 | 39 | 32 |
| potsdam/potsdam/impervious surface | 859 | 550 | 547 | 309 |
| potsdam/potsdam/building | 97 | 61 | 60 | 36 |
| potsdam/potsdam/low vegetation | 1188 | 791 | 790 | 397 |
| potsdam/potsdam/tree | 598 | 377 | 375 | 221 |
| potsdam/potsdam/car | 1 | 0 | 0 | 1 |
| potsdam/potsdam/clutter | 331 | 297 | 296 | 34 |
| udd5/udd5/vegetation | 130 | 83 | 83 | 47 |
| udd5/udd5/building | 17 | 0 | 0 | 17 |
| udd5/udd5/road | 341 | 157 | 156 | 184 |
| udd5/udd5/vehicle | 0 | 0 | 0 | 0 |
| udd5/udd5/other | 1562 | 1132 | 1131 | 430 |
| oem/oem/bareland | 0 | 0 | 0 | 0 |
| oem/oem/rangeland | 425 | 334 | 330 | 91 |
| oem/oem/developed space | 954 | 317 | 315 | 637 |
| oem/oem/road | 128 | 93 | 93 | 35 |
| oem/oem/tree | 824 | 697 | 677 | 127 |
| oem/oem/water | 3 | 3 | 3 | 0 |
| oem/oem/agriculture land | 14 | 11 | 11 | 3 |
| oem/oem/building | 344 | 131 | 131 | 213 |
| loveda/P/building | 0 | 0 | 0 | 0 |
| loveda/P/road | 11 | 9 | 9 | 2 |
| loveda/P/water | 77 | 54 | 54 | 23 |
| loveda/P/barren | 485 | 448 | 447 | 37 |
| loveda/P/tree | 451 | 301 | 292 | 150 |
| loveda/P/farm | 8 | 4 | 4 | 4 |
| loveda/D/background | 3360 | 2274 | 2251 | 1086 |
| loveda/D/building | 0 | 0 | 0 | 0 |
| loveda/D/road | 11 | 9 | 9 | 2 |
| loveda/D/water | 77 | 54 | 54 | 23 |
| loveda/D/barren | 489 | 452 | 451 | 37 |
| loveda/D/tree | 487 | 302 | 294 | 185 |
| loveda/D/farm | 9 | 4 | 4 | 5 |
| vaihingen/vaihingen/impervious surface | 615 | 426 | 417 | 189 |
| vaihingen/vaihingen/building | 4 | 3 | 3 | 1 |
| vaihingen/vaihingen/low vegetation | 1323 | 1122 | 1080 | 201 |
| vaihingen/vaihingen/tree | 288 | 212 | 204 | 76 |
| vaihingen/vaihingen/car | 2 | 2 | 2 | 0 |
| landcoverai/landcoverai/background | 413 | 276 | 269 | 137 |
| landcoverai/landcoverai/building | 2 | 2 | 2 | 0 |
| landcoverai/landcoverai/woodland | 316 | 134 | 134 | 182 |
| landcoverai/landcoverai/water | 0 | 0 | 0 | 0 |
| landcoverai/landcoverai/road | 21 | 19 | 19 | 2 |
| flair1/flair1/building | 22 | 2 | 2 | 20 |
| flair1/flair1/pervious surface | 704 | 397 | 391 | 307 |
| flair1/flair1/impervious surface | 288 | 197 | 191 | 91 |
| flair1/flair1/bare soil | 0 | 0 | 0 | 0 |
| flair1/flair1/water | 14 | 6 | 6 | 8 |
| flair1/flair1/coniferous | 18 | 18 | 18 | 0 |
| flair1/flair1/deciduous | 392 | 222 | 216 | 170 |
| flair1/flair1/brushwood | 326 | 225 | 219 | 101 |
| flair1/flair1/vineyard | 0 | 0 | 0 | 0 |
| flair1/flair1/herbaceous vegetation | 1017 | 746 | 722 | 271 |
| flair1/flair1/agricultural land | 9 | 3 | 3 | 6 |
| flair1/flair1/plowed land | 0 | 0 | 0 | 0 |

## Numerical And Shared Cost

Suite wall seconds: 146.2614

| Dataset | Worker seconds | Peak allocated MiB | Maximum CG/exact score error | Changed patch-centre controls |
| --- | ---: | ---: | ---: | ---: |
| vdd | 10.9564 | 5634.6118 | 9.6523396e-06 | 0 |
| potsdam | 12.0170 | 5627.6919 | 1.00005959e-05 | 0 |
| udd5 | 9.4318 | 5607.6851 | 1.5489349e-05 | 0 |
| oem | 12.4238 | 5657.2280 | 1.25035261e-05 | 0 |
| loveda | 20.1079 | 5721.2407 | 1.28263982e-05 | 0 |
| vaihingen | 11.4034 | 5615.5835 | 9.18719188e-06 | 0 |
| landcoverai | 11.4310 | 5615.5835 | 1.04912679e-05 | 0 |
| flair1 | 13.3350 | 5707.9629 | 8.08023904e-06 | 0 |

Shared audit cost, not standalone inference latency. Exact-system numerical audit is not an inference-solver change. All single-alias/group confusion matrices and class TP/FP changes are in audit_summary.json/per_image_audit.npz. Pre-mask numerical caches remain under the matching remote root. AUC ranks global-window attenuation actions using the frozen mean risk score; it does not validate querywise reliability or a transferable threshold.
