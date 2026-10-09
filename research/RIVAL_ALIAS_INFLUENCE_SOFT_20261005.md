# Fine-risk influence-aware soft weights

Same64 developed windows, original Geometry/finite VIP wide observer/fine source/all20/reconstruction. Follow-up motivated by the previous labeled64 pilot; not independent validation. New source no-admission scores and per-image predictions replay exactly. Scores persisted before masks; no parameter or domain fitting.

The new rule is `w=(1-r)*(1-pi)/(1-pi+r*pi)`, where r is unchanged fine contradiction risk and pi the actual original crop alias responsibility. Canonical/self/zero-risk weights stay1. This is an influence-aware attenuation heuristic, not a calibrated Bayesian posterior.

| Dataset/protocol | NoAdmission_Exact | RivalFineHard_Exact | FineSoft_Weighted | FineSoft_Excess | FineInfluence_Weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 53.7470 | 54.3441 | 54.2567 | 54.3308 | 54.3098 |
| potsdam/potsdam | 38.9203 | 40.6173 | 39.3772 | 39.4361 | 39.4145 |
| udd5/udd5 | 28.1758 | 34.0394 | 29.7268 | 29.5160 | 29.8675 |
| oem/oem | 39.0232 | 39.7906 | 39.2104 | 39.2784 | 39.2367 |
| loveda/P | 50.7066 | 52.5879 | 51.9329 | 51.8737 | 51.8209 |
| loveda/D | 30.7360 | 37.2156 | 32.7899 | 33.3658 | 32.9018 |
| vaihingen/vaihingen | 51.8270 | 52.9446 | 52.1839 | 52.2902 | 52.2183 |
| landcoverai/landcoverai | 66.9060 | 67.4834 | 67.1432 | 67.1392 | 67.1583 |
| flair1/flair1 | 33.6084 | 34.0624 | 33.7562 | 33.8539 | 33.7727 |
| Eight-domain mean, LoveDA D once | 42.8679 | 45.0622 | 43.5555 | 43.6513 | 43.6100 |

## Independent Window-Context Cost

Same first images and scope, warmed synchronized median of three executions. Controls were measured in the preceding session, not interleaved with the follow-up; small timing differences are not conclusive.

| Dataset | Influence seconds | Ratio vs cached hard | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| vdd | 0.399833 | 1.0674 | 5565.398 |
| potsdam | 0.296867 | 1.0928 | 5563.513 |
| udd5 | 0.345587 | 1.2055 | 5556.519 |
| oem | 0.287599 | 1.0456 | 5573.023 |
| loveda | 0.353353 | 1.0936 | 5593.403 |
| vaihingen | 0.281558 | 1.0304 | 5559.325 |
| landcoverai | 0.278357 | 1.0691 | 5559.325 |
| flair1 | 0.309797 | 1.0907 | 5797.511 |

## Per-Class Outcomes

| Dataset/protocol | Class | New IoU | Delta vs hard | Precision | Recall | Area |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 59.6020 | 2.3595 | 81.2869 | 69.0806 | 43.3890 |
| vdd/vdd | wall | 60.4088 | 1.9698 | 66.3379 | 87.1117 | 10.4464 |
| vdd/vdd | road | 23.1539 | 1.8368 | 23.2691 | 97.9068 | 13.9651 |
| vdd/vdd | vegetation | 46.1930 | -4.2366 | 94.9463 | 47.3574 | 11.4894 |
| vdd/vdd | vehicle | 48.8360 | 6.6545 | 48.8360 | 100.0000 | 0.7333 |
| vdd/vdd | roof | 87.9969 | -1.7320 | 88.4567 | 99.4128 | 6.7717 |
| vdd/vdd | water | 53.9779 | -7.0925 | 56.9611 | 91.1556 | 13.2051 |
| potsdam/potsdam | impervious surface | 66.2427 | -0.0019 | 85.6912 | 74.4813 | 35.5291 |
| potsdam/potsdam | building | 71.3380 | -1.1895 | 75.2686 | 93.1791 | 19.9594 |
| potsdam/potsdam | low vegetation | 15.7833 | -3.8900 | 80.5059 | 16.4105 | 3.5348 |
| potsdam/potsdam | tree | 56.1638 | -1.7752 | 91.9991 | 59.0479 | 11.3242 |
| potsdam/potsdam | car | 24.7792 | -0.1290 | 24.8321 | 99.1476 | 14.3164 |
| potsdam/potsdam | clutter | 2.1802 | -0.2309 | 2.7501 | 9.5192 | 15.3362 |
| udd5/udd5 | vegetation | 52.2686 | -15.4276 | 92.4265 | 54.6074 | 1.8240 |
| udd5/udd5 | building | 84.2511 | -1.3476 | 84.3896 | 99.8056 | 85.0645 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6497 |
| udd5/udd5 | vehicle | 6.0324 | -0.9780 | 6.0325 | 99.9773 | 6.9607 |
| udd5/udd5 | other | 6.7856 | -3.1061 | 29.8242 | 8.0749 | 5.5011 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2593 |
| oem/oem | rangeland | 49.5790 | -0.1143 | 70.5733 | 62.4994 | 12.8820 |
| oem/oem | developed space | 23.7250 | -0.1299 | 39.3721 | 37.3819 | 18.6070 |
| oem/oem | road | 54.6603 | -0.2622 | 68.1219 | 73.4471 | 6.7787 |
| oem/oem | tree | 56.0719 | -0.9118 | 91.5783 | 59.1204 | 17.0551 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | agriculture land | 81.6893 | -0.3176 | 82.0596 | 99.4507 | 25.7095 |
| oem/oem | building | 48.1684 | -2.6952 | 68.6730 | 61.7332 | 10.7083 |
| loveda/P | building | 62.5995 | 8.6280 | 77.1242 | 76.8730 | 0.0264 |
| loveda/P | road | 55.3999 | -3.1447 | 56.4448 | 96.7663 | 14.2672 |
| loveda/P | water | 62.4527 | -0.0962 | 68.0679 | 88.3322 | 19.0952 |
| loveda/P | barren | 3.9335 | 1.5869 | 90.4618 | 3.9498 | 0.4841 |
| loveda/P | tree | 43.3947 | -9.8179 | 99.2507 | 43.5373 | 7.2225 |
| loveda/P | farm | 83.1452 | -1.7579 | 83.4598 | 99.5487 | 58.9046 |
| loveda/D | background | 11.4855 | -14.4880 | 78.7256 | 11.8534 | 6.7382 |
| loveda/D | building | 35.8434 | -2.2452 | 40.0000 | 77.5244 | 0.0284 |
| loveda/D | road | 45.9772 | -2.6107 | 46.6993 | 96.7466 | 9.5253 |
| loveda/D | water | 54.6203 | -0.0885 | 58.8774 | 88.3099 | 12.1933 |
| loveda/D | barren | 2.8697 | 1.0932 | 81.8443 | 2.8880 | 0.2162 |
| loveda/D | tree | 39.3902 | -6.5993 | 96.8436 | 39.9025 | 3.7480 |
| loveda/D | farm | 40.1261 | -5.2580 | 40.2016 | 99.5341 | 67.5508 |
| vaihingen/vaihingen | impervious surface | 57.8315 | -0.9788 | 73.9651 | 72.6126 | 26.8171 |
| vaihingen/vaihingen | building | 66.1564 | -0.9200 | 66.2918 | 99.6924 | 31.0252 |
| vaihingen/vaihingen | low vegetation | 43.9485 | -1.0017 | 96.1938 | 44.7263 | 13.7480 |
| vaihingen/vaihingen | tree | 67.3176 | -0.2032 | 78.9104 | 82.0860 | 21.4692 |
| vaihingen/vaihingen | car | 25.8376 | -0.5278 | 25.9938 | 97.7270 | 6.9406 |
| landcoverai/landcoverai | background | 87.6772 | -0.3056 | 94.1545 | 92.7245 | 66.7308 |
| landcoverai/landcoverai | building | 44.5079 | -0.0511 | 44.8531 | 98.3004 | 3.3204 |
| landcoverai/landcoverai | woodland | 79.3915 | -1.1656 | 94.7708 | 83.0286 | 18.7545 |
| landcoverai/landcoverai | water | 97.6134 | 0.0000 | 97.6134 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | road | 26.6015 | -0.1030 | 29.0967 | 75.6218 | 2.7207 |
| flair1/flair1 | building | 56.9784 | -0.1968 | 58.2423 | 96.3310 | 11.8747 |
| flair1/flair1 | pervious surface | 47.0106 | 0.3832 | 91.7436 | 49.0873 | 9.1813 |
| flair1/flair1 | impervious surface | 52.2448 | -0.3298 | 60.0727 | 80.0372 | 21.8269 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1577 |
| flair1/flair1 | water | 62.1774 | -1.5104 | 64.0498 | 95.5093 | 6.6149 |
| flair1/flair1 | coniferous | 43.0807 | 0.3036 | 64.4169 | 56.5343 | 0.4933 |
| flair1/flair1 | deciduous | 59.9446 | -0.4362 | 78.8329 | 71.4439 | 15.4262 |
| flair1/flair1 | brushwood | 13.2547 | -1.6288 | 27.2887 | 20.4919 | 3.6934 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0041 |
| flair1/flair1 | herbaceous vegetation | 57.5432 | 0.0187 | 90.9967 | 61.0172 | 21.4809 |
| flair1/flair1 | agricultural land | 13.0377 | -0.0798 | 14.0220 | 65.0031 | 1.4135 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8329 |

All alternatives and historical files are retained; no automatic control promotion or full20k rerun.
