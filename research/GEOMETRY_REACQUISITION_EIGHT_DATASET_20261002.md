# Geometry-conditioned feature re-acquisition: full eight-domain trial

2026-10-02. All eight datasets completed under one frozen last-four-block replay rule, using physical A800 GPUs4-7. These are development results, not untouched validation.

The primary rereads cached native backbone donors with evolving real queries, retaining native special contributions and patch mass; the same original Geometry relation is used in the final head. No aliases were deleted, weights trained or dataset-specific rules selected.

## Verified complete performance

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | Reacquired_Native | MeanLogit_Reacquired | Reacquired_Geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 1669 | 64.9557 | 66.6890 | 67.9118 | 56.5549 | 63.1512 | 62.7304 |
| loveda/D | 1669 | 42.6589 | 40.1367 | 40.7530 | 37.0916 | 41.8632 | 41.1009 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 44.5046 | 48.8428 | 49.5202 |
| oem/oem | 384 | 44.6097 | 44.6481 | 45.7178 | 40.1271 | 44.1750 | 45.5864 |
| vdd/vdd | 80 | 38.8511 | 39.6247 | 39.5620 | 35.2647 | 38.2730 | 38.5793 |
| potsdam/potsdam | 504 | 40.6892 | 46.1823 | 43.3597 | 27.9514 | 35.2086 | 43.4517 |
| vaihingen/vaihingen | 113 | 48.8009 | 49.5488 | 49.2100 | 42.5662 | 46.8137 | 47.6738 |
| landcoverai/landcoverai | 1602 | 59.2340 | 62.9059 | 60.3542 | 54.4713 | 58.7696 | 58.7913 |
| flair1/flair1 | 15700 | 43.8071 | 44.4419 | 46.2372 | 39.9925 | 43.4590 | 43.5633 |

LoveDA P/D share1669 images. The eight-domain equal mean counts D once.

| Method | Eight-domain mean |
| --- | ---: |
| Geometry | 46.1508 |
| SCLIP_Two | 47.2215 |
| VIPProxy_Two | 46.8876 |
| Reacquired_Native | 40.2462 |
| MeanLogit_Reacquired | 44.6756 |
| Reacquired_Geometry | 46.0334 |

Primary domain wins: Geometry 2/8, SCLIP_Two 2/8, VIPProxy_Two 2/8, Reacquired_Native 8/8, MeanLogit_Reacquired 7/8.

## Same-observation coupling and nearest controls

Differences are primary minus control, in percentage points. Intervals are exploratory paired95% bootstrap intervals with2000 replicates by default; scene/acquisition grouping follows the prior audit. LoveDA/UDD5/VDD have image-level proxies, and intervals do not correct for multiple comparisons.

| Dataset/protocol | Comparator | Delta | Paired95% interval |
| --- | --- | ---: | --- |
| loveda/P | Geometry | -2.2253 | [-2.4457, -2.0137] |
| loveda/P | SCLIP_Two | -3.9586 | [-4.4951, -3.4287] |
| loveda/P | VIPProxy_Two | -5.1814 | [-5.5563, -4.8057] |
| loveda/P | Reacquired_Native | +6.1755 | [+5.6045, +6.7792] |
| loveda/P | MeanLogit_Reacquired | -0.4208 | [-0.7145, -0.1164] |
| loveda/D | Geometry | -1.5580 | [-1.7111, -1.4048] |
| loveda/D | SCLIP_Two | +0.9642 | [+0.5410, +1.3889] |
| loveda/D | VIPProxy_Two | +0.3479 | [+0.0013, +0.7004] |
| loveda/D | Reacquired_Native | +4.0093 | [+3.6036, +4.4395] |
| loveda/D | MeanLogit_Reacquired | -0.7623 | [-0.9561, -0.5514] |
| udd5/udd5 | Geometry | -1.0351 | [-1.6812, -0.3839] |
| udd5/udd5 | SCLIP_Two | -0.7635 | [-2.1534, +0.9669] |
| udd5/udd5 | VIPProxy_Two | -0.3869 | [-1.5851, +1.1480] |
| udd5/udd5 | Reacquired_Native | +5.0156 | [+3.8998, +6.3562] |
| udd5/udd5 | MeanLogit_Reacquired | +0.6774 | [+0.0105, +1.4733] |
| oem/oem | Geometry | +0.9767 | [+0.5866, +1.3723] |
| oem/oem | SCLIP_Two | +0.9383 | [-0.0632, +2.0080] |
| oem/oem | VIPProxy_Two | -0.1314 | [-0.8976, +0.6568] |
| oem/oem | Reacquired_Native | +5.4593 | [+4.6797, +6.2048] |
| oem/oem | MeanLogit_Reacquired | +1.4114 | [+0.9829, +1.8363] |
| vdd/vdd | Geometry | -0.2718 | [-0.6281, +0.0837] |
| vdd/vdd | SCLIP_Two | -1.0454 | [-2.0397, -0.0171] |
| vdd/vdd | VIPProxy_Two | -0.9827 | [-1.7718, -0.1352] |
| vdd/vdd | Reacquired_Native | +3.3146 | [+2.3924, +4.3541] |
| vdd/vdd | MeanLogit_Reacquired | +0.3063 | [-0.2058, +0.8266] |
| potsdam/potsdam | Geometry | +2.7625 | [+2.0764, +3.4374] |
| potsdam/potsdam | SCLIP_Two | -2.7306 | [-3.4801, -2.0336] |
| potsdam/potsdam | VIPProxy_Two | +0.0920 | [-0.7510, +0.9156] |
| potsdam/potsdam | Reacquired_Native | +15.5003 | [+14.0341, +16.8399] |
| potsdam/potsdam | MeanLogit_Reacquired | +8.2431 | [+6.9800, +9.4242] |
| vaihingen/vaihingen | Geometry | -1.1271 | [-1.7824, -0.4939] |
| vaihingen/vaihingen | SCLIP_Two | -1.8750 | [-3.0434, -0.7277] |
| vaihingen/vaihingen | VIPProxy_Two | -1.5362 | [-2.4610, -0.5257] |
| vaihingen/vaihingen | Reacquired_Native | +5.1076 | [+4.3045, +6.1552] |
| vaihingen/vaihingen | MeanLogit_Reacquired | +0.8601 | [+0.2875, +1.5645] |
| landcoverai/landcoverai | Geometry | -0.4427 | [-0.9225, -0.0039] |
| landcoverai/landcoverai | SCLIP_Two | -4.1146 | [-5.0600, -2.9820] |
| landcoverai/landcoverai | VIPProxy_Two | -1.5629 | [-2.2484, -0.7845] |
| landcoverai/landcoverai | Reacquired_Native | +4.3200 | [+3.1158, +5.6513] |
| landcoverai/landcoverai | MeanLogit_Reacquired | +0.0217 | [-0.6403, +0.6714] |
| flair1/flair1 | Geometry | -0.2438 | [-0.7683, +0.2329] |
| flair1/flair1 | SCLIP_Two | -0.8786 | [-1.9093, +0.5602] |
| flair1/flair1 | VIPProxy_Two | -2.6739 | [-3.5993, -1.4942] |
| flair1/flair1 | Reacquired_Native | +3.5708 | [+2.8524, +4.2159] |
| flair1/flair1 | MeanLogit_Reacquired | +0.1043 | [-0.5236, +0.6786] |

## Foreground metrics

| Protocol | Metric | Geometry | Primary |
| --- | --- | ---: | ---: |
| loveda/D | D foreground mIoU | 44.8329 | 43.9193 |
| udd5/udd5 | non-residual mIoU | 55.5480 | 54.8755 |
| landcoverai/landcoverai | foreground mIoU | 53.7542 | 53.5200 |

## Per-class effects

### loveda/P

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 86.5825 | 85.5108 | -1.0717 | 89.8428/95.9773 | 89.4697/95.0800 | -1101991 | +417473 |
| road | 58.6185 | 57.5371 | -1.0814 | 63.0348/89.3239 | 61.3321/90.2902 | +769076 | +3616057 |
| water | 70.0570 | 69.6493 | -0.4077 | 92.7864/74.0924 | 93.5048/73.1903 | -1800300 | -1349365 |
| barren | 40.3377 | 36.5403 | -3.7974 | 57.7814/57.1949 | 50.8597/56.4809 | -531099 | +9507119 |
| tree | 56.0994 | 52.4689 | -3.6305 | 61.9092/85.6693 | 57.2532/86.2616 | +744016 | +14691478 |
| farm | 78.0391 | 74.6760 | -3.3631 | 92.5485/83.2712 | 93.1351/79.0259 | -20678286 | -4284178 |

Primary beneficial/harmful/wrong-to-wrong: 21,820,752/44,419,336/20,443,113. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### loveda/D

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 29.6149 | 24.1903 | -5.4246 | 55.9241/38.6319 | 54.9477/30.1752 | -52029662 | -35107292 |
| building | 52.9095 | 54.0891 | 1.1796 | 54.5552/94.6060 | 56.1267/93.7103 | -1099974 | -6822602 |
| road | 40.9417 | 37.5411 | -3.4006 | 43.2448/88.4893 | 39.2390/89.6653 | +935942 | +18075650 |
| water | 51.5320 | 54.0898 | 2.5578 | 74.9765/62.2359 | 77.3705/64.2552 | +4029886 | -3947109 |
| barren | 23.2313 | 24.0376 | 0.8063 | 47.1459/31.4122 | 42.0197/35.9672 | +3388124 | +10721211 |
| tree | 42.1096 | 36.4099 | -5.6997 | 47.6266/78.4259 | 39.5953/81.9035 | +4368483 | +48620884 |
| farm | 58.2736 | 57.3482 | -0.9254 | 74.0712/73.2070 | 72.6567/73.1316 | -367146 | +9233605 |

Primary beneficial/harmful/wrong-to-wrong: 69,175,541/109,949,888/60,335,205. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### udd5/udd5

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 83.7856 | 1.2919 | 96.4862/85.0488 | 94.7738/87.8442 | +3642787 | +2276211 |
| building | 83.9035 | 80.1233 | -3.7802 | 88.3604/94.3293 | 88.9947/88.9351 | -9316872 | -2466540 |
| road | 45.7856 | 45.4111 | -0.3745 | 70.6780/56.5218 | 71.3257/55.5529 | -571682 | -658284 |
| vehicle | 10.0092 | 10.1818 | 0.1726 | 10.0323/97.7477 | 10.2161/96.8056 | -33182 | -908994 |
| other | 30.5846 | 28.0992 | -2.4854 | 52.8537/42.0591 | 46.2064/41.7602 | -222357 | +8258913 |

Primary beneficial/harmful/wrong-to-wrong: 12,576,681/19,077,987/10,702,270. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### oem/oem

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 11.1776 | 11.5383 | 0.3607 | 11.5194/79.0213 | 11.8690/80.5496 | +73469 | -425663 |
| rangeland | 29.1598 | 32.9266 | 3.7668 | 63.4047/35.0605 | 63.9609/40.4268 | +4208335 | +1994151 |
| developed space | 29.2459 | 28.7176 | -0.5283 | 46.0185/44.5188 | 52.6986/38.6907 | -4300195 | -12907977 |
| road | 38.5653 | 40.9045 | 2.3392 | 46.7713/68.7315 | 47.8218/73.8757 | +1353367 | +627332 |
| tree | 57.0767 | 59.2220 | 2.1453 | 80.7054/66.0959 | 79.8482/69.6289 | +2472560 | +1239310 |
| water | 67.5805 | 68.3775 | 0.7970 | 75.9711/85.9530 | 77.2228/85.6519 | -26469 | -169041 |
| agriculture land | 61.2505 | 60.4971 | -0.7534 | 87.8926/66.8946 | 90.8070/64.4440 | -1079448 | -1185244 |
| building | 62.8216 | 62.5076 | -0.3140 | 68.5826/88.2057 | 65.7707/92.6465 | +2945493 | +5180020 |

Primary beneficial/harmful/wrong-to-wrong: 18,896,367/13,249,255/14,734,930. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### vdd/vdd

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 15.2481 | 15.0528 | -0.1953 | 25.6981/27.2713 | 26.4779/25.8628 | -2543446 | -12706769 |
| wall | 22.9675 | 21.9528 | -1.0147 | 23.9610/84.7081 | 22.9906/82.9441 | -547913 | +2799138 |
| road | 48.5684 | 51.0128 | 2.4444 | 50.7006/92.0313 | 54.0987/89.9428 | -1100573 | -6942094 |
| vegetation | 70.0250 | 72.6247 | 2.5997 | 93.0434/73.8938 | 91.5038/77.8761 | +13653837 | +5849246 |
| vehicle | 9.4839 | 7.4649 | -2.0190 | 9.5584/92.4087 | 7.5208/90.9370 | -73557 | +12186403 |
| roof | 65.8483 | 62.9101 | -2.9382 | 85.6067/74.0462 | 85.0086/70.7605 | -6637174 | +58747 |
| water | 39.8168 | 39.0369 | -0.7799 | 89.9359/41.6736 | 92.4783/40.3170 | -1977782 | -2018063 |

Primary beneficial/harmful/wrong-to-wrong: 38,391,220/37,617,828/59,047,998. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### potsdam/potsdam

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 53.0394 | 53.1462 | 0.1068 | 73.3419/65.7067 | 76.9187/63.2300 | -3928410 | -7786785 |
| building | 79.5450 | 78.8843 | -0.6607 | 84.8364/92.7291 | 85.8020/90.7272 | -2413316 | -1882204 |
| low vegetation | 39.9270 | 49.3487 | 9.4217 | 91.6349/41.4373 | 89.0392/52.5404 | +11756475 | +2843047 |
| tree | 52.0633 | 57.6107 | 5.5474 | 81.5747/59.0017 | 83.6252/64.9360 | +5096883 | -525215 |
| car | 10.8915 | 11.9517 | 1.0602 | 10.9005/99.2513 | 11.9635/99.1826 | -6798 | -8061616 |
| clutter | 8.6688 | 9.7685 | 1.0997 | 14.1436/18.2972 | 14.8098/22.2980 | +926400 | +3981539 |

Primary beneficial/harmful/wrong-to-wrong: 31,030,906/19,599,672/23,935,484. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### vaihingen/vaihingen

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 49.9570 | 42.0756 | -7.8814 | 81.9262/56.1448 | 83.1778/45.9891 | -3237359 | -983468 |
| building | 73.3086 | 72.8432 | -0.4654 | 75.8673/95.6017 | 75.7855/94.9398 | -195336 | -22261 |
| low vegetation | 40.6120 | 45.3853 | 4.7733 | 84.9908/43.7498 | 80.5200/50.9833 | +1736553 | +1106280 |
| tree | 70.1372 | 69.3525 | -0.7847 | 80.5771/84.4075 | 80.7159/83.1259 | -322607 | -122428 |
| car | 9.9897 | 8.7125 | -1.2772 | 10.0132/97.7006 | 8.7328/97.3929 | -4500 | +2045126 |

Primary beneficial/harmful/wrong-to-wrong: 3,897,672/5,920,921/3,161,053. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### landcoverai/landcoverai

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 81.1530 | 79.8764 | -1.2766 | 91.7425/87.5478 | 92.0266/85.8154 | -4366777 | -1120796 |
| building | 36.2084 | 34.6788 | -1.5296 | 36.4852/97.9479 | 34.8991/98.2126 | +10174 | +487877 |
| woodland | 78.2983 | 77.4710 | -0.8273 | 91.6761/84.2907 | 88.9697/85.7025 | +1874298 | +3945537 |
| water | 75.3645 | 75.8690 | 0.5045 | 83.6697/88.3620 | 84.8976/87.7061 | -163137 | -408927 |
| road | 25.1457 | 26.0612 | 0.9155 | 26.2970/85.1709 | 27.1369/86.7981 | +104307 | -362556 |

Primary beneficial/harmful/wrong-to-wrong: 8,244,132/10,785,267/2,433,129. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

### flair1/flair1

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 58.5889 | 58.3950 | -0.1939 | 60.2451/95.5180 | 59.9700/95.6959 | +629804 | +2994960 |
| pervious surface | 36.8865 | 33.2423 | -3.6442 | 62.7436/47.2315 | 59.6716/42.8747 | -13168246 | +2813955 |
| impervious surface | 57.7769 | 60.6160 | 2.8391 | 69.1031/77.9009 | 73.9953/77.0243 | -5404595 | -47851179 |
| bare soil | 44.3452 | 40.8721 | -3.4731 | 55.1066/69.4267 | 51.0609/67.1947 | -4003850 | +14069189 |
| water | 76.4129 | 76.4425 | 0.0296 | 84.2026/89.2006 | 86.0006/87.3064 | -4664688 | -6213592 |
| coniferous | 11.6147 | 13.3416 | 1.7269 | 42.8612/13.7426 | 40.7980/16.5446 | +2757485 | +5597002 |
| deciduous | 53.7748 | 54.0259 | 0.2511 | 77.5327/63.7012 | 75.9541/65.1730 | +8425101 | +12441884 |
| brushwood | 26.4178 | 26.3408 | -0.0770 | 32.3517/59.0216 | 32.6533/57.6730 | -3833050 | -12694486 |
| vineyard | 62.2491 | 62.2293 | -0.0198 | 83.3337/71.1008 | 89.3238/67.2297 | -6164262 | -9847616 |
| herbaceous vegetation | 40.8036 | 43.1511 | 2.3475 | 73.1251/48.0020 | 71.9940/51.8555 | +35155450 | +23084224 |
| agricultural land | 29.6360 | 27.5635 | -2.0725 | 54.4432/39.4090 | 52.3112/36.8142 | -7417953 | +1671470 |
| plowed land | 27.1787 | 26.5394 | -0.6393 | 32.9694/60.7447 | 31.6315/62.2441 | +1390480 | +10232513 |

Primary beneficial/harmful/wrong-to-wrong: 181,552,190/177,850,514/177,951,040. These counts diagnose pixel accuracy; their net is not an mIoU criterion.

## Diagnostics and evaluation cost

These timings execute all six arms together. Max shard duration excludes queuing; sum shard elapsed time is not measured kernel-active time. Standalone cost, when available, is reported separately in independent_latency.json.

| Dataset | Max shard seconds | Sum shard seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| loveda | 669.510 | 2646.156 | 3838.889 |
| udd5 | 579.232 | 579.232 | 3837.823 |
| oem | 244.319 | 462.138 | 3838.144 |
| vdd | 728.468 | 1427.401 | 3838.837 |
| potsdam | 283.441 | 545.981 | 3836.441 |
| vaihingen | 103.057 | 103.057 | 3835.506 |
| landcoverai | 59.933 | 234.670 | 3734.500 |
| flair1 | 835.735 | 3328.997 | 3735.594 |

Complete suite wall time, including scheduling: 2725.909 seconds.

| Dataset/protocol | Native patch mass | Backbone cosine | Mean absolute backbone change | Prefix change |
| --- | ---: | ---: | ---: | ---: |
| loveda/P | 0.620848 | 0.838782 | 0.114950 | 0.000000 |
| loveda/D | 0.620848 | 0.838782 | 0.114950 | 0.000000 |
| udd5/udd5 | 0.617977 | 0.834370 | 0.116327 | 0.000000 |
| oem/oem | 0.650822 | 0.835335 | 0.114791 | 0.000000 |
| vdd/vdd | 0.599547 | 0.839109 | 0.114108 | 0.000000 |
| potsdam/potsdam | 0.644007 | 0.819219 | 0.123005 | 0.000000 |
| vaihingen/vaihingen | 0.659964 | 0.809956 | 0.123315 | 0.000000 |
| landcoverai/landcoverai | 0.593904 | 0.857606 | 0.107311 | 0.000000 |
| flair1/flair1 | 0.623477 | 0.838009 | 0.115080 | 0.000000 |

### Independent deployed cost

One1024x1024 LoveDA image on an idle physical A800 GPU4;9 windows/image. Five window warmups,20 window repetitions, one image warmup and3 image repetitions. These are independent single-arm measurements, not full-dataset throughput. Text/model loading and label evaluation are excluded; image assembly is included. Standalone primary features exactly match the evaluation path (maximum error0).

| Method | Median window ms | Median image seconds | Image peak allocated MiB |
| --- | ---: | ---: | ---: |
| Geometry | 22.1250 | 0.323385 | 3605.347 |
| Reacquired_Geometry | 30.7006 | 0.405549 | 3625.570 |

Primary/Geometry cost ratio: windows 1.3876x; complete image 1.2541x.

## VIP comparison boundary

VIPProxy_Two is a matched DINO.text attention-operator adaptation with the same20 words, physical view and assembly. It is not the full VIP system. SCLIP_Two is also an operator adaptation.
The valid historical upstream-configured full-system references on these locked masks are VDD52.0647 and Potsdam44.0643. Compare the primary to these as additional system targets, while disclosing their different official short words, view, scoring and background rules.
- vdd: primary38.5793; historical upstream-configured VIP52.0647; delta-13.4854.
- potsdam: primary43.4517; historical upstream-configured VIP44.0643; delta-0.6126.

Historical uncorrected Vaihingen VIP5.6908 is excluded from victory claims. VIP's other fixed20 constructor-default transfers do not reproduce its full prompt-generation/distillation procedure. They cannot establish eight-dataset SOTA. iSAID has no labeled validation here; LandCover.ai is the eighth available labeled dataset.

## Mechanism interpretation

Uniform-relation execution recovered the native backbone and original Geometry exactly in both FP32 and bf16, and the original Geometry/SCLIP/VIP-proxy confusions exactly replayed the earlier full trial. Thus any changed performance is attributable to the declared conditioned feature path under this protocol, rather than a changed baseline or alias bank.
Compare Reacquired_Native to the previous Native audit as an information-source diagnostic, but do not credit improvements over Native as improvements over original Geometry. Coupling can outperform a weak same-source fusion and still fail the intended final model target.
The diagnostics measure actual feature displacement and native attention mass. They do not prove that excluded context caused an error or that preserved prefixes protect class calibration. Overactivation, missed coverage and class competition must be judged by the per-class TP/FP/FN and transitions above. Instance size and boundary claims need a separate spatial audit.

## Decision

The fixed candidate does not achieve the intended eight-domain dominance. Preserve its complete negative and positive outcomes; do not select a different branch or replay depth by dataset. The results reject the corresponding universal improvement claim, not all possible training-free readouts.

The primary improves Potsdam and OEM over original Geometry, but loses on the other six primary domains. In Potsdam, low-vegetation and tree IoU improve by9.4217 and5.5474 points; car FP falls by8,061,616 while TP falls by6,798. In VDD, vehicle FP increases by12,186,403 and its IoU falls by2.0190 points; water recall also falls. These opposite effects rule out claiming that this fixed intervention reliably solves false activation and missed coverage across domains. Feature movement and preserved native prefixes alone are not semantic correctness tests. Do not promote this version over preserved Geometry or historical best models, and do not add dataset-specific replay tuning or another confidence gate from these labeled outcomes.
