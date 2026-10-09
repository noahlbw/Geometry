# Bounded Crop-Head Alias: Frozen Eight-Domain Pilot

UDD5 full40 plus eight evenly spaced COMPLETE images in each other domain;96 developed images, not full eight datasets or independent validation. Fixed20 aliases, bounded896/448 inputs, patch-only2 Geometry and original H. Crop-native semantic heads reuse existing raw patch tokens and prefixes; no additional RGB/backbone encoding, no fine pixels and no VIP-proxy witness. Soft alias weights depend on query/class/rival; high response alone is not reliability.

| Dataset/protocol | Geometry | NoAdmission_Exact | Geometry_PatchOnly2Coupled | CropHead_RivalSoft | CropHead_RivalHard | CropHead_ClassMean | CropHead_AliasShuffle | CropHead_ObservationMean | Primary-baseline pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 42.7217 | 46.3491 | 46.8468 | 46.8337 | 46.6398 | 46.8626 | 46.6870 | 45.7711 | -0.0131 |
| potsdam/potsdam | 45.6466 | 49.7562 | 51.5168 | 51.3210 | 50.7985 | 51.5063 | 51.5416 | 50.5852 | -0.1958 |
| udd5/udd5 | 47.1219 | 47.0096 | 47.0786 | 47.1360 | 47.1809 | 47.0728 | 46.9962 | 46.8534 | +0.0574 |
| oem/oem | 31.4297 | 25.7311 | 25.2530 | 25.1062 | 24.9014 | 25.2291 | 25.2851 | 24.4973 | -0.1468 |
| loveda/P | 70.8274 | 67.0933 | 67.2104 | 66.8945 | 66.9454 | 67.1908 | 66.9296 | 67.9522 | -0.3159 |
| loveda/D | 46.6369 | 41.0700 | 39.4705 | 39.4118 | 39.1977 | 39.4553 | 39.4996 | 40.3333 | -0.0587 |
| vaihingen/vaihingen | 46.0831 | 48.3789 | 49.1468 | 49.2314 | 49.4696 | 49.1300 | 49.0132 | 53.4563 | +0.0846 |
| landcoverai/landcoverai | 57.9352 | 75.1447 | 76.2441 | 76.1378 | 75.7262 | 76.2318 | 76.2246 | 73.5138 | -0.1063 |
| flair1/flair1 | 39.8935 | 37.2481 | 37.6036 | 37.5512 | 37.3249 | 37.5952 | 37.8311 | 37.3796 | -0.0524 |

| Arm | Equal-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| CropHead_RivalSoft | 46.5911 |
| CropHead_RivalHard | 46.4049 |
| CropHead_ClassMean | 46.6354 |
| CropHead_AliasShuffle | 46.6348 |
| CropHead_ObservationMean | 46.5487 |

## Matched Complete-Image Timing

| Domain | Baseline ms | Primary ms | VIP20 ms | Primary/baseline |
| --- | ---: | ---: | ---: | ---: |
| vdd | 328.13 | 559.73 | 125.91 | 1.706x |
| potsdam | 237.38 | 666.30 | 101.25 | 2.807x |
| udd5 | 404.80 | 597.71 | 173.21 | 1.477x |
| oem | 235.57 | 664.63 | 101.49 | 2.821x |
| loveda | 258.27 | 1123.15 | 207.91 | 4.349x |
| vaihingen | 235.41 | 659.24 | 99.53 | 2.800x |
| landcoverai | 230.08 | 652.36 | 97.03 | 2.835x |
| flair1 | 238.45 | 679.87 | 103.68 | 2.851x |

Three fixed complete inputs/domain, three synchronized singleton repeats, serial idle-GPU timing. Includes all views, alias action, H writing, restoration and argmax; excludes loading/text/decoding/masks. Shared-resident peaks are not standalone memory.

## Frozen Advancement Gate

```json
{
  "passed": false,
  "checks": {
    "mean_gain": false,
    "domain_wins": false,
    "worst_protocol_loss": true,
    "above_class_mean": false,
    "above_alias_shuffle": false,
    "above_same_source_mean": true,
    "all_eight_timings": true,
    "mean_cost_ratio": false,
    "domain_mean_below1000ms": false
  },
  "domain_wins": 2,
  "mean_gain_pp": -0.053887500000001864,
  "worst_protocol_delta_pp": -0.3159000000000134
}
```

No post-result primary, threshold or domain-route changes and no automatic full20092 promotion. A passed developed pilot still requires vocabulary stress and unchanged full-domain validation. A failed gate rejects this implementation, not every possible conditional alias mechanism.

## Coverage And Competition Outcomes

Confusion rows are targets and columns are predictions. Delta TP counts changes in correct coverage; delta FP counts changes in competitor activation. Negative delta FP is beneficial, but does not establish useful suppression if correct coverage also falls.

| Dataset/protocol | Class | Baseline IoU | Soft IoU | Delta pp | Delta TP pixels | Delta FP pixels | Baseline precision | Soft precision | Baseline recall | Soft recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.6232 | 20.5875 | -0.0357 | +14108 | +118462 | 32.3110 | 32.1560 | 36.3108 | 36.3970 |
| vdd/vdd | wall | 46.3090 | 46.0740 | -0.2350 | -20902 | -22068 | 84.1267 | 84.7435 | 50.7428 | 50.2414 |
| vdd/vdd | road | 15.2940 | 15.2090 | -0.0850 | -96 | +50050 | 15.7348 | 15.6450 | 84.5199 | 84.5140 |
| vdd/vdd | vegetation | 45.8052 | 45.3270 | -0.4782 | -139629 | -10212 | 96.1118 | 96.1448 | 46.6700 | 46.1661 |
| vdd/vdd | vehicle | 23.8613 | 24.7810 | +0.9197 | -670 | -41637 | 24.5176 | 25.5091 | 89.9132 | 89.6725 |
| vdd/vdd | roof | 85.4306 | 85.4243 | -0.0063 | +6604 | +10040 | 85.8504 | 85.8258 | 99.4309 | 99.4554 |
| vdd/vdd | water | 90.6042 | 90.4329 | -0.1713 | -1606 | +37556 | 91.1218 | 90.9557 | 99.3769 | 99.3684 |
| potsdam/potsdam | impervious surface | 75.3698 | 75.2578 | -0.1120 | +3647 | +10350 | 84.9896 | 84.7392 | 86.9432 | 87.0569 |
| potsdam/potsdam | building | 80.4415 | 80.7098 | +0.2683 | -444 | -5057 | 80.4757 | 80.7708 | 99.9471 | 99.9064 |
| potsdam/potsdam | low vegetation | 60.1752 | 59.4153 | -0.7599 | -18645 | -2146 | 93.9502 | 94.0117 | 62.6009 | 61.7524 |
| potsdam/potsdam | tree | 57.3938 | 57.1596 | -0.2342 | -4539 | -1943 | 77.3966 | 77.4409 | 68.9511 | 68.5786 |
| potsdam/potsdam | car | 31.4528 | 31.2587 | -0.1941 | -37 | +3155 | 31.5636 | 31.3703 | 98.8959 | 98.8738 |
| potsdam/potsdam | clutter | 4.2680 | 4.1248 | -0.1432 | +83 | +15576 | 5.7679 | 5.4987 | 14.0993 | 14.1695 |
| udd5/udd5 | vegetation | 66.2953 | 66.2441 | -0.0512 | -72633 | -6406 | 96.5410 | 96.5450 | 67.9084 | 67.8526 |
| udd5/udd5 | building | 81.2771 | 81.2747 | -0.0024 | +29039 | +41717 | 82.2868 | 82.2727 | 98.5127 | 98.5295 |
| udd5/udd5 | road | 37.2416 | 37.2110 | -0.0306 | +45656 | +182636 | 65.9709 | 65.7178 | 46.0967 | 46.1741 |
| udd5/udd5 | vehicle | 16.9950 | 17.3477 | +0.3527 | -9018 | -382193 | 17.8319 | 18.2346 | 78.3585 | 78.1025 |
| udd5/udd5 | other | 33.5843 | 33.6024 | +0.0181 | +59251 | +111951 | 47.1961 | 47.1708 | 53.7992 | 53.8788 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0.0000 | +0 | +8248 | 0.0000 | 0.0000 | N/A | N/A |
| oem/oem | rangeland | 4.3949 | 4.1394 | -0.2555 | -3787 | -8510 | 66.4403 | 72.1599 | 4.4947 | 4.2066 |
| oem/oem | developed space | 21.7107 | 21.6310 | -0.0797 | -6512 | -14522 | 24.6999 | 24.6643 | 64.2081 | 63.7528 |
| oem/oem | road | 26.4384 | 26.4259 | -0.0125 | +426 | +1936 | 45.0353 | 44.8768 | 39.0337 | 39.1260 |
| oem/oem | tree | 10.8604 | 10.4583 | -0.4021 | -4414 | -722 | 86.7166 | 86.7472 | 11.0442 | 10.6281 |
| oem/oem | water | 18.0795 | 17.3834 | -0.6961 | +362 | +6660 | 22.5533 | 21.3151 | 47.6827 | 48.5178 |
| oem/oem | agriculture land | 69.4583 | 69.3790 | -0.0793 | +30 | +1230 | 69.6777 | 69.5959 | 99.5487 | 99.5529 |
| oem/oem | building | 51.0820 | 51.4328 | +0.3508 | +14249 | +5326 | 78.2527 | 78.2032 | 59.5335 | 60.0397 |
| loveda/P | building | 88.2412 | 88.2482 | +0.0070 | -82 | -122 | 92.8501 | 92.8810 | 94.6743 | 94.6502 |
| loveda/P | road | 79.8873 | 78.4726 | -1.4147 | +557 | +18244 | 83.5543 | 81.9573 | 94.7923 | 94.8602 |
| loveda/P | water | 62.0850 | 63.0239 | +0.9389 | +13087 | -248 | 98.5503 | 98.5985 | 62.6572 | 63.5936 |
| loveda/P | barren | 70.3607 | 70.2547 | -0.1060 | -57 | +239 | 71.3382 | 71.2490 | 98.0897 | 98.0523 |
| loveda/P | tree | 39.2222 | 37.5650 | -1.6572 | -18435 | -203 | 94.6144 | 94.4326 | 40.1179 | 38.4158 |
| loveda/P | farm | 63.4659 | 63.8028 | +0.3369 | +787 | -13767 | 64.3610 | 64.6889 | 97.8555 | 97.8982 |
| loveda/D | background | 4.0203 | 4.0822 | +0.0619 | +1578 | -3523 | 49.2011 | 50.3256 | 4.1944 | 4.2536 |
| loveda/D | building | 45.7347 | 45.7914 | +0.0567 | -181 | -1263 | 47.1134 | 47.1870 | 93.9861 | 93.9330 |
| loveda/D | road | 63.0492 | 62.0848 | -0.9644 | +623 | +20087 | 65.4549 | 64.3808 | 94.4918 | 94.5678 |
| loveda/D | water | 58.9731 | 59.9507 | +0.9776 | +14182 | -555 | 90.9392 | 91.1225 | 62.6546 | 63.6694 |
| loveda/D | barren | 27.9609 | 28.2293 | +0.2684 | +723 | -2474 | 28.1841 | 28.4165 | 97.2456 | 97.7206 |
| loveda/D | tree | 34.5990 | 33.5919 | -1.0071 | -11686 | +483 | 81.3407 | 80.8147 | 37.5818 | 36.5028 |
| loveda/D | farm | 41.9559 | 42.1522 | +0.1963 | +564 | -18558 | 42.4483 | 42.6433 | 97.3098 | 97.3404 |
| vaihingen/vaihingen | impervious surface | 60.8940 | 60.9862 | +0.0922 | +10360 | +12382 | 84.1446 | 83.7509 | 68.7868 | 69.1707 |
| vaihingen/vaihingen | building | 65.1989 | 65.3070 | +0.1081 | -155 | -5508 | 65.4089 | 65.5209 | 99.5099 | 99.5025 |
| vaihingen/vaihingen | low vegetation | 27.2861 | 27.0208 | -0.2653 | -4508 | -1509 | 84.6746 | 84.7938 | 28.7036 | 28.3969 |
| vaihingen/vaihingen | tree | 65.8712 | 65.8476 | -0.0236 | -1156 | -1043 | 77.8059 | 77.8390 | 81.1120 | 81.0403 |
| vaihingen/vaihingen | car | 26.4838 | 26.9955 | +0.5117 | -61 | -8802 | 27.2384 | 27.7844 | 90.5299 | 90.4838 |
| landcoverai/landcoverai | background | 82.5166 | 82.3802 | -0.1364 | -933 | +148 | 93.9248 | 93.8959 | 87.1691 | 87.0416 |
| landcoverai/landcoverai | building | N/A | N/A | N/A | +0 | +0 | N/A | N/A | N/A | N/A |
| landcoverai/landcoverai | woodland | 91.5012 | 91.4742 | -0.0270 | -222 | +68 | 95.2495 | 95.2420 | 95.8766 | 95.8545 |
| landcoverai/landcoverai | water | 91.7035 | 91.4942 | -0.2093 | +10 | +843 | 91.7116 | 91.4997 | 99.9904 | 99.9934 |
| landcoverai/landcoverai | road | 39.2550 | 39.2026 | -0.0524 | +5 | +81 | 46.0574 | 45.9781 | 72.6614 | 72.6795 |
| flair1/flair1 | building | 49.1283 | 49.0684 | -0.0599 | +240 | +727 | 51.3696 | 51.2325 | 91.8434 | 92.0737 |
| flair1/flair1 | pervious surface | 17.8467 | 17.7414 | -0.1053 | -67 | -44 | 64.2446 | 64.2724 | 19.8148 | 19.6825 |
| flair1/flair1 | impervious surface | 66.4519 | 66.4520 | +0.0001 | -401 | -604 | 74.8732 | 74.9654 | 85.5244 | 85.4046 |
| flair1/flair1 | bare soil | 21.3471 | 20.7465 | -0.6006 | +1 | +2283 | 21.3761 | 20.7737 | 99.3672 | 99.3731 |
| flair1/flair1 | water | 94.9931 | 95.0874 | +0.0943 | +52 | -1 | 99.7998 | 99.8019 | 95.1745 | 95.2672 |
| flair1/flair1 | coniferous | 5.9298 | 5.9623 | +0.0325 | +0 | -31 | 7.4090 | 7.4597 | 22.8997 | 22.8997 |
| flair1/flair1 | deciduous | 64.3305 | 64.1622 | -0.1683 | -1146 | -1040 | 81.0241 | 81.3037 | 75.7421 | 75.2675 |
| flair1/flair1 | brushwood | 3.3685 | 3.2734 | -0.0951 | -239 | -2375 | 6.8848 | 6.8113 | 6.1874 | 5.9284 |
| flair1/flair1 | vineyard | 51.9276 | 52.0080 | +0.0804 | -9 | -361 | 52.0611 | 52.1440 | 99.5087 | 99.5010 |
| flair1/flair1 | herbaceous vegetation | 37.8267 | 37.8983 | +0.0716 | +618 | +419 | 86.6034 | 86.5037 | 40.1776 | 40.2801 |
| flair1/flair1 | agricultural land | 38.0928 | 38.2145 | +0.1217 | +1553 | +2415 | 83.6119 | 82.8762 | 41.1664 | 41.4905 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0.0000 | +0 | -1990 | 0.0000 | 0.0000 | N/A | N/A |
