# VIP-Resolution Coupling: Full-Image Accuracy

Frozen signature `geometry-shared-rival-soft-vip448-v1-20261005`. Completed 8/8 domains; only completed full-domain evaluations appear below. No partial-shard mIoU is used as a full score.

Whole-image uint8/OpenCV resizing to long edge448, aspect preserved;336 crops, stride112, direct21x21 Geometry tokens. At most4 Geometry and4 wide encodings per input, zero native-resolution/fine encodings. Retained fixed20 words/class, Geometry depth2, local RS/wide ImageNet text banks, soft admission rule, reconstruction and Hann probability stitching. Thus the previous physical local/wide distinction is changed; accuracy is measured anew.

| Dataset/protocol | Images | Finite VIP All20 | New Geometry | New no admission | New soft primary | Soft - VIP | Soft - no admission | Old native soft | New - old native soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 1669 | 56.3110 | 63.5153 | 58.2889 | 58.2547 | +1.9437 | -0.0342 | 62.6510 | -4.3963 |
| loveda/D | 1669 | 36.1457 | 40.7596 | 37.7093 | 37.6762 | +1.5305 | -0.0331 | 40.2095 | -2.5333 |
| udd5/udd5 | 40 | 41.4336 | 38.9008 | 43.4033 | 43.4129 | +1.9793 | +0.0096 | 49.3542 | -5.9413 |
| oem/oem | 384 | 33.2064 | 37.0368 | 33.0227 | 33.0366 | -0.1698 | +0.0139 | 37.8676 | -4.8310 |
| vdd/vdd | 80 | 51.4827 | 43.1489 | 50.1588 | 50.1751 | -1.3076 | +0.0163 | 55.4932 | -5.3181 |
| potsdam/potsdam | 504 | 42.3028 | 28.1005 | 39.4194 | 39.4059 | -2.8969 | -0.0135 | 43.6900 | -4.2841 |
| vaihingen/vaihingen | 113 | 47.3620 | 40.1480 | 46.0436 | 46.0494 | -1.3126 | +0.0058 | 51.3649 | -5.3155 |
| landcoverai/landcoverai | 1602 | 54.9420 | 54.8100 | 61.8658 | 61.8982 | +6.9562 | +0.0324 | 63.6909 | -1.7927 |
| flair1/flair1 | 15700 | 35.2977 | 40.6290 | 41.1866 | 41.2008 | +5.9031 | +0.0142 | 42.5105 | -1.3097 |
| Eight-domain mean (LoveDA D once) | 20092 | 42.7716 | 40.4417 | 44.1012 | 44.1069 | +1.3353 | +0.0057 | 48.0226 | -3.9157 |

VIP comparators are the completed finite-row-repaired All20 evaluations, including corrected-input Vaihingen. Original official short words, historical NaN-contaminated VIP scores and distilled words are not substituted. Same words/input layout does not imply identical templates, visual readout, stitching or background thresholds. These are local method evaluations, not published VIP benchmark numbers. LandCover.ai substitutes for unlabeled iSAID. Method development used prior validation windows; this is exploratory, not independent validation.

## Admission Controls

| Protocol | No admission | Soft | Hard | Shuffled | Mean noncanonical risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 58.2889 | 58.2547 | 58.2887 | 58.2945 | 0.005651 |
| loveda/D | 37.7093 | 37.6762 | 37.7114 | 37.7119 | 0.005467 |
| udd5/udd5 | 43.4033 | 43.4129 | 43.4022 | 43.3805 | 0.004966 |
| oem/oem | 33.0227 | 33.0366 | 33.0229 | 32.9983 | 0.005338 |
| vdd/vdd | 50.1588 | 50.1751 | 50.1588 | 50.1517 | 0.004204 |
| potsdam/potsdam | 39.4194 | 39.4059 | 39.4148 | 39.4543 | 0.006770 |
| vaihingen/vaihingen | 46.0436 | 46.0494 | 46.0434 | 46.0244 | 0.006902 |
| landcoverai/landcoverai | 61.8658 | 61.8982 | 61.8651 | 61.8761 | 0.003300 |
| flair1/flair1 | 41.1866 | 41.2008 | 41.1921 | 41.2237 | 0.005461 |

Primary beats VIP on 4/8 domains; mean delta +1.3353pp. Soft versus matched no-admission mean delta +0.0057pp; versus shuffled weights +0.0043pp. No post-result rule/parameter change or control promotion was made.

Final-model decision: this candidate fails the requested all-eight-domain accuracy criterion and is not selected as the final model. A higher cross-domain mean does not cancel individual-domain or small-class regressions. This is a changed sampling protocol, not accuracy-preserving execution acceleration; no latency claim is made without a new primary-only measurement. The previous native model is retained as an accuracy reference, not declared efficient.

## Foreground And Non-Residual IoU

| Protocol (excluded class) | VIP | No admission | Soft |
| --- | ---: | ---: | ---: |
| loveda/D (background) | 40.1802 | 41.8137 | 41.7955 |
| udd5/udd5 (other) | 44.1680 | 46.2392 | 46.2526 |
| vdd/vdd (other) | 53.9689 | 52.8954 | 52.9103 |
| landcoverai/landcoverai (background) | 50.1322 | 57.0680 | 57.0999 |

## Per-Class Outcomes

| Protocol | Class | VIP IoU | Old native soft IoU | New soft IoU | New - VIP | Precision | Recall | Predicted area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | building | 81.4273 | 88.7605 | 85.5347 | 4.1074 | 90.0294 | 94.4851 | 11.8345 |
| loveda/P | road | 63.5157 | 68.3958 | 63.2700 | -0.2457 | 76.4705 | 78.5649 | 7.5084 |
| loveda/P | water | 65.2910 | 70.3325 | 65.8835 | 0.5925 | 93.7599 | 68.9048 | 13.4672 |
| loveda/P | barren | 34.6101 | 41.1390 | 39.3942 | 4.7841 | 60.1482 | 53.3083 | 6.0534 |
| loveda/P | tree | 21.2621 | 33.3535 | 24.0756 | 2.8135 | 90.2898 | 24.7155 | 3.1574 |
| loveda/P | farm | 71.7595 | 73.9245 | 71.3702 | -0.3893 | 73.7736 | 95.6346 | 57.9790 |
| loveda/D | background | 11.9386 | 13.1988 | 12.9606 | 1.0220 | 59.1736 | 14.2334 | 8.6833 |
| loveda/D | building | 47.6742 | 52.0637 | 50.3819 | 2.7077 | 52.5897 | 92.3084 | 12.6478 |
| loveda/D | road | 50.1601 | 51.6682 | 48.6800 | -1.4801 | 56.7719 | 77.3517 | 6.3629 |
| loveda/D | water | 55.8824 | 59.3715 | 56.1699 | 0.2875 | 78.6452 | 66.2788 | 9.8684 |
| loveda/D | barren | 22.9091 | 29.8463 | 29.0737 | 6.1646 | 47.7448 | 42.6427 | 3.8981 |
| loveda/D | tree | 16.4150 | 26.6504 | 19.4809 | 3.0659 | 66.7565 | 21.5737 | 2.3819 |
| loveda/D | farm | 48.0404 | 48.6673 | 46.9863 | -1.0541 | 48.2348 | 94.7788 | 56.1576 |
| udd5/udd5 | vegetation | 64.3148 | 68.3339 | 59.4448 | -4.8700 | 96.3222 | 60.8253 | 18.7037 |
| udd5/udd5 | building | 81.1240 | 82.5379 | 78.6096 | -2.5144 | 79.8614 | 98.0450 | 48.1973 |
| udd5/udd5 | road | 19.3962 | 39.9653 | 31.9673 | 12.5711 | 61.3051 | 40.0479 | 8.7603 |
| udd5/udd5 | vehicle | 11.8369 | 20.9299 | 14.9886 | 3.1517 | 16.1202 | 68.1027 | 3.3818 |
| udd5/udd5 | other | 30.4958 | 35.0039 | 32.0542 | 1.5584 | 43.8618 | 54.3529 | 20.9569 |
| oem/oem | bareland | 6.9087 | 9.9333 | 9.8718 | 2.9631 | 10.1614 | 77.5978 | 9.8556 |
| oem/oem | rangeland | 9.0957 | 17.8451 | 15.6051 | 6.5094 | 63.0371 | 17.1768 | 5.7370 |
| oem/oem | developed space | 22.7255 | 28.1972 | 26.6962 | 3.9707 | 30.4304 | 68.5092 | 44.5972 |
| oem/oem | road | 30.3281 | 34.3982 | 27.1282 | -3.1999 | 45.3700 | 40.2885 | 6.2721 |
| oem/oem | tree | 31.4655 | 36.9726 | 28.7879 | -2.6776 | 89.3889 | 29.8065 | 6.2652 |
| oem/oem | water | 61.2792 | 68.3011 | 64.5807 | 3.3015 | 75.4372 | 81.7766 | 2.5589 |
| oem/oem | agriculture land | 61.9014 | 61.7309 | 59.7212 | -2.1802 | 63.5505 | 90.8353 | 16.9035 |
| oem/oem | building | 41.9472 | 45.5628 | 31.9015 | -10.0457 | 79.3263 | 34.7943 | 7.8107 |
| vdd/vdd | other | 36.5655 | 35.4141 | 33.7643 | -2.8012 | 48.4921 | 52.6450 | 20.4222 |
| vdd/vdd | wall | 29.0582 | 48.3571 | 36.3375 | 7.2793 | 69.9383 | 43.0636 | 1.9922 |
| vdd/vdd | road | 45.0432 | 41.8065 | 38.9340 | -6.1092 | 41.0970 | 88.0917 | 11.7661 |
| vdd/vdd | vegetation | 50.2515 | 66.6633 | 58.4102 | 8.1587 | 96.3952 | 59.7145 | 22.1245 |
| vdd/vdd | vehicle | 29.6526 | 26.3114 | 20.6636 | -8.9890 | 23.0922 | 66.2707 | 1.4941 |
| vdd/vdd | roof | 85.5153 | 83.6761 | 80.2784 | -5.2369 | 81.0892 | 98.7698 | 25.6297 |
| vdd/vdd | water | 84.2928 | 86.2240 | 82.8380 | -1.4548 | 86.8288 | 94.7432 | 16.5711 |
| potsdam/potsdam | impervious surface | 61.2488 | 61.3570 | 56.6634 | -4.5854 | 69.2379 | 75.7282 | 34.4212 |
| potsdam/potsdam | building | 77.2218 | 79.0066 | 75.9033 | -1.3185 | 78.0071 | 96.5688 | 29.6110 |
| potsdam/potsdam | low vegetation | 28.5736 | 34.1299 | 27.0396 | -1.5340 | 93.3355 | 27.5719 | 6.2062 |
| potsdam/potsdam | tree | 50.9752 | 54.7287 | 46.4064 | -4.5688 | 84.4388 | 50.7464 | 10.2415 |
| potsdam/potsdam | car | 30.4756 | 26.6547 | 24.9647 | -5.5109 | 25.0993 | 97.8964 | 7.6638 |
| potsdam/potsdam | clutter | 5.3220 | 6.2632 | 5.4581 | 0.1361 | 7.1812 | 18.5316 | 11.8562 |
| vaihingen/vaihingen | impervious surface | 56.9979 | 61.2549 | 55.7268 | -1.2711 | 71.5706 | 71.5693 | 28.4534 |
| vaihingen/vaihingen | building | 64.9474 | 68.0060 | 63.7826 | -1.1648 | 64.5001 | 98.2860 | 40.1426 |
| vaihingen/vaihingen | low vegetation | 30.2954 | 33.1817 | 26.5577 | -3.7377 | 86.8452 | 27.6708 | 6.8277 |
| vaihingen/vaihingen | tree | 62.0184 | 69.1505 | 62.1453 | 0.1269 | 80.8293 | 72.8886 | 20.2611 |
| vaihingen/vaihingen | car | 22.5506 | 25.2312 | 22.0345 | -0.5161 | 23.5181 | 77.7431 | 4.3153 |
| landcoverai/landcoverai | background | 74.1810 | 82.5181 | 81.0914 | 6.9104 | 85.1900 | 94.3993 | 66.5116 |
| landcoverai/landcoverai | building | 44.5106 | 47.8759 | 47.1400 | 2.6294 | 48.7092 | 93.6031 | 1.7586 |
| landcoverai/landcoverai | woodland | 52.0468 | 73.5000 | 69.9786 | 17.9318 | 96.2374 | 71.9471 | 23.6338 |
| landcoverai/landcoverai | water | 64.8284 | 76.4114 | 73.5219 | 8.6935 | 87.2499 | 82.3719 | 5.5916 |
| landcoverai/landcoverai | road | 39.1431 | 38.1494 | 37.7591 | -1.3840 | 44.1148 | 72.3820 | 2.5045 |
| flair1/flair1 | building | 60.2824 | 61.5362 | 60.7975 | 0.5151 | 63.0333 | 94.4875 | 12.9320 |
| flair1/flair1 | pervious surface | 32.1094 | 34.8516 | 34.1698 | 2.0604 | 69.1000 | 40.3327 | 4.2990 |
| flair1/flair1 | impervious surface | 56.6589 | 59.1300 | 57.3472 | 0.6883 | 67.4160 | 79.3376 | 17.6807 |
| flair1/flair1 | bare soil | 23.1537 | 36.8814 | 34.7884 | 11.6347 | 36.2620 | 89.5404 | 10.7940 |
| flair1/flair1 | water | 74.1419 | 80.1481 | 79.0643 | 4.9224 | 86.6039 | 90.0811 | 6.2419 |
| flair1/flair1 | coniferous | 14.4692 | 15.5778 | 15.2226 | 0.7534 | 45.9505 | 18.5428 | 0.9677 |
| flair1/flair1 | deciduous | 48.7353 | 55.9525 | 53.6953 | 4.9600 | 78.1962 | 63.1502 | 11.2651 |
| flair1/flair1 | brushwood | 11.1440 | 26.8455 | 26.0897 | 14.9457 | 42.7048 | 40.1402 | 6.5105 |
| flair1/flair1 | vineyard | 57.6837 | 63.9008 | 62.1543 | 4.4706 | 77.0409 | 76.2842 | 3.8422 |
| flair1/flair1 | herbaceous vegetation | 18.8786 | 31.5795 | 29.5248 | 10.6462 | 79.0984 | 32.0232 | 9.0004 |
| flair1/flair1 | agricultural land | 16.1240 | 23.4237 | 21.7866 | 5.6626 | 35.4820 | 36.0796 | 7.0837 |
| flair1/flair1 | plowed land | 10.1912 | 20.2991 | 19.7693 | 9.5781 | 20.4816 | 85.0392 | 9.3827 |

## Evaluation Cost

| Dataset | All-control worker wall s | Aggregate worker s | Peak allocated MiB | Geometry encodings/image | Wide encodings/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda | 228.56 | 907.64 | 5413.15 | 4 | 4 |
| udd5 | 44.63 | 44.63 | 5716.14 | 2 | 2 |
| oem | 147.36 | 147.36 | 5380.82 | 4 | 4 |
| vdd | 72.59 | 72.59 | 5917.40 | 2 | 2 |
| potsdam | 200.58 | 200.58 | 5353.27 | 4 | 4 |
| vaihingen | 41.06 | 41.06 | 5343.98 | 4 | 4 |
| landcoverai | 518.84 | 518.84 | 5343.98 | 4 | 4 |
| flair1 | 1314.66 | 5201.08 | 5461.36 | 4 | 4 |

These worker costs include all five simultaneous readouts, image decoding, masks, original-size output restoration, confusion accumulation and checkpointing. They are NOT warmed primary-only inference latency. No new speed benchmark was run; old native-resolution timings are not assigned to this resized model.

Verified 20092 unique full images, per-image confusion sums, matching scored target counts and class order versus VIP, frozen checkpoints/vocabulary and bounded encodings. Reducing resolution changes visual sampling as well as branch geometry; differences from native results do not isolate only a resize operator.
