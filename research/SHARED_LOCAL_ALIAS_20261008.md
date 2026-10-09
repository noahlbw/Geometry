# Shared-local alias candidate

Frozen96 developed COMPLETE RS inputs, fixed20 aliases. Original Geometry and wide path retained. At most4 extra attributed VIPProxy_Two heads reuse the original Geometry backbone tokens, replacing16 extra fine image encodings. This is a source substitution, not an equivalent acceleration or a new proxy operator. No masks fitted the rule. Old model retained.

| Domain/protocol | No-fine baseline | Same-source mean | Soft | Pooled class | Word shuffle | Retained full-fine | Soft minus retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.8468 | 48.5101 | 48.7258 | 48.9509 | 48.6774 | 49.8732 | -1.1474 |
| potsdam/potsdam | 51.5168 | 54.0657 | 54.4671 | 54.7176 | 54.4527 | 55.1778 | -0.7107 |
| udd5/udd5 | 47.0786 | 49.2523 | 49.5374 | 49.9249 | 49.5218 | 51.2305 | -1.6931 |
| oem/oem | 25.2530 | 28.7491 | 29.1718 | 29.5229 | 29.1478 | 29.9714 | -0.7996 |
| loveda/P | 67.2104 | 73.3196 | 73.8975 | 74.5415 | 73.8610 | 75.5356 | -1.6381 |
| loveda/D | 39.4705 | 42.9670 | 43.3196 | 43.7918 | 43.2734 | 46.6718 | -3.3522 |
| vaihingen/vaihingen | 49.1468 | 51.6522 | 52.0030 | 52.1488 | 51.9518 | 51.8593 | +0.1437 |
| landcoverai/landcoverai | 76.2441 | 76.7347 | 76.7394 | 76.5888 | 76.7259 | 78.0381 | -1.2987 |
| flair1/flair1 | 37.6036 | 39.9654 | 40.2891 | 40.5097 | 40.2975 | 38.9230 | +1.3661 |

Equal-domain means, LoveDA D once: {"Geometry": 44.683575, "NoAdmission_Exact": 46.3359625, "Geometry_PatchOnly2Coupled": 46.645025000000004, "SharedLocal_Soft": 49.28165, "SharedLocal_ObservationMean": 48.9870625, "SharedLocal_PooledClass": 49.519425, "SharedLocal_AliasShuffle": 49.256037500000005, "OneSide_Stream": 50.2181375}

| Domain | Baseline ms | Local mean ms | New soft ms | Retained fine ms | VIP ms | New/VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 426.51 | 438.02 | 473.46 | 815.21 | 176.11 | 2.688 |
| potsdam | 236.96 | 249.28 | 314.97 | 652.41 | 101.68 | 3.098 |
| udd5 | 388.65 | 398.25 | 426.33 | 709.36 | 166.42 | 2.562 |
| oem | 238.56 | 251.94 | 314.72 | 659.33 | 103.34 | 3.045 |
| loveda | 251.36 | 267.87 | 391.35 | 787.35 | 203.39 | 1.924 |
| vaihingen | 240.28 | 251.35 | 311.10 | 652.98 | 100.93 | 3.082 |
| landcoverai | 230.76 | 242.00 | 301.54 | 639.87 | 97.74 | 3.085 |
| flair1 | 232.91 | 246.45 | 316.40 | 669.29 | 101.97 | 3.103 |

## Interpretation

New soft minus retained mean: -0.9365pp; worst protocol: -3.3522pp. Same-source attenuation gain: +0.2946pp; gain over pooled class: -0.2378pp; gain over fixed within-class word shuffle: +0.0256pp. The source substitution improves the no-fine baseline, but positive observation and word-specific attenuation must not be conflated. A pooled-class or shuffled control does not become the primary method after seeing these outcomes.

| Protocol | Class | New soft IoU | Retained IoU | Delta pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | other | 19.0627 | 21.0681 | -2.0054 |
| vdd/vdd | wall | 51.0552 | 55.6308 | -4.5756 |
| vdd/vdd | road | 15.4134 | 16.5334 | -1.1200 |
| vdd/vdd | vegetation | 50.6422 | 48.9341 | +1.7081 |
| vdd/vdd | vehicle | 26.4369 | 26.8570 | -0.4201 |
| vdd/vdd | roof | 86.3699 | 87.5307 | -1.1608 |
| vdd/vdd | water | 92.1006 | 92.5582 | -0.4576 |
| potsdam/potsdam | impervious surface | 77.3177 | 77.3746 | -0.0569 |
| potsdam/potsdam | building | 82.1808 | 83.9116 | -1.7308 |
| potsdam/potsdam | low vegetation | 64.8406 | 65.3738 | -0.5332 |
| potsdam/potsdam | tree | 61.1102 | 62.6535 | -1.5433 |
| potsdam/potsdam | car | 33.4538 | 33.1180 | +0.3358 |
| potsdam/potsdam | clutter | 7.8992 | 8.6353 | -0.7361 |
| udd5/udd5 | vegetation | 72.3672 | 73.1690 | -0.8018 |
| udd5/udd5 | building | 82.6404 | 83.7786 | -1.1382 |
| udd5/udd5 | road | 43.4736 | 44.7628 | -1.2892 |
| udd5/udd5 | vehicle | 16.5053 | 19.4049 | -2.8996 |
| udd5/udd5 | other | 32.7005 | 35.0373 | -2.3368 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0.0000 |
| oem/oem | rangeland | 6.9432 | 6.8975 | +0.0457 |
| oem/oem | developed space | 22.1062 | 22.7224 | -0.6162 |
| oem/oem | road | 29.8108 | 29.6258 | +0.1850 |
| oem/oem | tree | 24.8411 | 27.0871 | -2.2460 |
| oem/oem | water | 16.7931 | 17.0185 | -0.2254 |
| oem/oem | agriculture land | 72.6562 | 72.9856 | -0.3294 |
| oem/oem | building | 60.2237 | 63.4347 | -3.2110 |
| loveda/P | building | 89.5261 | 89.6315 | -0.1054 |
| loveda/P | road | 84.0949 | 85.1967 | -1.1018 |
| loveda/P | water | 68.4816 | 72.6192 | -4.1376 |
| loveda/P | barren | 76.2647 | 68.0139 | +8.2508 |
| loveda/P | tree | 56.3007 | 64.6523 | -8.3516 |
| loveda/P | farm | 68.7169 | 73.1002 | -4.3833 |
| loveda/D | background | 4.5197 | 14.6919 | -10.1722 |
| loveda/D | building | 48.0800 | 48.4265 | -0.3465 |
| loveda/D | road | 63.4927 | 65.9654 | -2.4727 |
| loveda/D | water | 64.6932 | 67.7771 | -3.0839 |
| loveda/D | barren | 28.7369 | 26.2285 | +2.5084 |
| loveda/D | tree | 48.8538 | 52.2329 | -3.3791 |
| loveda/D | farm | 44.8608 | 51.3805 | -6.5197 |
| vaihingen/vaihingen | impervious surface | 64.4438 | 63.9288 | +0.5150 |
| vaihingen/vaihingen | building | 70.0403 | 70.2291 | -0.1888 |
| vaihingen/vaihingen | low vegetation | 30.8945 | 29.7808 | +1.1137 |
| vaihingen/vaihingen | tree | 67.6291 | 68.1148 | -0.4857 |
| vaihingen/vaihingen | car | 27.0071 | 27.2431 | -0.2360 |
| landcoverai/landcoverai | background | 83.9853 | 85.1704 | -1.1851 |
| landcoverai/landcoverai | building | N/A | N/A | N/A |
| landcoverai/landcoverai | woodland | 92.0134 | 92.7243 | -0.7109 |
| landcoverai/landcoverai | water | 92.7996 | 93.8996 | -1.1000 |
| landcoverai/landcoverai | road | 38.1592 | 40.3582 | -2.1990 |
| flair1/flair1 | building | 56.9132 | 59.6299 | -2.7167 |
| flair1/flair1 | pervious surface | 30.4475 | 28.8614 | +1.5861 |
| flair1/flair1 | impervious surface | 68.8619 | 69.6020 | -0.7401 |
| flair1/flair1 | bare soil | 21.9535 | 11.4686 | +10.4849 |
| flair1/flair1 | water | 96.7062 | 95.6854 | +1.0208 |
| flair1/flair1 | coniferous | 7.2359 | 2.8284 | +4.4075 |
| flair1/flair1 | deciduous | 67.2038 | 66.8983 | +0.3055 |
| flair1/flair1 | brushwood | 6.4162 | 8.3984 | -1.9822 |
| flair1/flair1 | vineyard | 57.1268 | 56.9048 | +0.2220 |
| flair1/flair1 | herbaceous vegetation | 37.6746 | 35.3889 | +2.2857 |
| flair1/flair1 | agricultural land | 32.9293 | 31.4099 | +1.5194 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0.0000 |

N/A retains the evaluator's undefined class IoU (zero union); it is not converted to zero.

Prospective engineering/mechanism checks: {"mean_accuracy_loss_le1": true, "worst_protocol_loss_le2": false, "same_source_gain": true, "above_pooled_class": false, "above_word_shuffle": true, "mean_latency_ratio_le3": true}

Three fixed complete images/domain,5 rotated warmed singleton repeats, exclusive GPU7. Other GPU jobs are allowed and recorded; not guaranteed whole-server idle. Extra semantic head cost is included. Both models and graph pools resident: reported memory is not standalone deployment memory. Model/text/graph setup excluded from warm inference.

No full20092 rollout or original-model replacement. All per-class results and matched confusions are in the dataset JSON/NPZ files and summary.json. An inherited proxy head is not claimed as an original module, and these developed samples do not establish SOTA.

## Deployment decision

Mean of eight domain latency ratios: 2.823x VIP; range 1.924-3.103x; 3/8 domains at or below3x. Mean paired latency reduction versus the retained fine model: 49.26%. These are warmed, image-already-loaded inference measurements, including resizing, all branches, alias processing, writeback, restoration and argmax; image decode, checkpoint/text initialization and initial graph capture are excluded.

Keep the retained accuracy model. The candidate is an optional speed/accuracy tradeoff, not a lossless replacement. In particular, the worst-domain loss and the pooled/shuffled controls must accompany any accuracy/innovation claim. Do not choose per-dataset routing from this pilot. Natural-image domains and large class counts have not been validated by this experiment.
