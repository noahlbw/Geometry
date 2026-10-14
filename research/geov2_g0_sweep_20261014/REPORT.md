# Geov2 fixed structure writeback: ten-protocol g0 sweep

Full model: 2026-10-13 retained best Alias + QuerySlot (24 slots) + DonorBefore, with original local strength on every dataset. PC60 replaces LandCover.ai. All other task-specific calibration stays frozen.

Writer: Z = L + g0 H(W-L). g0=0 is the local endpoint; g0=1 is the unrelaxed quadratic optimum. The wide endpoint and literal equal-logit mean are separate controls. Each image shares visual observations across all 30 arms; masks are used only after inference.

16,124 protocol images, ten complete splits. VOC20/VOC21 share 1,449 images. Each dataset mIoU comes from its summed confusion matrix; means below weight protocols equally.

## Retained Alias: full split mIoU (%)

| Dataset | N | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 | Equal mean | Wide | Best g0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VDD | 80 | 50.9434 | 54.8898 | 58.0367 | 59.5457 | 59.9781 | 58.9907 | 57.9078 | 55.6405 | 59.5918 | 57.0648 | 1 |
| Potsdam | 504 | 38.2874 | 48.7679 | 54.4385 | 54.4724 | 51.8644 | 46.3901 | 42.6734 | 38.4350 | 51.5642 | 42.7619 | 0.75 |
| UDD5 | 40 | 51.1651 | 54.4283 | 55.2567 | 51.6318 | 43.1212 | 30.7991 | 22.3378 | 15.5770 | 42.5750 | 22.2537 | 0.5 |
| OEM | 384 | 44.2947 | 44.7354 | 43.7090 | 41.8539 | 39.4550 | 34.0712 | 29.6684 | 24.0278 | 38.7022 | 28.7751 | 0.25 |
| Vaihingen | 113 | 58.0583 | 58.8018 | 57.5554 | 54.9732 | 51.8264 | 46.3648 | 42.6989 | 38.6462 | 50.9289 | 42.6777 | 0.25 |
| PC60 | 5105 | 31.6589 | 40.6747 | 44.1406 | 44.5929 | 44.0788 | 42.5743 | 40.8893 | 37.4295 | 44.0181 | 40.8293 | 0.75 |
| ADE150 | 2000 | 27.8471 | 31.4851 | 30.7014 | 29.6812 | 28.8769 | 27.5379 | 26.6112 | 25.3533 | 28.8492 | 26.6347 | 0.25 |
| VOC20 | 1449 | 83.0869 | 90.9307 | 92.5337 | 93.0960 | 93.2805 | 93.2139 | 93.0180 | 92.5851 | 93.1699 | 92.8069 | 1 |
| VOC21 | 1449 | 41.4698 | 61.4669 | 72.1656 | 74.6881 | 75.2094 | 74.5100 | 70.9306 | 61.2277 | 74.8929 | 70.9990 | 1 |
| COCO-Object81 | 5000 | 33.4590 | 46.6243 | 49.6209 | 46.7791 | 42.3194 | 38.4352 | 38.5930 | 38.3068 | 42.1360 | 38.6809 | 0.5 |
| Equal-protocol mean | — | 46.0271 | 53.2805 | 55.8158 | 55.1314 | 53.0010 | 49.2887 | 46.5328 | 42.7229 | 52.6428 | 46.3484 | 0.5 |

Best shared g0 = **0.5**, mean **55.8158%**. Per-dataset selection reference = 56.7433%; shared-step cost = 0.9275 pp. These best values use validation labels for analysis; inference itself has no labels.

The shared step exceeds equal fusion on 7/10 protocols; it is lower on VDD, VOC20, VOC21. The positive macro difference is not an all-protocol improvement.

| Paired comparison (shared g0) | Mean difference (pp) | Conditional 95% interval (pp) |
| --- | --- | --- |
| retained__g0.5 vs retained__g0.75 | 0.6844 | [0.5276, 0.8685] |
| retained__g0.5 vs retained__equal_mean | 3.1730 | [2.8311, 3.5425] |

Intervals resample images within each dataset and pair shared VOC20/VOC21 images. They are conditional on the inspected candidates; selection of the winning g0 is not accounted for.

## Held-out g0 selection

| Dataset | Best g0 | Shared-step loss (pp) | LODO g0 | LODO mIoU | Grouped LODO g0 | Grouped LODO mIoU |
| --- | --- | --- | --- | --- | --- | --- |
| VDD | 1 | 1.9414 | 0.5 | 58.0367 | 0.5 | 58.0367 |
| Potsdam | 0.75 | 0.0340 | 0.5 | 54.4385 | 0.5 | 54.4385 |
| UDD5 | 0.5 | 0.0000 | 0.5 | 55.2567 | 0.5 | 55.2567 |
| OEM | 0.25 | 1.0264 | 0.5 | 43.7090 | 0.5 | 43.7090 |
| Vaihingen | 0.25 | 1.2464 | 0.5 | 57.5554 | 0.5 | 57.5554 |
| PC60 | 0.75 | 0.4523 | 0.5 | 44.1406 | 0.5 | 44.1406 |
| ADE150 | 0.25 | 0.7837 | 0.5 | 30.7014 | 0.5 | 30.7014 |
| VOC20 | 1 | 0.7468 | 0.5 | 92.5337 | 0.5 | 92.5337 |
| VOC21 | 1 | 3.0438 | 0.5 | 72.1656 | 0.5 | 72.1656 |
| COCO-Object81 | 0.5 | 0.0000 | 0.5 | 49.6209 | 0.5 | 49.6209 |

LODO mean = 55.8158%; grouped LODO mean = 55.8158%. Grouped selection holds VOC20/VOC21 out together. Only g0 selection is held out: previously selected aliases/calibration are still task-specific.

Weighting the VOC20/VOC21 pair as one group gives nine-group best g0=0.5, mean=52.8676%. This is a sensitivity analysis; the declared primary remains ten equally weighted protocols.

## Alias interfaces

retained keeps all aliases. canonical keeps each class’s first canonical phrase and the same cached templates. count_normalized keeps all aliases and existing scores, subtracting log(alias_count)/tau only from wide class LSE. The current QuerySlot CompactContextScorer has salience disabled; the incumbent VIP salience is not used here. Local class LME already normalizes alias count. PC60 keeps all 401 residual concepts and residual_max in every arm. This control isolates the explicit wide count offset, not every statistical effect of having more words.

| Interface | Best shared g0 | Best shared mean | Mean at retained shared g0 | Per-dataset reference |
| --- | --- | --- | --- | --- |
| retained | 0.5 | 55.8158 | 55.8158 | 56.7433 |
| canonical | 0.5 | 46.7308 | 46.7308 | 47.3638 |
| count_normalized | 0.5 | 50.9657 | 50.9657 | 51.7303 |

| Dataset | Mean alias count | Count CV | retained best g0 | canonical best g0 | normalized best g0 |
| --- | --- | --- | --- | --- | --- |
| VDD | 1.29 | 0.351 | 1 | 0.25 | 0.5 |
| Potsdam | 1.83 | 0.582 | 0.75 | 0.5 | 0.75 |
| UDD5 | 6.20 | 1.169 | 0.5 | 0 | 0 |
| OEM | 20.00 | 0.000 | 0.25 | 0.25 | 0.25 |
| Vaihingen | 13.80 | 0.244 | 0.25 | 1 | 0 |
| PC60 | 4.40 | 0.907 | 0.75 | 0.5 | 0.5 |
| ADE150 | 2.89 | 0.706 | 0.25 | 0.25 | 0.25 |
| VOC20 | 5.35 | 0.621 | 1 | 0.25 | 0.75 |
| VOC21 | 5.67 | 2.120 | 1 | 0.25 | 0.5 |
| COCO-Object81 | 1.93 | 1.959 | 0.5 | 0.5 | 0.5 |

## Dataset family and descriptive relationships

| Family | Best shared g0 | Mean mIoU |
| --- | --- | --- |
| remote_sensing | 0.5 | 53.7993 |
| natural | 0.5 | 57.8324 |

| Covariate | Spearman correlation with retained optimal g0 |
| --- | --- |
| wide_minus_local_pp | 0.819 |
| mean_alias_count | -0.444 |
| alias_count_cv | 0.319 |
| correction_energy_ratio | 0.094 |

Ten correlated protocols are insufficient to infer causality or establish a reliable label-free g0 selector. Endpoint advantage uses labels and is explanatory only. Canonical changes both phrase diversity and count; its calibration was not re-tuned. A single g0 does not imply a globally fixed full model because temperatures, vocabularies and background rules remain task-specific.

## Figures

![Dataset curves](g0_dataset_curves.png)

![Protocol means](g0_macro_curves.png)

## Reproduce and audit

| Dataset | Original g0 | 10/13 mIoU | Current mIoU | Difference (pp) |
| --- | --- | --- | --- | --- |
| VDD | 1.0 | 59.9810 | 59.9781 | -0.0029 |
| Potsdam | 0.5 | 54.4390 | 54.4385 | -0.0006 |
| UDD5 | 0.5 | 55.2575 | 55.2567 | -0.0008 |
| OEM | 0.25 | 44.7355 | 44.7354 | -0.0000 |
| Vaihingen | 0.25 | 58.8021 | 58.8018 | -0.0002 |
| PC60 | 0.75 | 44.5927 | 44.5929 | +0.0002 |
| ADE150 | 0.25 | 31.4852 | 31.4851 | -0.0001 |
| VOC20 | 1.0 | 93.2804 | 93.2805 | +0.0000 |
| VOC21 | 0.75 | 74.6882 | 74.6881 | -0.0002 |
| COCO-Object81 | 0.5 | 49.6209 | 49.6209 | -0.0000 |

Historical references use the same full splits and target support. Tiny differences can arise from CUDA reductions and slot clustering; current paired arms share one set of observations.

Run tools/collect_geov2_g0.py, then python tools/report_geov2_g0.py. protocol.json records the predeclared scan; expected_keys.json records authoritative full split order. full/*/statistics.npz stores per-image TP, predicted and target supports for each arm, sufficient to recompute dataset mIoU. The audit verifies full split keys, all 29 shards, aggregate matrices, class supports and the unchanged normalized local endpoint. Evaluated scripts are retained under source/scripts/. configs_shared_g0/ contains opt-in deployment profiles for the selected scalar, with labeled-selection provenance; use --wide-branch query_slot --efficient --writeback coupled with the frozen inference/evaluation entry points. No latency claim is derived from the 30-arm sweep.
