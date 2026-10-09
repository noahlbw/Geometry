# Matched VIP/Geometry component diagnosis on VDD and Potsdam

Date: 2026-09-30. This is an exploratory diagnosis, not a new segmentation method or a claim of independent validation. The pinned VIP source commit is `5bd25ee03ec25c1538622cf7da661e8c0461e769`; frozen DINOv3/DINO.text checkpoints and dataset-specific 20-alias vocabularies are unchanged. Target labels enter confusion matrices and paired error counts only.

All full results have verified unique coverage and matching global sample-key SHA against the historical runs: VDD 80 images, Potsdam 504 prepared tiles. Four A800 GPUs (4-7) ran the paired experiments; all were idle at the final check. All matched arms use the same 336-pixel crops, stride 112, raw-logit averaging, original-size evaluation masks, and no confidence-to-background rule. The 448/1024 factor specifies the image's resized long edge before those crops. On the matched arms, the two heads use the **same encoded alias vectors**. Therefore, comparisons within each table isolate the stated factor; the old native methods do not share all these settings.

## Visual head by field of view

Six remote-sensing templates are averaged into one feature per alias, and both heads use normalized log-mean-exp (LME, tau 0.07). Values are full-resolution mIoU percent on the complete scored set.

| Dataset | Geometry 448 | VIP 448 | Geometry 1024 | VIP 1024 |
|---|---:|---:|---:|---:|
| VDD | 42.8211 | 46.3049 | 41.3052 | 36.1873 |
| Potsdam | 27.5517 | 32.7554 | 28.0671 | 31.0176 |

VDD VIP loses 10.1176 points when the same 336 crop observes a smaller physical field in the 1024 image; Geometry loses 1.5159. VDD VIP water IoU falls 85.11 -> 41.31 and roof 78.50 -> 58.43. Potsdam VIP loses 1.7378 points at 1024, whereas Geometry gains 0.5154. These are field-of-view changes within this matched 336-crop protocol, not changes to the historical 512/128 Geometry-Multiscale evaluator.

The fixed-seed 16-image VDD and 32-tile Potsdam screens showed the same direction and were followed by the unmodified full configurations. Their merged results remain separately saved. Full matched outputs: `matched_head_fov_vdd_full_20260930_results.json` and `matched_head_fov_potsdam_full_20260930_results.json`.

## Visual head by class aggregation

The view is fixed at long edge 448, the **same six-template alias vectors** feed both heads, and the background threshold stays off. VIPScore means its pooled-patch alias salience, weight-scaled alias logits, and tau log-sum-exp (without the original ImageNet template vectors).

| Dataset | Geometry + LME | VIP + LME | Geometry + VIPScore | VIP + VIPScore |
|---|---:|---:|---:|---:|
| VDD | 42.8211 | 46.3049 | 42.7055 | 47.8037 |
| Potsdam | 27.5517 | 32.7554 | 30.0981 | 36.0664 |

VIPScore helps the VIP head by 1.4988 points on VDD and 3.3110 on Potsdam; for Geometry it changes VDD by -0.1156 and Potsdam by +2.5464. The LME controls reproduce the previous table to four decimal places. The sum of absolute confusion-cell differences is at most 54 among 960 million VDD pixels and 34 among 504 million Potsdam pixels because one diagnostic computes argmax after positive temperature scaling and the other before it; the mIoU values agree. Complete outputs: `matched_head_score_vdd_full_20260930_results.json` and `matched_head_score_potsdam_full_20260930_results.json`.

## Visual head by text representation

VIPScore and the 448 view are fixed. RS uses the averaged feature from six remote-sensing templates per alias; ImageNet uses VIP's pinned OpenAI ImageNet templates with per-template similarity averaging. Alias strings and parent classes are **exactly identical** within each dataset. Therefore this factor includes template wording and the way template features are retained; it is not a pure prompt-phrasing intervention.

| Dataset | Geometry + RS | VIP + RS | Geometry + ImageNet | VIP + ImageNet |
|---|---:|---:|---:|---:|
| VDD | 42.7055 | 47.8037 | 41.7415 | **51.0697** |
| Potsdam | 30.0981 | 36.0664 | 36.0449 | **42.3519** |

The two RS controls reproduce the preceding scorer table. `VIP + ImageNet` reproduces the historical native VIP all-20, threshold-off **confusion matrices exactly** on both datasets. On VDD the VIP-versus-Geometry head gain is 5.0982 points with RS text but 9.3282 with ImageNet text: a 4.2300-point head/text interaction. On Potsdam the corresponding gains are 5.9683 and 6.3070 points. ImageNet text helps both heads on Potsdam but helps only VIP on VDD.

Per-class complements are substantial: VDD vehicle IoU is 26.90 for Geometry + ImageNet versus 23.09 for VIP + ImageNet, while water is 71.62 versus 83.24. Potsdam car is 39.17 versus 29.31, but low vegetation is 9.91 versus 30.15 and tree 27.94 versus 53.52. These IoUs do not imply that selecting a head by ground-truth class would work: changing one class's predicted area changes other classes' false positives. Full outputs: `matched_head_text_vdd_full_20260930_results.json` and `matched_head_text_potsdam_full_20260930_results.json`.

## VIP official-short alias-count offset

Potsdam's pinned short queries have counts `[2,1,3,2,1,1]`. Holding VIP's exact predictions, features and queries fixed, subtract `log(N_c)/tau` from each class logit before applying softmax/threshold. `LSE` is VIP's original aggregation; `LME` removes this count term. This only tests an algebraic score offset; it does not change or select words.

| Potsdam full 504 tiles | Threshold off | VIP official threshold 0.25 on |
|---|---:|---:|
| Original LSE | 43.5166 | **44.0643** |
| Count-normalized LME | 44.3838 | 43.5188 |

The original LSE arms exactly reproduce VIP's prior complete 504-tile results, and sample-key SHA matches. Without the threshold, removing count offset raises car IoU 16.85 -> 28.46 but lowers low vegetation 50.79 -> 39.23. With the fixed threshold, the net mIoU change reverses. This demonstrates that official-short performance cannot be attributed to lexical selection alone. Full output: `vip_count_offset_potsdam_full_20260930_results.json`.

## Research decision

The current strongest complete native references remain Geometry-Multiscale all20 at 41.2235 VDD / 40.4433 Potsdam, and upstream-configured VIP short-query/threshold at 52.0647 / 44.0643. None of the matched tables is a newly proposed method; their absolute scores should not be presented as improvements over the historical systems.

The controlled results point to a coupled problem. VDD gains from VIP's broad-field head and its ImageNet text representation together; a finer-view VIP readout collapses water and roof. Potsdam gains from the VIP head, its salience-weighted aggregation, and its text representation, while a count offset/threshold tradeoff changes car versus low-vegetation errors. A viable next model should compare an independently computed broad-view semantic observation with local Geometry evidence, then correct the *active class margin* in visually supported regions. It must show positive and harmful changed-pixel counts against both the strong Multiscale and VIP references, with a frozen rule transferred beyond these two development sets. Naive class-specific winner selection from these validation IoUs would be label leakage.

Combined inference cost (two shards per dataset) was 156.22/884.78 s parallel wall time for the head/view VDD/Potsdam runs, 70.91/86.60 s for head/score, and 71.51/109.41 s for head/text. The Potsdam count-offset run took 93.65 s on one GPU. Peak CUDA allocations and exact checkpoint/vocabulary signatures are in each merged JSON. The longer 1024-view Potsdam run is expected because it evaluates many more 336 crops.
