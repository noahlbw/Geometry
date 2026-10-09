# One-sided soft alias: execution-only acceleration

Same96 developed complete RS inputs, fixed20 aliases, unchanged Geometry/wide/fine observations, risk rule and reconstruction. UDD5 full40;8 images/other domain. This is not a new semantic method or independent full-dataset validation.

| Domain/protocol | Original soft mIoU | Stream mIoU | Delta pp |
| --- | ---: | ---: | ---: |
| vdd/vdd | 49.8732 | 49.8732 | +0.0000 |
| potsdam/potsdam | 55.1778 | 55.1778 | +0.0000 |
| udd5/udd5 | 51.2305 | 51.2305 | +0.0000 |
| oem/oem | 29.9714 | 29.9714 | +0.0000 |
| loveda/P | 75.5356 | 75.5356 | +0.0000 |
| loveda/D | 46.6718 | 46.6718 | +0.0000 |
| vaihingen/vaihingen | 51.8593 | 51.8593 | +0.0000 |
| landcoverai/landcoverai | 78.0381 | 78.0381 | +0.0000 |
| flair1/flair1 | 38.9230 | 38.9230 | +0.0000 |

Equal-domain means (LoveDA D once): {"Geometry_PatchOnly2Coupled": 46.645025000000004, "OneSide_ObservationMean": 49.8154375, "OneSide_Soft": 50.2181375, "OneSide_Stream": 50.2181375}

| Domain | Baseline ms | Same-fine no attenuation ms | Original soft ms | Stream ms | Speed gain % | VIP20 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 453.36 | 803.15 | 860.42 | 845.80 | 1.70 | 189.30 |
| potsdam | 232.09 | 580.11 | 673.83 | 645.83 | 4.16 | 99.81 |
| udd5 | 412.18 | 705.02 | 738.92 | 727.06 | 1.61 | 175.75 |
| oem | 239.29 | 587.73 | 685.68 | 658.77 | 3.92 | 102.96 |
| loveda | 259.93 | 664.49 | 858.49 | 803.08 | 6.46 | 208.71 |
| vaihingen | 239.91 | 589.28 | 683.13 | 654.57 | 4.18 | 103.05 |
| landcoverai | 225.07 | 573.11 | 663.47 | 635.80 | 4.17 | 95.13 |
| flair1 | 231.30 | 585.46 | 692.67 | 668.69 | 3.46 | 101.11 |

All timing complete: True; mean paired latency ratio: 0.9629345748365596.

Both soft arms include the previously established cached-burst fine execution. Timings are complete original-resolution predictions on3 fixed images/domain,5 warmed singleton repeats, serial exclusive GPU7; not full throughput. Model/text/graph setup excluded from warm timing, graph setup recorded in each timing.json. Memory is shared-resident, not isolated deployment memory.

All96 complete-image candidate/reference predictions are identical, and every per-image confusion matches the archived original. Patch-score FP64 rounding is reported separately in summary.json. No accuracy gain or improved word-specific attribution is claimed. The prior class-calibration counterexample remains; this execution change cannot establish CVPR novelty.

## Result

Mean paired complete-image latency reduction is 3.71%, with domain means 635.80-845.80ms. Accuracy is unchanged on the96 developed inputs. This is a modest execution gain, not an accuracy improvement or a new alias rule. The same-fine no-attenuation cost remains substantial; optimizing this word reduction alone does not remove the main visual-observation cost or close the gap to VIP. Preserve the candidate as an optional execution backend; do not promote a new semantic method from this experiment.

## Isolated reader and shared-resident memory

Reader time replays the first image/tile with already computed source observations; it excludes image encoding, source assembly and restoration. Memory below is from the whole-image benchmark and includes both models, validation snapshots, captured graph pools and one retained profiling tile. Compare arms within this harness only.

| Domain | Original reader ms | Stream reader ms | Original peak MiB | Stream peak MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 8.397 | 4.362 | 5897.64 | 5897.64 |
| potsdam | 11.576 | 4.780 | 5798.89 | 5798.89 |
| udd5 | 7.670 | 3.574 | 5764.68 | 5764.68 |
| oem | 12.612 | 6.028 | 5847.27 | 5847.27 |
| loveda | 11.913 | 4.910 | 5864.93 | 5864.93 |
| vaihingen | 11.401 | 4.346 | 5777.17 | 5777.17 |
| landcoverai | 10.952 | 4.208 | 5777.17 | 5777.17 |
| flair1 | 15.482 | 9.417 | 5991.48 | 5991.48 |

All panel patch scores bitwise identical: True. Three focused CPU tests and the same three CUDA tests passed, including extreme-logit fallback, noncontiguous alias groups, padding, multiple crops/stencils and rival block boundaries. Both mask-free checkpoint smokes and singleton-versus-joint prediction checks passed.

## Timing environment amendment

The original all-server-idle guard stopped scheduling after UDD5 because unrelated jobs started on GPUs0,1,3. No worker or result failed and no unrelated job was stopped. Remaining datasets use exclusive GPU7 with paired rotated methods while other GPUs may be busy. Background host/power contention remains possible; these are not all-server-idle measurements. Environment snapshots are retained per dataset. Do not interpret small cross-dataset absolute-time differences as model effects. No semantic rule was altered.
