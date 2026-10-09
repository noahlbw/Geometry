# Shared-family ownership: eight-dataset evaluation

Status: paused by user after UDD5 and OEM completed. This is a frozen exploratory evaluation, not an
eight-dataset improvement claim. The candidate uses the same fixed 20 aliases
per class as the earlier GEAR-OV full runs and does not use target labels to
select aliases or parameters. Model signature:
`geometry-shared-family-ownership-v2-20260930`.

## Implementation

The new readout gives each near-identical text family one spatial ownership
distribution. The supplied class label is used only when reading that family's
contribution to its class. The semantic reference omits the whole family and
averages remaining families rather than counting every synonymous alias as an
independent vote. Local, detail and context observations are judged separately,
corrected separately, and fused in the same order as the matched Multiscale
baseline. The original alias count and class-score denominator are retained.

The shared-family, source-swap, and exact fallback behavioral checks passed
on the A800 PyTorch environment. A one-image GPU smoke completed. The current
readout is an implementation candidate, not a validated final model.

## Completed: UDD5

Four shards completed and merged: 40/40 unique images with verified coverage.
The global sample-key and vocabulary SHA match the prior full all-20 run.

| Method | Full mIoU | Non-residual mIoU | Vegetation | Building | Road | Vehicle | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geometry | 50.5553 | 55.5480 | 82.4937 | 83.9035 | 45.7856 | 10.0092 | 30.5846 |
| Multiscale | 51.0405 | 55.9955 | 83.4756 | 85.1348 | 45.6545 | 9.7170 | 31.2206 |
| COR_Shared | 50.0013 | 55.0990 | 83.4467 | 84.8429 | 43.8888 | 8.2176 | 29.6106 |

COR_Shared is 1.0392 mIoU below its matched Multiscale baseline. Mean soft
rejection is 0.4620 and mean unresolved ownership is 0.0051. This diagnostic
indicates much more intervention than COR v1, but does not by itself identify
which rejected aliases caused the errors. Parallel wall time was 278.5 s and
peak allocated CUDA memory was 3883.2 MiB. Source:
`research/shared_family_ownership_udd5_full_20260930_results.json`.

## Completed: OEM

Four shards completed and merged: 384/384 unique validation images. Global
sample-key and vocabulary SHA match the prior full all-20 run. Geometry,
Multiscale and COR_Shared mIoU are 44.6097, 45.6498 and 45.5522. The new
readout is 0.0976 point below its direct Multiscale baseline. Per-class IoU
for COR_Shared is bareland 11.9416, rangeland 30.1053, developed space
29.9514, road 38.5468, tree 57.1653, water 70.6183, agriculture land
62.6947 and building 63.3940. Mean soft rejection is 0.5779 and mean
unresolved ownership is 0.0035. Parallel wall time was 242.8 s; peak CUDA
memory was 3895.8 MiB. Source:
`research/shared_family_ownership_oem_full_20260930_results.json`.

## Remaining datasets

FLAIR-1 (15,700 images) and VDD (80) were interrupted at the user's request.
Their incomplete shard outputs and logs remain on A800 and are not full-dataset
results. Potsdam (504), Vaihingen (113), LandCover.ai (1,602) and LoveDA
(1,669, P/D) were not launched. The monitoring automation is paused. An
eight-dataset table requires complete, verified merges before it can be
reported. iSAID is not part of this
eight-dataset protocol because labeled masks were unavailable in the earlier
study; LandCover.ai is the eighth labeled dataset.
