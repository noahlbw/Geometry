# Natural wide resolution: matched scale-stratum diagnosis

Post-freeze labelled diagnosis only; membership uses RGB dimensions. All stratum confusion sums replay the full results. Within each stratum both methods use the same union-of-support class mask; these are diagnostic scores, not new benchmark results.

| Dataset | Stratum | Images | Frozen matched mIoU | Short-edge matched mIoU | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| voc21 | same | 727 | 70.6437 | 70.6437 | +0.0000 |
| voc21 | increased | 583 | 70.6477 | 70.6374 | -0.0103 |
| voc21 | reduced | 139 | 64.2717 | 64.9993 | +0.7275 |
| context60 | same | 2615 | 41.9928 | 41.9928 | +0.0000 |
| context60 | increased | 1995 | 40.9507 | 41.2141 | +0.2635 |
| context60 | reduced | 495 | 41.0915 | 40.8696 | -0.2218 |
| ade150 | same | 880 | 31.1903 | 31.1903 | +0.0000 |
| ade150 | increased | 585 | 29.6191 | 29.7949 | +0.1758 |
| ade150 | reduced | 535 | 27.4493 | 27.4432 | -0.0061 |

Unchanged-size confusion matrices are identical on VOC21, PC60 and ADE150.

The relationship is not monotonically "higher resolution improves accuracy": VOC21's increased-resolution stratum is essentially flat (-0.0103pp), while the reduced-resolution stratum improves+0.7275pp. PC60 gains+0.2635pp on increased views but loses-0.2218pp on reduced views; ADE gains+0.1758pp on increased views and is almost flat on reduced views. These post-freeze strata do not justify fitting separate per-dataset/aspect-ratio winner policies.

