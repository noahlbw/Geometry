# Bounded Fine Execution: Exactness And Complete-Image Cost

Same semantic model, fixed20 words, Geometry896/wide448, at most16 fine512 views. Three fixed complete inputs/domain,24 total; no labels loaded. Cached and cached-burst risks and reader scores verify BITWISE at every tile/protocol; full predictions and diagnostics also agree. Frozen heads, checkpoints, vocabularies and forward caps verify. This does not add accuracy or fix the preceding coverage candidate's failed semantic gate.

| Domain | Legacy ms | Cached ms | Cached+burst ms | Exact speed gain % |
| --- | ---: | ---: | ---: | ---: |
| vdd | 1018.67 | 851.34 | 745.15 | 26.85 |
| potsdam | 1120.56 | 780.63 | 683.19 | 39.03 |
| udd5 | 960.62 | 820.65 | 742.05 | 22.75 |
| oem | 1125.87 | 785.86 | 690.97 | 38.63 |
| loveda | 1609.17 | 936.14 | 845.88 | 47.43 |
| vaihingen | 1133.54 | 789.28 | 686.06 | 39.48 |
| landcoverai | 1128.11 | 781.09 | 678.84 | 39.82 |
| flair1 | 1148.93 | 815.02 | 721.29 | 37.22 |

Mean paired ratios: cache0.720559; cache+burst0.635977.

These are UNINSTRUMENTED three-repeat warmed synchronized rotated whole-image timings, not one-window timing, full-suite throughput or standalone memory. Cold graph setup is excluded only from warm latency and recorded below.

## Synchronized Stage Attribution

Each interval includes host dispatch and device completion. Extra barriers perturb execution; these are not pure kernel-time percentages. Nested fine acquisition is counted once via subtraction. Unattributed includes output score computation, upsampling/softmax, dispatch and other uninstrumented work.

| Domain | Arm | Geometry prepare | Wide read | Fine read | Fine assembly | Alias risk/write | Solver | Restore | Other ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | legacy | 93.76 | 45.22 | 346.78 | 54.94 | 274.75 | 9.95 | 2.24 | 191.49 |
| vdd | cached | 93.57 | 45.23 | 346.84 | 55.02 | 103.53 | 9.92 | 2.34 | 191.00 |
| vdd | cached_burst | 93.67 | 45.24 | 246.36 | 55.02 | 103.75 | 9.92 | 2.15 | 191.25 |
| potsdam | legacy | 93.23 | 88.01 | 343.34 | 53.76 | 482.04 | 9.96 | 0.57 | 52.93 |
| potsdam | cached | 92.87 | 87.88 | 341.76 | 53.68 | 143.17 | 9.95 | 0.42 | 53.17 |
| potsdam | cached_burst | 93.03 | 87.95 | 246.25 | 53.72 | 143.49 | 9.96 | 0.41 | 52.62 |
| udd5 | legacy | 79.22 | 45.83 | 284.74 | 44.63 | 222.34 | 8.54 | 1.83 | 275.37 |
| udd5 | cached | 79.51 | 46.02 | 284.58 | 44.59 | 82.63 | 8.56 | 1.87 | 276.65 |
| udd5 | cached_burst | 79.33 | 45.73 | 205.36 | 44.71 | 82.77 | 8.55 | 1.81 | 275.20 |
| oem | legacy | 93.04 | 87.66 | 340.99 | 55.32 | 488.37 | 9.98 | 0.57 | 54.91 |
| oem | cached | 92.65 | 87.56 | 340.71 | 55.38 | 147.79 | 9.97 | 0.41 | 55.09 |
| oem | cached_burst | 92.90 | 87.74 | 246.17 | 55.52 | 148.31 | 9.99 | 0.42 | 55.08 |
| loveda | legacy | 92.26 | 86.69 | 336.99 | 60.41 | 954.62 | 9.91 | 1.24 | 69.60 |
| loveda | cached | 91.86 | 86.68 | 336.91 | 60.34 | 285.12 | 9.95 | 0.87 | 69.19 |
| loveda | cached_burst | 92.90 | 87.69 | 246.36 | 60.68 | 284.41 | 9.95 | 0.85 | 73.69 |
| vaihingen | legacy | 94.82 | 89.63 | 362.80 | 55.58 | 506.42 | 10.00 | 0.59 | 53.76 |
| vaihingen | cached | 94.77 | 89.18 | 348.80 | 53.64 | 143.09 | 9.98 | 0.44 | 52.40 |
| vaihingen | cached_burst | 94.16 | 89.18 | 246.28 | 53.94 | 143.21 | 9.97 | 0.43 | 52.84 |
| landcoverai | legacy | 94.33 | 88.45 | 348.66 | 54.11 | 490.22 | 9.96 | 0.47 | 45.70 |
| landcoverai | cached | 94.18 | 88.25 | 348.41 | 54.22 | 144.62 | 9.93 | 0.33 | 44.53 |
| landcoverai | cached_burst | 93.87 | 88.30 | 246.03 | 54.07 | 144.69 | 9.91 | 0.31 | 44.56 |
| flair1 | legacy | 92.22 | 86.27 | 340.80 | 57.65 | 509.67 | 9.95 | 0.33 | 54.14 |
| flair1 | cached | 92.13 | 86.22 | 340.07 | 57.68 | 178.77 | 9.93 | 0.31 | 53.97 |
| flair1 | cached_burst | 92.04 | 86.05 | 246.41 | 57.74 | 178.39 | 9.93 | 0.30 | 54.82 |

## Setup And Shared Memory

Graph setup forwards are real warmup/capture work, not part of the per-inference16 cap. Memory includes both backbones, frozen-head verification snapshots and captured graph pools; this is not isolated deployment memory.

| Domain | Graph setup seconds | Setup forwards | Legacy peak MiB | Burst peak MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.476 | 16 | 5844.52 | 5844.52 |
| potsdam | 0.466 | 16 | 5746.35 | 5746.81 |
| udd5 | 0.460 | 16 | 5729.50 | 5728.86 |
| oem | 0.467 | 16 | 5768.15 | 5768.15 |
| loveda | 0.461 | 16 | 5805.34 | 5805.34 |
| vaihingen | 0.476 | 16 | 5737.30 | 5736.61 |
| landcoverai | 0.473 | 16 | 5737.30 | 5736.61 |
| flair1 | 0.509 | 16 | 5862.82 | 5921.07 |

17 focused CPU tests pass locally (import-only cv2 stand-in) and on A800 with real dependencies. No model/risk/threshold/word rules changed. Execution speedup is not semantic innovation or permission to promote the coverage model; keep its accuracy/cost failures visible.
