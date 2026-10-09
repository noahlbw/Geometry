# Bounded branch-union alias reuse

Same64 developed windows, not full datasets or independent validation. Fixed Geometry, wide observations, H, checkpoints, all20 vocabularies and templates. Reuse native local features, no fine RGB forwards. Baseline/Geometry/wide top2 union, at most6 classes/patch; no alias-by-all-class-rival tensor. Original Geometry/no-admission/native-top2 per-image confusions replay exactly. Scores precede masks.

| Dataset/protocol | Geometry | NoAdmission_Exact | SparseNativeSoft | SparseNativeUnion6Soft | FineResponsibilityOnly_Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.1743 | 54.3823 | 55.9376 |
| potsdam/potsdam | 35.7035 | 38.9203 | 38.3518 | 38.0662 | 41.9214 |
| udd5/udd5 | 30.5010 | 28.1758 | 29.8826 | 30.3146 | 34.6160 |
| oem/oem | 39.8205 | 39.0232 | 39.0743 | 39.1614 | 41.0076 |
| loveda/P | 49.5399 | 50.7066 | 48.9625 | 46.6336 | 52.5053 |
| loveda/D | 33.8779 | 30.7360 | 31.5169 | 30.8703 | 36.9916 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 51.5842 | 51.7941 | 54.8162 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.1181 | 67.1814 | 68.4598 |
| flair1/flair1 | 35.6059 | 33.6084 | 33.1799 | 32.8819 | 34.4439 |
| Eight-domain mean | 40.5310 | 42.8679 | 43.1103 | 43.0815 | 46.0243 |

New primary delta versus previous native-top2: -0.0287pp; versus slow fine-only: -2.9427pp. LoveDA D counts once; P separately. Same original scored-class support.

## First Complete Image Timing

Three warmed synchronized singleton repeats with stitching/argmax, unchanged CPU accumulator. Both backbones/slow graph resident. Workers use distinct GPUs but share the CPU host; this is not isolated average dataset throughput.

| Domain | No admission ms | Union6 ms | Slow fine-only ms |
| --- | ---: | ---: | ---: |
| vdd | 8053.58 | 9425.47 | 18516.71 |
| potsdam | 1086.72 | 2589.74 | 4344.02 |
| udd5 | 6552.42 | 10458.50 | 20729.67 |
| oem | 528.44 | 855.97 | 1646.31 |
| loveda | 559.79 | 1208.54 | 2089.67 |
| vaihingen | 453.32 | 774.01 | 1536.47 |
| landcoverai | 139.29 | 176.77 | 261.99 |
| flair1 | 157.58 | 196.04 | 295.03 |

Per-class IoU and first-complete-image metrics remain in results/summary JSON. Do not combine a favorable top-left panel with unrelated full-domain claims. No new threshold fitting, domain routing, final-model promotion or full20092 launch follows.
