# Frozen Geometry deployment: execution-only optimization

2026-10-10. Architecture, frozen per-task configurations, selected words, visual budget and numerical dtypes are unchanged. This optimizes the retained Kev-selected model; it does not select a new model from the full results.

## Changes

- Remove both text towers from online GPU residence after encoding. The portable API drops them before CUDA loading when a matching text cache already exists, and releases their CPU storage after fresh encoding.
- Compute only the requested Geometry head: omit unused native diagnostic heads and unused alternate Geometry heads.
- Cache wide text conversions, normalized text means and integer class groups. Keep template-level BF16 multiplication and the original within-class reduction; no averaging-before-multiplication approximation.
- Transfer final class IDs as uint8 for these <=256-class protocols. Defer the unused PC60 intermediate prediction/CPU copy. Class indices and calibrated decisions are unchanged.
- Keep Geometry FP32 and wide FP16 visual weight copies. Sharing them would change rounding. Keep original local/wide crops, alias counts, thresholds, FP64 reconstruction and output interpolation.

## Paired whole-image costs

Same A800, same seven deterministic whole RGB inputs per protocol, two warmups and seven synchronized repeats. Includes resize, all encoding/readout, reconstruction, full-size restoration and CPU prediction; excludes decode/init/offline text and Kev selection. The baseline and compact are measured in the same process in successive phases. VDD/UDD5 were additionally repeated alone after observed system-level timing variance; their isolated measurements are primary below when present. Other domains were measured while independent GPU jobs could share the host CPU. VIP is the earlier same-text native measurement on the same samples, not a simultaneous remeasurement or a paper number. Peak is CUDA allocated MiB during warmed online inference; initialization and first vocabulary encoding are separate costs.

| Protocol | Reference ms | Compact ms | Latency reduction | Reference MiB | Compact MiB | VIP ms | Compact/VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 333.30 | 317.05 | +4.9% | 5993.5 | 2847.4 | 190.85 | 1.661 |
| potsdam | 247.59 | 229.28 | +7.4% | 5515.8 | 2156.6 | 104.11 | 2.202 |
| voc20 | 203.13 | 181.55 | +10.6% | 5514.3 | 2153.3 | 58.16 | 3.121 |
| voc21 | 193.63 | 174.91 | +9.7% | 5517.0 | 2158.4 | 59.50 | 2.940 |
| ade150 | 295.76 | 264.69 | +10.5% | 6433.2 | 3296.8 | 113.98 | 2.322 |
| context60 | 265.04 | 236.60 | +10.7% | 5624.8 | 2484.3 | 112.06 | 2.111 |
| udd5 | 319.07 | 303.75 | +4.8% | 5825.4 | 2677.3 | 191.15 | 1.589 |
| oem | 247.81 | 228.98 | +7.6% | 5540.3 | 2227.7 | 105.38 | 2.173 |
| vaihingen | 240.22 | 228.92 | +4.7% | 5519.8 | 2183.0 | 108.55 | 2.109 |
| landcoverai | 234.73 | 217.34 | +7.4% | 5529.2 | 2198.4 | 102.71 | 2.116 |
| loveda | 235.49 | 217.84 | +7.5% | 5532.1 | 2217.2 | 105.86 | 2.058 |
| context59 | 258.69 | 227.71 | +12.0% | 5567.8 | 2424.9 | 111.84 | 2.036 |
| coco_object81 | 286.07 | 250.99 | +12.3% | 5799.4 | 2700.4 | 131.46 | 1.909 |
| coco_stuff171 | 376.62 | 334.49 | +11.2% | 6537.0 | 3399.1 | 185.87 | 1.800 |
| flair1 | 242.27 | 226.81 | +6.4% | 5548.5 | 2265.2 | 111.14 | 2.041 |

Initial simultaneous host-use measurements are retained, including negative speed outcomes:

| Dataset | Initial reference ms | Initial compact ms | Isolated reference ms | Isolated compact ms |
| --- | ---: | ---: | ---: | ---: |
| vdd | 342.07 | 425.09 | 333.30 | 317.05 |
| udd5 | 333.72 | 356.60 | 319.07 | 303.75 |

This change of scheduling regime shows sensitivity to host/system conditions; it does not establish a specific causal bottleneck. Do not choose arbitrary best repetitions or claim uniform acceleration under contention. The earlier implementation-only phase is retained as `sample_*.json`. Its UDD5 successive-phase timing was anomalous (336.10 ->428.27ms); an alternating-order check with both text towers on CPU measured470.03 ->456.67ms. Do not erase the negative phase or interpret its difference as a precision change. Final endpoint costs above additionally include the byte-label/deferred-output optimization.

## Accuracy verification

All15 protocols: seven deterministic full images each, zero changed prediction pixels between reference and compact. This is sampled agreement, not full15-protocol accuracy re-evaluation. Five complete protocols also replayed the optimized visual/readout path and matched every stored per-image confusion matrix:

| Dataset | Full images | mIoU | Per-image confusion |
| --- | ---: | ---: | --- |
| vdd | 80 | 60.4352 | exact |
| potsdam | 504 | 56.1934 | exact |
| udd5 | 40 | 55.8266 | exact |
| voc21 | 1449 | 69.4757 | exact |
| ade150 | 2000 | 31.3398 | exact |

The final uint8 transfer is a lossless class-ID conversion for the frozen <=256-class protocols; final15-protocol sampled checks cover that endpoint. Full checks used historical text caches; they do not assert that all re-encoded caches on other machines are bit-identical.

Three focused CPU tests cover exact requested Geometry head outputs, text release preserving visual weights, and deferred prediction preserving probabilities.

## Independent reproduction entry

YAML/configuration and actual selected words are exported for15 protocols. `infer_frozen_geometry.py` and `evaluate_frozen_geometry.py` build directly from checkpoint/words without historical experiment directories. Fresh-encoding smoke tests:

| Protocol | Local vector max error | Wide vector max error | Changed prediction pixels |
| --- | ---: | ---: | ---: |
| vdd | 0 | 0 | 0 |
| oem | 0 | 0 | 0 |
| ade150 | 0 | 0 | 0 |
| context60 | 0 | 0 | 0 |
| loveda | 0 | 0 | 0 |

PC60 includes the exact401 residual names and its protected rule. LoveDA P is an independent six-class readout. Fresh encode smoke does not replace full accuracy verification of all regenerated caches. Cache identities and each smoke output are included. The cached visual-only constructor is separately checked where `portable_cached_*.json` exists.

## Interpretation for a training-free paper

Training-free means no weight training, not zero inference cost. Report full-image mIoU/latency/peak memory together with hardware, resolution, view count, warmup, timing boundaries and offline language/development costs. There is no universal CVPR maximum latency or VRAM allowance. Use same-device VIP for direct efficiency comparison; the paper screenshot is not a matched-hardware experiment. Keep accuracy gains, residual weaknesses and added costs explicit; reducing deployment waste is an implementation contribution, not a new semantic-module contribution.

The frozen source words and per-task profiles used labeled development and previously developed data. All inherited accuracy comparisons remain exploratory; these execution checks do not turn them into independent SOTA validation. The original model/reports and raw negative timings remain preserved.
