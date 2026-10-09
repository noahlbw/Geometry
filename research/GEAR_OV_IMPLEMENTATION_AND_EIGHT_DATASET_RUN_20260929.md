# GEAR-OV implementation and evaluation record

Date: 2026-09-29. This is an implementation/evaluation record, not a claim that the
candidate improves all datasets or is publication-ready.

## Frozen implementation

Entry point: `DINOtool/dinotool/gear_ov.py`, `GearOVSegmenter`.

```
Frozen DINOv3 + DINO.text; two Geometry head blocks, prefixes preserved
             |
512 local + four 512 detail quadrants + 1024 context resized to 512
             |
independent 20-alias responses + raw visual tokens + image coordinates
             |
sparse cross-view support K; centered text basis E and correlation metric W
             |
local anchor + robust detail residual + robust context observation residual
             |
six image-local latent updates at most; encoder weights never updated
             |
class log-mean-exp; signed score residual written to local Geometry logits
             |
existing 512/128 sliding windows, bilinear output, Hann probability blending
```

Three matched outputs share the frozen encoder observations:

1. `Geometry`: local two-block Geometry, all 20 aliases.
2. `Multiscale`: fixed equal mean of local/detail/context class logits.
3. `GEAR_OV`: alias-preserving reconstruction and signed residual writeback.

The prototype solves each sliding-window workspace separately. Detail observations
use four non-overlapping quadrants rather than a globally blended detail pyramid.
It is not a global joint optimizer or a trained segmentation decoder. All datasets
share the same `GearConfig`; no dataset-specific metric selects its parameters.
Timing/memory in current results cover the combined three-output evaluation, not
separate per-method latency measurements.

## Correction And Verification

The initial implementation incorrectly centered text across feature coordinates.
The design requires `T_tilde = T - T.mean(dim=0)`, removing the shared alias mode.
The corrected signature is `gear-ov-v2-alias-centered-20260929`.

`gear_full_*_20260929` and local `gear_ov_*_full_20260929_results.json` are v1 pilot
artifacts. Do not use them as the final GEAR-OV results or mix them with v2.
They were preserved; only this task's active v1 sessions were stopped.

Five reconstruction tests pass, including zero-mean alias basis, exact identity
for matching observations, alias-specific responsibilities, signed correction,
and valid cross-scale support. Two additional dataset tests pass: FLAIR RGB/label
mapping and official LandCover.ai crop indexing including skipped partial edges.

## Dataset Set And Protocols

| Dataset | Fixed set | Protocol and vocabulary |
|---|---|---|
| LoveDA | 1,669 validation images | Existing P (6 scored foreground classes) / D (7 classes including background); `gar_llm_raw20_loveda_v1.json` |
| UDD5 | 40 validation images | Existing 5-class protocol; `hero_udd5_vip20.json` |
| OEM | 384 available validation images | Existing 8-class mapping; `hero_oem_vip20.json`; not all 500 split entries |
| VDD | 80 validation images | Corrected official other/wall/road/vegetation/vehicle/roof/water taxonomy |
| Potsdam | 504 1,000-pixel tiles from 14 original scenes | Existing RGB/6-class preprocessing |
| Vaihingen | 113 prepared validation tiles | Existing first-three-band input and 5-class mapping, clutter ignored; not newly synthesized RGB |
| FLAIR-1 | 15,700 paired test-archive images | RGB bands 1/2/3; official main classes 1..12, other IDs ignored; `gear_flair1_main12_20.json` |
| LandCover.ai v1 | 1,602 official validation patches from 41 TIFFs | Official `val.txt` and `split.py` 512 export rules; classes 0..4; `gear_landcoverai_v1_20.json` |

iSAID is unavailable for labeled evaluation: the existing test images have no
semantic masks and the validation download failed. LandCover.ai is an explicitly
identified replacement for this eight-dataset run, not an iSAID result.

Existing vocabularies remain dataset-specific fixed files. The new two dictionaries
are hand-authored without image scores or mask statistics, not actual LLM outputs.
All arms retain exactly the same 20 aliases per class on each dataset. This is a
matched mechanism comparison, not yet a controlled cross-dataset prompt-generator
ablation. Target labels enter pairing/protocol verification and metrics only.

FLAIR label source: https://arxiv.org/html/2310.13336v1 , Tables 9 and 10.
LandCover.ai source: https://landcover.ai.linuxpolska.com/ , Version 1 and Reproduce.
FLAIR is the FLAIR-1 downloaded test archive, not a claim to reproduce FLAIR-2's
different geographical split or its multispectral/temporal baseline.
The downloaded archive contains D012, D022, D026, D064, D068, D071, D075,
D076, D083 and D085. This differs from the test-domain list in the linked
2023 paper. Report it explicitly as this release archive's paired test split,
not an exactly matched comparison with the paper's published benchmark score.

## Verified v2 Full Results

| Dataset | Geometry | Multiscale | GEAR-OV | GEAR - Geometry | GEAR - Multiscale |
|---|---:|---:|---:|---:|---:|
| UDD5, all 5 classes | 50.5553 | 51.0405 | 50.0877 | -0.4676 | -0.9528 |
| UDD5, excluding other | 55.5480 | 55.9955 | 55.0012 | -0.5468 | -0.9943 |
| OEM, 8 classes | 44.6097 | 45.6498 | 46.4536 | +1.8439 | +0.8038 |
| LoveDA P, 6 scored classes | 64.9557 | 67.3891 | 64.4423 | -0.5134 | -2.9468 |
| LoveDA D, 7 classes | 42.6589 | 43.9686 | 42.4984 | -0.1605 | -1.4702 |
| Potsdam, 6 classes | 40.6892 | 40.4433 | 40.4554 | -0.2338 | +0.0121 |
| Vaihingen, 5 classes | 4.8494 | 4.5630 | 4.7272 | -0.1222 | +0.1642 |
| VDD, 7 official classes | 38.8511 | 41.2235 | 39.2423 | +0.3912 | -1.9812 |
| LandCover.ai v1, all 5 classes | 59.2340 | 59.5143 | 59.5201 | +0.2861 | +0.0058 |
| FLAIR-1 release test archive, 12 main classes | 43.8071 | 44.1297 | 43.2535 | -0.5536 | -0.8762 |

The eight-dataset equal-weight auxiliary average, counting LoveDA D once, is
40.6568 / 41.3166 / 40.7798 for Geometry / Multiscale / GEAR-OV. This average
mixes different taxonomies, GSDs and evaluation units, so the per-dataset table
is primary. LoveDA P is additional reporting on the same images, not a ninth set.

These files have complete unique coverage and consistent shard signatures:

- `research/gear_ov_v2_udd5_full_20260929_results.json`
- `research/gear_ov_v2_oem_full_20260929_results.json`
- `research/gear_ov_v2_loveda_full_20260929_results.json`
- `research/gear_ov_v2_potsdam_full_20260929_results.json`
- `research/gear_ov_v2_vaihingen_full_20260929_results.json`
- `research/gear_ov_v2_vdd_full_20260929_results.json`
- `research/gear_ov_v2_landcoverai_full_20260929_results.json`
- `research/gear_ov_v2_flair1_full_20260929_results.json`

UDD5 parallel evaluation time: 266.98 seconds; peak allocated CUDA memory: 3,886.83 MiB.
OEM parallel evaluation time: 214.59 seconds; peak allocated CUDA memory: 3,897.10 MiB.
LoveDA has complete unique 1,669-image coverage under both P and D label protocols;
its D foreground mIoU is 44.8329 / 46.3489 / 44.4374 for the three methods.
Potsdam has complete 504-tile coverage; Vaihingen has complete 113-tile coverage.
VDD has complete 80-image coverage; LandCover.ai has complete 1,602-patch
official-validation coverage. FLAIR-1 has complete 15,700-image unique coverage,
matching eight-shard signatures and exactly 20 aliases in each of the 12 classes.
Its parallel wall time was 603.15 seconds, aggregate GPU time 4,741.42 seconds,
and peak allocated CUDA memory 3,861.50 MiB. The other seven sets have the same
coverage/signature validation; each merged JSON includes the full class IoU table,
confusion matrices, settings, checkpoint manifest, runtime and memory.

### FLAIR-1 Per-Class IoU

| Class | Geometry | Multiscale | GEAR-OV |
|---|---:|---:|---:|
| building | 58.59 | 55.29 | 57.60 |
| pervious surface | 36.89 | 38.15 | 36.13 |
| impervious surface | 57.78 | 58.11 | 59.40 |
| bare soil | 44.35 | 45.31 | 43.30 |
| water | 76.41 | 77.51 | 77.19 |
| coniferous | 11.61 | 9.74 | 10.29 |
| deciduous | 53.77 | 55.09 | 53.04 |
| brushwood | 26.42 | 27.71 | 26.36 |
| vineyard | 62.25 | 63.24 | 59.51 |
| herbaceous vegetation | 40.80 | 39.91 | 40.93 |
| agricultural land | 29.64 | 31.79 | 29.53 |
| plowed land | 27.18 | 27.70 | 25.76 |

### Other Datasets: Per-Class IoU

Values are percentages, rounded to two decimals. Method order is Geometry /
Multiscale / GEAR-OV. Full-precision values and confusion matrices are in the
merged JSON files listed above.

| Dataset / protocol | Class | Geometry | Multiscale | GEAR-OV |
|---|---|---:|---:|---:|
| LoveDA P | building | 86.58 | 87.76 | 86.86 |
| LoveDA P | road | 58.62 | 61.75 | 59.52 |
| LoveDA P | water | 70.06 | 72.82 | 69.87 |
| LoveDA P | barren | 40.34 | 41.96 | 38.86 |
| LoveDA P | tree | 56.10 | 59.62 | 54.65 |
| LoveDA P | farm | 78.04 | 80.42 | 76.88 |
| LoveDA D | background | 29.61 | 29.69 | 30.86 |
| LoveDA D | building | 52.91 | 51.97 | 53.37 |
| LoveDA D | road | 40.94 | 41.51 | 41.46 |
| LoveDA D | water | 51.53 | 53.93 | 50.09 |
| LoveDA D | barren | 23.23 | 26.42 | 23.79 |
| LoveDA D | tree | 42.11 | 43.65 | 41.12 |
| LoveDA D | farm | 58.27 | 60.61 | 56.79 |
| UDD5 | vegetation | 82.49 | 83.48 | 83.10 |
| UDD5 | building | 83.90 | 85.13 | 83.31 |
| UDD5 | road | 45.79 | 45.65 | 44.07 |
| UDD5 | vehicle | 10.01 | 9.72 | 9.53 |
| UDD5 | other | 30.58 | 31.22 | 30.43 |
| OEM | bareland | 11.18 | 11.52 | 10.60 |
| OEM | rangeland | 29.16 | 29.98 | 30.82 |
| OEM | developed space | 29.25 | 29.25 | 29.92 |
| OEM | road | 38.57 | 38.87 | 42.19 |
| OEM | tree | 57.08 | 58.04 | 61.36 |
| OEM | water | 67.58 | 70.63 | 70.12 |
| OEM | agriculture land | 61.25 | 63.62 | 60.80 |
| OEM | building | 62.82 | 63.30 | 65.81 |
| VDD | other | 15.25 | 14.42 | 13.96 |
| VDD | wall | 22.97 | 24.35 | 22.77 |
| VDD | road | 48.57 | 51.10 | 51.17 |
| VDD | vegetation | 70.03 | 72.07 | 71.10 |
| VDD | vehicle | 9.48 | 9.27 | 8.88 |
| VDD | roof | 65.85 | 70.67 | 65.87 |
| VDD | water | 39.82 | 46.68 | 40.95 |
| Potsdam | impervious surface | 53.04 | 51.62 | 52.71 |
| Potsdam | building | 79.55 | 79.62 | 79.43 |
| Potsdam | low vegetation | 39.93 | 40.86 | 41.79 |
| Potsdam | tree | 52.06 | 52.33 | 50.24 |
| Potsdam | car | 10.89 | 10.10 | 10.40 |
| Potsdam | clutter | 8.67 | 8.13 | 8.17 |
| Vaihingen | impervious surface | 0.44 | 0.09 | 0.36 |
| Vaihingen | building | 0.09 | 0.03 | 0.06 |
| Vaihingen | low vegetation | 3.01 | 1.14 | 2.67 |
| Vaihingen | tree | 19.53 | 20.33 | 19.25 |
| Vaihingen | car | 1.18 | 1.24 | 1.29 |
| LandCover.ai | background | 81.15 | 81.60 | 81.06 |
| LandCover.ai | building | 36.21 | 33.31 | 35.42 |
| LandCover.ai | woodland | 78.30 | 79.50 | 77.97 |
| LandCover.ai | water | 75.36 | 75.39 | 75.40 |
| LandCover.ai | road | 25.15 | 27.77 | 27.75 |

LoveDA D foreground mIoU is 44.8329 / 46.3489 / 44.4374; UDD5 excluding
`other` is 55.5480 / 55.9955 / 55.0012; LandCover.ai foreground is
53.7542 / 53.9918 / 54.1358. All use the same method order as above.

### Combined Evaluation Cost

All three outputs were evaluated in the same run; these are not separate
per-method costs. Wall time is the parallel elapsed time across shards.

| Dataset | Wall seconds | Aggregate GPU seconds | Peak allocated MiB |
|---|---:|---:|---:|
| LoveDA | 761.06 | 5,956.97 | 3,893.92 |
| UDD5 | 266.98 | 1,051.94 | 3,886.83 |
| OEM | 214.59 | 831.18 | 3,897.10 |
| VDD | 615.54 | 2,440.33 | 3,893.52 |
| Potsdam | 279.63 | 1,098.78 | 3,886.58 |
| Vaihingen | 122.24 | 242.65 | 3,885.88 |
| LandCover.ai | 208.69 | 413.76 | 3,861.03 |
| FLAIR-1 | 603.15 | 4,741.42 | 3,861.50 |

### Decision

GEAR-OV exceeds matched Multiscale materially only on OEM (+0.8038); its
LandCover.ai advantage is +0.0058, below a meaningful practical margin. It is
inferior on LoveDA P/D, UDD5, VDD and FLAIR-1, with tiny positive margins on
Potsdam and Vaihingen. The fixed reconstruction objective is therefore **not a
validated eight-dataset improvement**. Do not present it as a CVPR-ready final
method or tune its coefficients separately against each target set. Its strong
OEM result and signed alias reconstruction remain useful diagnostic evidence for
studying when extra views help, but ordinary multiscale is the stronger current
training-free control across these sets.
Vaihingen's first-three-band input and five-class label mapping follow the existing
prepared protocol. This very low score should be treated as an input/domain issue;
GEAR-OV cannot claim to solve it, nor can it be compared as if it were native RGB.
The mechanism is beneficial on OEM but not UDD5. Energy reduction alone does not
establish segmentation gain; do not tune per-dataset rules to conceal this result.

## Execution Locations

A100 code: `/data/code/ovss/dino_ovss_training_free_20260924/DINOtool_gear_20260929`.
LoveDA sessions: `gear29_fullv2_loveda_s0` through `s7`.
LoveDA run: `results/gear_fullv2_loveda_20260929`.

A800 code: `/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926`.
External sessions: `gear29_fullv2_DATASET_sN`.
External runs: `results/gear_fullv2_DATASET_20260929`.
Launchers check the selected GPUs and do not terminate other users' processes.

Merge with `scripts/merge_gear_ov_shards.py --inputs RUN/s0 ... --output RUN/merged.json`.
The merger verifies completed shard counts, unique sample coverage, the reconstructed
global sample sequence, inference settings, vocabulary and checkpoint manifests.
Do not report unfinished shard metrics as whole-dataset results.
