# Geometry-supported denoising observation: fixed diagnostic results

Original VDD/Potsdam each8 fixed development images,16 saved native512 windows/dataset. Numbers below are overlapping patch-center mIoU, not full-image evaluation. All source identities,16 distinct window coverage and original Geometry confusions verified.

| Dataset | Geometry | DenoiseLocal20 | DenoiseGeometry20 | DenoiseGlobal20 | DenoiseGeometryCanonical |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 44.3340 | 11.5002 | 22.1666 | 31.4587 | 21.0308 |
| potsdam | 40.0511 | 11.4330 | 19.9017 | 11.3467 | 12.7543 |

All20 aliases are uniformly averaged as epsilon errors, not encoded as a mean text embedding and not selected by the best loss. Canonical prompts are a separate control. Diffusion Classifier(ICCV2023) is explicitly attributed; regional loss pooling is a diagnostic adaptation, not an official-system reproduction or a novelty claim.

## vdd

| Method | Beneficial | Harmful | Wrong-to-wrong |
| --- | ---: | ---: | ---: |
| DenoiseLocal20 | 653 | 7523 | 3544 |
| DenoiseGeometry20 | 869 | 4510 | 2980 |
| DenoiseGlobal20 | 746 | 3921 | 2704 |
| DenoiseGeometryCanonical | 778 | 4728 | 3296 |

| Class | Geometry IoU | Supported all20 IoU | Geometry precision/recall | Supported precision/recall | TP change | FP change |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| other | 2.3392 | 3.6965 | 10.4389/2.9265 | 18.3702/4.4230 | +45 | -164 |
| wall | 6.3243 | 0.7984 | 6.4110/82.3944 | 0.8740/8.4507 | -105 | -347 |
| road | 63.6776 | 29.2974 | 64.1553/98.8442 | 43.8086/46.9347 | -1033 | +99 |
| vegetation | 68.5895 | 45.1655 | 98.5974/69.2653 | 88.8390/47.8825 | -1237 | +291 |
| vehicle | 2.1714 | 0.3779 | 2.1789/86.3636 | 0.3804/36.3636 | -11 | +1242 |
| roof | 78.1117 | 45.9724 | 84.4123/91.2778 | 58.6557/68.0109 | -1027 | +1372 |
| water | 89.1242 | 29.8583 | 98.4060/90.4297 | 35.9581/63.7695 | -273 | +1148 |

Alternating half-schedule supported class agreement: 47.3572%.
Max shard elapsed: 299.868s; sum shard elapsed: 599.698s; peak allocated: 2647.355MiB. Cost includes observation computation but not initial model loading.
Prospective acceptance checks: `{"diagnostic_miou_improves": false, "beneficial_exceeds_harmful": false, "vehicle_fp_falls": false, "at_least_99_percent_vehicle_tp_retained": false}`.

Actual saved all20 errors reconstruct the local scores with max discrepancy 5.96046448e-08. Separately batched identical canonical prompts differ by at most 0.00035661459 in the FP16 local error maps. This numerical sensitivity further limits interpretation of tiny compatibility margins; canonical-versus-all20 differences are not purely a word-content comparison.

## potsdam

| Method | Beneficial | Harmful | Wrong-to-wrong |
| --- | ---: | ---: | ---: |
| DenoiseLocal20 | 1222 | 7568 | 3410 |
| DenoiseGeometry20 | 1648 | 5752 | 2626 |
| DenoiseGlobal20 | 1108 | 7316 | 2064 |
| DenoiseGeometryCanonical | 1326 | 7596 | 3317 |

| Class | Geometry IoU | Supported all20 IoU | Geometry precision/recall | Supported precision/recall | TP change | FP change |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 48.5694 | 20.9087 | 79.0113/55.7641 | 41.8234/29.4838 | -1288 | +1284 |
| building | 72.7498 | 15.2660 | 79.4197/89.6507 | 33.1582/22.0524 | -1548 | +486 |
| low vegetation | 61.2922 | 32.2266 | 91.4652/65.0104 | 71.4972/36.9772 | -1354 | +419 |
| tree | 50.4195 | 43.9135 | 83.0826/56.1881 | 63.6228/58.6359 | +89 | +803 |
| car | 5.4791 | 3.9265 | 5.4791/100.0000 | 3.9592/82.6087 | -36 | +577 |
| clutter | 1.7964 | 3.1690 | 3.1343/4.0385 | 4.3619/10.3846 | +33 | +535 |

Alternating half-schedule supported class agreement: 44.6716%.
Max shard elapsed: 256.928s; sum shard elapsed: 511.217s; peak allocated: 2644.540MiB. Cost includes observation computation but not initial model loading.
Prospective acceptance checks: `{"diagnostic_miou_improves": false, "beneficial_exceeds_harmful": false, "vehicle_fp_falls": false, "at_least_99_percent_vehicle_tp_retained": false}`.

Actual saved all20 errors reconstruct the local scores with max discrepancy 5.96046448e-08. Separately batched identical canonical prompts differ by at most 0.000418901443 in the FP16 local error maps. This numerical sensitivity further limits interpretation of tiny compatibility margins; canonical-versus-all20 differences are not purely a word-content comparison.

## Decision

The observer fails the prespecified gate. Do not launch this spatial epsilon-loss proxy as an eight-domain correction module, tune prompts/noise from these targets, or hide the failure behind another gate. Preserve original Geometry and historical best models. This rejects the tested low-budget spatial loss observer, not all generative features or a training-free ceiling.

The32 uniformly sampled timesteps and regional loss pooling differ from the original adaptive 50/500-sample classifier. Lower loss is a compatibility proxy; full-window context remains. Poor agreement can arise from sampling variation, step sensitivity and domain mismatch; this diagnostic does not uniquely identify their causes. Better Geometry support than unpooled loss does not demonstrate better semantics than the original model.
