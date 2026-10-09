# Geometry-supported image-level semantics: fixed hypothesis

Frozen semantic-path v2 fails Potsdam's fixed eight-image screen (40.5264 versus Geometry42.5695). Preserving native MLP calibration alone cannot solve the earlier semantic error. Its full UDD5/OEM results remain to be collected; no full eight-domain rollout is authorized by this gate.

The next specific information-source hypothesis uses the existing DINO.text image-level alignment. The checkpoint's image embedding concatenates a head CLS and pooled patch features, with a matching full2048-dimensional text embedding. Current Geometry/VIP dense scores only use the patch-aligned text half. The earlier support-crop verifier used native full-image descriptors but broadcast their preference over crop proposals; it did not make dense Geometry-supported virtual CLS queries.

For each raw-DINO Geometry relation row, replicate the actual native backbone CLS state. At each head block, compute each virtual CLS's own learned self/register/patch attention mass. Keep its prefix reading unchanged; replace only its conditional patch reading by that Geometry row. Virtual CLS tokens do not enter the native donor stream or attend other virtual queries. Native patches/registers retain their exact original path. Apply frozen residual/MLP/LN/projection as usual. Concatenate the virtual CLS with the same-support mean of raw final native patch features, then read the checkpoint's full text representation.

This supplies a new region-conditioned semantic observation without external models, new images/crops, learned parameters or label-selected words. The backbone CLS can still carry scene contamination, and modifying its head support does not establish calibrated probabilities. This is an unverified hypothesis, not a claim of reliable correction or a CVPR-ready contribution.

Primary **AnchoredRegion**: retain original local Geometry and solve the already-frozen unit-weight anchored objective using this observation. Controls: original Geometry, broadcast NativeGlobal, RegionCLS-only, RegionFull-only and same-source MeanLogitRegion. Keep all original20 aliases per class, six RS templates,512/128 windows, normalized alias LME .07 and Hann probability blending. No target-label parameter tuning, gate or class-specific choice.

Screen full UDD5/OEM and the existing fixed eight-image VDD/Potsdam sequences, one unchanged rule. Require improvement on both difficult-domain screens without degradation on either transfer set before promoting this primary. A semantic source or solver's energy reduction is not an alternative success criterion. Preserve the exact original Geometry confusion matrices and saved `(old,new,GT)` changes. All these datasets are exploratory development.

## Verified rejection

All four runs completed. Complete unique coverage, exact original Geometry confusion matrices and zero native-global replay error were verified. Merged files are downloaded under `research/geometry_region_cls_v1_*_full_20261001_results.json` and `research/geometry_readout_diagnostic_8_20261001/*_geometry_region_cls_v1_merged.json`.

| Dataset | Images | Geometry | RegionFull | MeanLogitRegion | AnchoredRegion |
|---|---:|---:|---:|---:|---:|
| UDD5 | 40 full | 50.5553 | 26.5832 | 41.6032 | 41.2446 |
| OEM | 384 full | 44.6097 | 32.3092 | 42.6569 | 42.9456 |
| VDD | 8 diagnostic | 46.5339 | 35.6611 | 44.9757 | 44.1793 |
| Potsdam | 8 diagnostic | 42.5695 | 23.9041 | 36.6916 | 36.2748 |

Reject this observation and its fixed anchored writeback. A Geometry-conditioned virtual CLS does not provide sufficiently correct dense semantics. Do not expand it to eight domains or choose a different branch on each dataset.
