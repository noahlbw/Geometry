# Current Bounded-Resolution Geometry Coupling

Entry point: `DINOtool/scripts/eval_vip_resolution_coupling.py`.
Implementation: `geometry-shared-rival-soft-vip448-v1-20261005`.
The earlier native-resolution evaluators/results are historical references,
not this model's input protocol or performance.

## Input And Visual Work

- Match VIP's uint8 conversion, OpenCV linear resize, keep-ratio long edge448,
  half-up output dimensions, crop336 and stride112.
- Both Geometry and the finite VIP observer consume the same resized-image
  windows. Short edges receive right/bottom zero padding, not enlargement.
- Geometry runs directly on336x336:21x21 tokens. There are no512px local crops,
  original-resolution window loops, or additional fine-view forwards.
- VDD3000x4000 becomes336x448:two Geometry plus two VIP-observer encodings.
- Square inputs become448x448:four Geometry plus four VIP-observer encodings.
- Maximum four crops per branch is enforced before inference. Input dimensions
  cannot silently increase encoder count.

## Retained Model

Original Geometry depth2/config/checkpoints, RS local text bank, ImageNet wide
text bank, fixed20 aliases/class, native/Geometry contradiction, shared-rival
soft weights, canonical protection, normalized weighted LME and reconstruction
are retained. All spatial sampling now uses resized-image coordinates.

Local class probabilities retain Hann-weighted overlap stitching on the small
resized canvas. The result is bilinearly restored to original image size before
argmax. This matches VIP's input/crop layout, not its logit stitching, text
templates, background settings, or entire readout. The two branches no longer
have the old high-resolution-local versus low-resolution-wide physical-view
difference. Their distinct readouts and overlapping image-wide observations
remain; old physical-detail gains must not be attributed to this version.

The standalone entry defaults to mask-free smoke validation. `--mode full`
uses the existing dataset loader, original target masks/sample sequence, separate
output directory, and a new input-protocol signature. It refuses existing
outputs. No new full dataset evaluation or latency benchmark is launched by
this protocol change.

## Verification

Five CPU tests passed on A800:literal VIP crop-layout/coverage parity,
uint8/OpenCV resize parity, short-edge padding/21-grid coordinates, complete
output and singleton-primary consistency, and a3000x4000 input using only two
336px Geometry calls. Syntax compilation passed locally.

Real-model smoke outputs are kept separately at
`results/vip_resolution_coupling_smoke_20261005/{vdd,potsdam}/results.json`.
They verify actual336px pruned Geometry fields against the original operator,
unchanged frozen heads/weights, original output size, singleton consistency,
and the bounded crop protocol without loading masks. Successful smoke does not
establish the new model's mIoU or latency.

Both real-model smokes completed successfully on GPU4/5. VDD reports resized
336x448, Geometry2/wide2; Potsdam reports448x448, Geometry4/wide4. Both report
native-resolution encodings0, fine forwards0, canonical-weight error0, exact
prepared-field and singleton-primary checks, and unchanged frozen heads.
Collected files: `research/vip_resolution_coupling_smoke_20261005/`.
