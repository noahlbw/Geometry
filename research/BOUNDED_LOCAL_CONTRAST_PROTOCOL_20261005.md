# Bounded local contrast: fixed four-domain coupled pilot

The joint final-model objective remains open. No prior output/model is replaced.
All source domains were previously developed on; this is not independent validation.

## Fixed Hypothesis

The zero-DC score experiment improved vegetation but worsened car/vehicle false
positives, and removing broad offsets cost7.7134pp on VDD. Retain those offsets,
the original Geometry reconstruction H and all wide observations. Test the
already implemented conservative Value-level contrast compensation INSIDE both
frozen local head blocks, not removal of final class priors.

Primary uses the original Geometry read plus its surviving spatial Value
contrast, scaled by input-derived per-head contraction `clamp(1-Eread/Evalue,0,1)`.
Center only the added read across patch queries. Both head blocks evolve with
unchanged prefix-query attention, residuals, MLP and projections. Mean preservation
is an intermediate invariant, not proof of semantic correctness or final area.
This has standard contrast-enhancement precedents; do not claim originality from
the formula alone. No new label fit, temperature, cutoff or gain parameter.

## Arms

Original Geometry and exact original no-admission coupling; primary conservative
local head coupled with unchanged H/B; whole-read DoubleRow; input-derived
whole-read GainOnly; published SCLIP_Two coupled. Controls isolate simple strength
from centered contrast and are not silently promoted as our novel contribution.

## Data And Views

Full VDD80, Potsdam504, UDD540, OEM384:1008 unique complete images. Original20 words,
RS local/ImageNet wide templates, checkpoints, masks, background settings and
evaluation remain fixed. Geometry long-edge896 with at most4 real512/stride384
crops; wide independently resized original RGB long-edge448 with at most4
336/stride112 crops. No native-resolution or fine RGB encodings. Joint controls
share every backbone observation and H/B; standalone primary may replay an extra
small head but never an extra backbone. Restore probabilities before argmax.

Twelve existing mathematical/replay tests and mask-free VDD/Potsdam whole-image
smoke tests must pass before masks: baseline/singleton equality, finite outputs,
frozen weights and actual encoding budgets. Verify full original confusion
equality, unique keys and per-image confusion sums before reporting.

Independent cost panel: first/middle/last COMPLETE images, three warmed synchronized
rotated repeats. Launch each domain serially on GPU7 after all GPUs become idle.
No initialization/text/image loading or masks; include every view and full output.
Shared-resident memory is not standalone memory. Do not reuse old head timings.

Do not advance from mIoU mean alone. Inspect every domain, foreground metrics,
Potsdam car/VDD vehicle precision/recall/area, original head versus strength
controls, both corrected VIP20 and official/distilled comparisons, and actual
runtime. Full all-eight evidence is still required before selecting a final model.
