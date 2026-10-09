# Bounded Geometry patch-only route: fixed configuration study

The conservative-head pilot improved full mIoU on all four developed domains,
but Potsdam44.2659 remained below official-short VIP44.8995. Car/vehicle IoU
slightly decreased; do not select that candidate as the final model.

This is a fixed readout-route comparison, NOT a new innovation claim. Keep
Geometry's original raw cosine/spatial relation, both frozen blocks, original
relation-derived H, all wide observations/class calibration,20 words/class,
RS local/ImageNet wide text, masks, thresholds and output assembly unchanged.
Geometry branch long edge896/four or fewer512 crops; wide independently resized
original RGB long edge448/four or fewer336 crops. No native/fine RGB encoding.

Patch-only1 uses the existing Geometry_BlockPrefix operator exactly. Patch-only2
multiplies only the geometric patch Value read by2 before frozen projection;
prefix-query reads remain native, patch queries receive no special-token Values.
The value2 is the predeclared summed-read strength used by the published SCLIP
comparison, not an optimized scalar or claim that attention mass is confidence.
H always uses the ORIGINAL G, never the doubled matrix. SCLIP is a control, not
automatically our final model. No class-specific or domain-specific routing.

Four tests establish baseline equality, doubled patch-only read, unchanged prefix
queries/relation, finite head outputs and rejection of undeclared strengths.
Mask-free VDD/Potsdam whole-image original endpoint/singleton/frozen-weight smoke
checks precede full VDD80/Potsdam504/UDD540/OEM384 evaluation. Verify1008 unique
images, matched original confusion sums and per-image coverage. This is reused
development, not untouched validation. Require all-eight final accuracy and
matched independent runtime evidence before any final-model selection.
