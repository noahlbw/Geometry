# Frozen Soft Alias: Label-Free Word-Action Path Audit

Implementation `frozen-soft-alias-word-path-audit-v1-20261006`.
This is a numerical diagnosis of the existing single-sided soft rule, not a new
model, accuracy experiment or latency benchmark. Keep the retained full model
and all completed candidates unchanged.

Use the first and last samples of the existing96 developed complete-input panel
in each domain, two/domain,16 total. LoveDA P/D share input IDs. Read no target
masks and do not select words, thresholds, routes or magnitudes from labels.
Each audited complete prediction must exactly match the existing soft predictor.

Decompose the same risk into the class-mean noncanonical component and its
within-class word variation, retaining canonical/self/padding protection.
Compute the unchanged fixed-slot directed action D and matched class-mean D0.
Then E=(D-D0)-(D-D0)^T; u=mean_r E; gradient=u_c-u_r; cycle=E-gradient.
Verify gradient/cycle orthogonality. Record word action norm before projection,
class potential norm, H attenuation, relative word write norm, and changed patch
argmax versus the same-information class-only source. Ratios are numerical
field diagnostics, not accuracy gains, semantic truth or native-pixel error rates.

Capture diagnostics on the existing observation cache without new visual sources.
Keep original G/H, all20 scoring words, sources, calibration and output assembly.
At most4 Geometry/4 wide/16 fine encodings per prediction; a second unchanged
prediction verifies exact replay. Diagnostic elapsed time is not model latency.
Report per-tile fields and unweighted tile-average summaries, no fitted model.
Preserve outputs on failure; never relaunch an existing output/session.
