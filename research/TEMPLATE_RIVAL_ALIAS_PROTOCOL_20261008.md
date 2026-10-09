# Predeclared template-rival margin trial

Follow-up to negative full VDD salience-mixture results. Source pools and all
task profiles remain fixed. VDD uses original20 and ADE150 uses the existing
curated20 pool; input gains are measured in the separate ADE factorial, never
attributed to this mechanism. Launch only after both predecessor suites finish
and are collected. Astra high proposed this single readout change; it has no
performance evidence yet. No additional masks select words or coefficients.

Let x[i,a,t]=40*cos(f[i],text[a,t]) be existing wide per-template responses.
Use the unmodified Base patch class scores to choose the highest other class
r[i,c] once, with stable class-order ties. Let a_c and a_r be the two existing
canonical slots. With population standard deviation and coefficient one:

    d[i,a] = SD_t(x[i,a_c,t]-x[i,a_r,t])
             - SD_t(x[i,a,t]-x[i,a_r,t])

Add d to the original raw alias logit before its unchanged salience multiplier
and LSE. Canonical correction is exactly zero. Keep all twenty denominator
slots, original local Geometry, temperatures, overlap/restore and Z=L+.5H(W-L).
Do not stack the failed log-prior Mix, a fine observer, or a new semantic head.

The rival-dependent covariance in Var(x_a-x_r) is the conditional observable.
The reverse triangle inequality gives |d|<=SD_t(x_a-x_a_c); template-identical
responses reduce exactly to Base. This is a robust readout rule, not a confidence
interval or a proof of semantic reliability. Consistently wrong templates cannot
be corrected by this mechanism.

Five model arms: Base, TemplateRival_Robust (sole primary), Pooled (class mean
of nineteen noncanonical corrections, canonical zero), AliasShuffle (permute
those nineteen corrections, seed20261008), TemplateUnpair (permute only rival
template order, same seed, preserving every response marginal distribution).
The last control tests whether matched-template covariance is useful. Pooled
preserves total raw correction, not the final LSE/writer norm. Additional same20
VIP accuracy and official-query/configuration VIP timing are comparators.

At most four existing Geometry and four wide encodings, zero additional RGB,
fine forwards or semantic heads. Existing response shape [patch,C*20,T]; new
margin workspace [patch,8,20,T], corrections [patch,C,20], one rival per class.
No [patch,alias,all-rivals] or class-square competition tensor. Base all-class
scoring and final class maps remain necessary. Full diagnostics report image
means of per-image workspace maxima, not a global peak-memory guarantee.

Require independent mask-free Base replay and singleton/joint prediction equality;
full VDD80/ADE2000 unique coverage, exact same-input Base confusion, identical
scored targets/checkpoints/text strings, and per-image archive sums. Use1000
paired image bootstrap resamples, seed20261008. Primary must improve Base and
all three controls on both domains; tiny zero-crossing differences are inconclusive.
Targets VDD54.3/ADE29.1 are saved VIP paper numeric values, not certified protocol
reproduction. Also require whole-image exclusive singleton<=6x both same20/
template VIP and official-query/configuration VIP. First/middle/last images,
five warmed rotated synchronized repetitions; include every online operation,
exclude decode/model/text/plan setup. Shared-resident memory is explicitly labeled.

Prior task and vocabulary development used labels, so the full runs are
exploratory. Preserve every predecessor result. Do not promote a control or
change the coefficient after seeing outcomes. If template correspondence or
alias identity fails its control, this is calibration evidence at most, not a
successful conditional-alias contribution or established CVPR novelty.
