# Frozen Geometry-Supported Conditional Response Covariance

Signature geometry-bounded896-conditional-covariance-fixedmass-v1-20261006.
Primary GeometryCov_RivalSoft. Retain fixed20 aliases, bounded896 patch-only2
Geometry, attributed bounded448 VIP wide observer, original G/H, both text
banks, calibration and complete-image stitching. No extra RGB or semantic-head
forward. Do not overwrite prior outputs or change rules after results.

For each query, use original-G top32 positive valid donors, excluding the query
itself and padding; normalize their Geometry mass. This fixed capacity is an
implementation budget, not a label-selected threshold. Use the unchanged
baseline top2 classes A/B. Correlate wide raw alias response on these donors
with local canonical A-minus-B response. Reverse the direction for B aliases.
Centered weighted Pearson rho maps to weight(1+rho)/2, floor1e-6; canonical1.
Zero variation (variance<=1e-12), no support, equal pairs and invalid queries
recover weight1. No class-labelled pseudo-reference cohorts or agreement gate.
Canonical semantics/correlation remain fallible and are not correctness scores.
Constant coherent false responses can survive the neutral fallback.

Weights enter outside the exponent in original wide salience-profiled crops.
Preserve all original K slots and salience; do not divide by surviving weight
mass. Each direct class delta is log(sum(alias responsibility*weight)), <=0.
Write half the pair-delta difference antisymmetrically through original H.
The source nonincrease property is NOT a final class monotonicity, semantic
correctness or no-harm guarantee after pair centering and H transport.

Necessary controls use the same visual/text fields: class-mean noncanonical
weights preserving each class mass; within-class alias shuffle preserving
canonical and every weight spectrum; global evenly spaced32-donor covariance
without Geometry-local support; canonical-response position shuffle; pure text
direction cosine using the same alias/canonical banks; normalized weighted LME
with the primary weights. GlobalSupport changes donor identities and weights,
not the original H writer; it does not isolate only matrix coefficients.
Do not promote any control after observing results.

Before masks, CPU tests cover direct weighted correlation, offsets/positive
scale invariance, donor exclusion/cap/mass, missing variation fallback,
pair-specific word use, canonical protection, controls and explicit fixed-slot
LSE replay. VDD/Potsdam complete-input mask-free smoke must prove exact first3
historical predictions and primary singleton equality, frozen weights, actual
<=4 Geometry/<=4 wide encodings and zero fine/additional semantic forwards.

Evaluate the same developed96 complete images: UDD5 full40 and eight fixed
evenly spaced inputs per other domain. LoveDA D once, P separately. Check unique
ordered identities, vocabulary/checkpoints, all original per-image confusion
endpoints and scored target counts. This is not independent validation.

Frozen advancement: mean gain>=0.1pp, >=5/8 main-domain wins, worst protocol
loss<=0.5pp; >=0.05pp above class-mean and alias shuffle; above GlobalSupport,
ResponseShuffle, TextOnly and NormalizedPool means. Eight serial idle-GPU
timing panels (three complete inputs/domain, three warmed synchronized singleton
repetitions); mean paired latency ratio<=1.3 and each domain mean<=1000ms.
No post-result changes or automatic full20092 evaluation. A passing pilot
still requires vocabulary stress and unchanged full-domain evaluation.

Deployment repair before labels: initial VDD/Potsdam smoke exited because the
local descriptor batch axis was not removed before forming canonical fields.
The correction restores the existing [patch,alias] contract without changing
the equations or rules. Preserve original smoke logs under
bounded_covariance_alias_screen_20261006; rerun only in the distinct
bounded_covariance_alias_screen_r2_20261006 root. A tiny real-head batch-contract
regression test is included. No accuracy evaluation was launched in the first
root. Status reports terminal logs even when no results.json was created.
