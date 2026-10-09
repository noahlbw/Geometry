# Readout-aware continuous alias attenuation

Follow-up rule prospectively frozen AFTER the developed64 speed/soft pilot was
inspected. Reuses exactly its64 keys; not independent validation or a new held-out set.
No dataset routing, parameter sweep, target-derived threshold or automatic promotion.

Fix original Geometry, wide observer, all20 words, fine observation and reconstruction.
Only change the soft weight mapping. Existing fine alias/rival contradiction risk is r.
At each original crop stencil, pi is the alias's actual class softmax responsibility.

    w = (1-r)*(1-pi)/(1-pi+r*pi)

Use float64 responsibilities, clamp pi to1-epsilon with the existing1e-6 numerical
epsilon. Then use normalized weighted log-mean-exp, its original zero-gauge pair
potential, and unchanged Geometry reconstruction. Canonical and self risks remain0,
so their weight is exactly1; zero risk recovers original scores exactly. For fixed r,
a high-responsibility alias is attenuated more strongly than a weak contributor.
For fixed pi, higher risk yields smaller weight. This mapping is a hypothesis, not a
calibrated posterior or proven optimal decision rule.

Motivation: ordinary1-r weighting improved all eight domains versus no admission,
but lost to the retained hard rule everywhere. Normalized soft alias mass stayed near
19/20, while hard counts were roughly18/20. This suggests testing suppression of
dominant contradictory aliases, rather than selecting a word list or adding visual
branches. The observation, vocabulary, Geometry and coupling are unchanged.

Four unit tests: monotonicity, canonical/zero-risk identity, saturation stability,
invalid/self action protection. Persist predictions before masks; require exact
original unscreened scores and per-image confusions. Measure the same warmed,
independent window-context latency and memory. Report all outcomes, including losses.
