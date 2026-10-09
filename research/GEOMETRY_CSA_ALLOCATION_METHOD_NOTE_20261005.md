# Geometry conditional value under a fixed semantic allocation

This is a mechanism-study candidate. No claim of original CSA, SOTA or an
accepted final Geometry replacement is made before verified results.

## Per-input identity

At a given frozen head block input, let C be SCLIP's sum of QQ and KK softmax
attention. For patch query i, s_i=C_iS V_S and m_i=sum(C_iP). Let H_i=C_iP/m_i
be its conditional patch relation. Source read is s_i+m_i H_i V_P; candidate
read is s_i+m_i G_i V_P. Prefix query rows are native QK in both new arms.
Thus candidate-source displacement is m_i(G_i-H_i)V_P.

It sums to zero over patch coefficients. A spatially common patch Value shift
cannot change this displacement at fixed C/G. Unlike the prior native-row
composition, this fixed-source conditional replacement does not move group
allocations between neighboring queries. It does, however, replace native-QK
group allocation with attributed CSA allocation. Rows sum to2, not1.

Both full forwards recompute C/Values from their respective evolving states.
The local algebra is NOT an additive causal explanation of their final mIoU,
and does not guarantee correct semantic identity. A coherent wrong donor can
still survive G. Class attention mass/norm must not be called reliability.

## Interpretation of contrasts

- Geometry_DoubleRow minus Geometry: patch-query Value-read strength, preserving
  native special/patch relative allocation and native prefix-query computation.
- CSA_NativePrefixSum minus Mean: identical relation/relative allocation, different
  Value-read strength. Projection bias/query residual are not themselves doubled.
- Primary minus Geometry_CSAAllocationMean: strength at the same Geometry/CSA
  allocation architecture, with subsequent nonlinear state evolution included.
- Primary minus CSA_NativePrefixSum: Geometry versus conditional CSA relations,
  with identical group-allocation construction, strength and prefix policy.
- Primary minus Geometry_CSAMassSum: conditional special-token content source;
  both use CSA group allocation and Geometry patch relations.
- CSA_NativePrefixSum minus original SCLIP_Two: native versus CSA prefix-query
  update policy, which can affect patches through the next block's special Values.
- Geometry_CSAConditional versus original Geometry: conditional patch relation
  change in the original shell. This is related to the previously tested
  CSA_SameShell, now without its valid-edge-only replacement convention.

These contrasts must be reported together, including LoveDA foreground and
VDD water/Potsdam low-vegetation/car tradeoffs. A source-relation control cannot
be hidden behind gains over the native head. If simple doubling is sufficient,
report a calibration observation, not a newly invented semantic certificate.

## Reproducibility boundary

Fixed12 arms, reused264 developed images, all20 aliases, six RS templates,
native512/overlap128/Hann; no target masks before whole-image prediction.
Eleven synthetic formula tests and actual-checkpoint fp32/bf16 smoke precede
labels. Four historical per-image baselines must exactly replay. No model rule
changes, target tuning, automatic full rollout or control-to-primary promotion.
