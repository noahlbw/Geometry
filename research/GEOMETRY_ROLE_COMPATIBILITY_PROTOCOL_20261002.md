# Geometry and native attention-profile compatibility

Signature: `geometry-native-role-compatibility-v1-20261002`. Primary: `Geometry_RoleCompatible`.
Physical A800 GPUs4-7 only, when idle; preserve the running strong-observer UDD5 diagnostic on GPU4.

## Evidence and hypothesis

The native donor-budget candidate failed every domain. Its converged projection increased global donor concentration and worsened car/vehicle false activation, while reducing tree/vegetation coverage. Native donor popularity is not semantic ownership. In contrast, matched SCLIP_Two's learned self-relations are a stronger full-domain control than original Geometry. These facts motivate testing query-specific learned roles, not forcing incoming global mass.

For each current frozen head block, let P_i be query i's native attention distribution over valid patches and original special tokens. Compare queries by squared Hellinger distance `d_ij=1-sum_k sqrt(P_ik P_jk)`. Shared attention to the same hub supplies no distinction when all profiles coincide. This does not assert that a role is a correct semantic label.

Geometry G supplies the structural prior. Define a per-head scale as the Geometry-weighted mean distance over valid queries, excluding padded query/donor displacement. The conditional readout is `R_ij proportional G_ij exp(-d_ij/scale)`. Preserve exactly G's total relation mass into valid donors and every invalid edge. Zero numerical role variation or identical profiles gives R=G exactly. Keep the original native patch mass, special-key and special-query contributions. Write `o_new=o_G+m_patch(R-G)V` into both frozen blocks; residual, MLP, LN, projection and the next evolving Q/K/V remain active.

This rowwise Gibbs update is standard KL regularization, not a training loss or a new mathematical invention. Hellinger kernels, attention-profile similarity and adaptive kernel scales have precedent; no exhaustive novelty claim is made. The architectural hypothesis is that learned inquiry compatibility can constrain the structural relation without reintroducing native hubs or removing uniform large-region semantics. It is not an independent semantic teacher and can still preserve coherent errors, amplify tiny role noise or suppress useful cross-role context.

## Fixed inputs and decision

Keep all20 aliases, six RS templates, normalized LME .07, Geometry configuration, 512/128 sliding windows and Hann probability assembly. No alias deletion, class quotas, new encoder, learned parameter, confidence gate or domain-specific winner.

Controls: Geometry, matched SCLIP_Two/VIPProxy_Two, RoleOnly (same role profiles without the Geometry prior), and fixed50% MeanLogit_RoleCompatible. Neither controls nor per-class outcomes are used to choose a different deployment arm.

Run an actual-checkpoint FP32/bf16 no-label smoke before evaluation. Reuse the prior fixed96-image sequence: full40 UDD5 in four shards and eight images for each other domain. Verify exact per-image baseline confusions, fixed checkpoints/vocabulary, unique full sample sequence and no writes to padding/prefix queries.

Prospective promotion requires nondegradation versus Geometry on every protocol, equal-domain mean above Geometry/SCLIP_Two/VIPProxy_Two/RoleOnly/fixed mean-logit, and VDD/Potsdam above Geometry/VIPProxy/mean-logit. LoveDA D enters the domain mean once; P and foreground are separately reported. If passed, run unchanged full eight domains. A successful screen is not final superiority to official VIP; that requires reconciled full-system protocols, all full results and a frozen independent evaluation.

All eight domains are development data. Corrected IRRG Vaihingen is authoritative; LandCover.ai substitutes for unlabeled iSAID. Failure of this exact candidate stops promotion, not a basis for target-label tuning of profile scales or per-dataset switches.
