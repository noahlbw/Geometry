# Geometry semantic budget: frozen development protocol

Old OVSS controllers and GPU workers were stopped at the user's request; outputs remain intact.
Stop manifest: `STOPPED_OVSS_GPU_TASKS_20261004.json`.

## Candidate

Each frozen head block has its own per-head semantic relation
S = (softmax(QQ * scale) + softmax(KK * scale)) / 2 on valid patch donors.
This is attributed SCLIP CSA, not a new attention primitive.

Original Geometry logits L use frozen DINO cosine / .10 and the original .25 spatial prior.
The cost is d_ij = max_valid_j L_ij - L_ij. The budget is b_i = sum_j G_valid_ij d_ij,
where G_valid is the original Geometry relation normalized on valid donors.

The candidate solves min_R KL(R || S), with R >= 0, sum_j R_ij = 1, and
sum_j R_ij d_ij <= b_i. If S satisfies the budget it remains unchanged.
Otherwise R is softmax(log S - lambda_i d), using a bracketed one-dimensional root.
Each block/head evolves independently. The budget is frozen before evaluation; no label fitting.
KL projection is standard machinery, not claimed as mathematical novelty.

The conditional relation replaces only valid-to-valid Geometry edges. Original invalid-edge mass,
native patch/special allocation, prefix queries, residual, MLP and output normalization are retained.
It is a geometric locality restriction, not a certificate that the resulting semantics are correct.

## Controls And Measurements

Geometry, historical matched SCLIP_Two and VIPProxy_Two are replayed exactly.
CSA_SameShell and Proxy_SameShell use unit conditional patch relations in the same native-mass,
prefix-preserving Geometry shell; these are operator controls, not official end-to-end methods.
The same padding rule applies to both same-shell controls and the candidate.

Unchanged 20 aliases/class, six RS templates, normalized LME .07, native512/stride128/Hann.
No fine view, alias screening, broad-view observer or coupling changes in this local Geometry test.

Reuse the existing fixed96 complete-image manifests: UDD5 full40, seven other domains8 each.
All domains are developed validation. LoveDA P and D are both reported; the domain mean counts D once.
Corrected-input Vaihingen; LandCover.ai replaces unlabeled iSAID.
Record exact baseline replay, per-class IoU/precision/recall/area, correction transitions,
per-block attention/projection/residual norms, class margins, constraint activity and independent window latency.

Do not automatically start another full eight-domain suite. Candidate improvement requires a higher
eight-domain mean than Geometry and both same-shell controls, improvements on VDD and Potsdam,
and no protocol degradation exceeding 1 pp. Report failures without tuning or changing candidates.
