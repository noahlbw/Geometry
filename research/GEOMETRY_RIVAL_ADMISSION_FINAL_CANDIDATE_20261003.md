# Geometry With Rival-Conditioned Alias Admission: Complete Candidate

## Decision

Freeze RivalFineHard_Exact as the next complete-model research candidate,
signature geometry-physical8-rival-admission-coupled-v1-20261003. It passes its
prospectively frozen developed-window gate. This is stronger evidence than the
previous source/writer candidates, not a CVPR acceptance claim, all-eight SOTA
result or full-dataset validation. Earlier failed gates remain failed. Original
Geometry and unscreened deployment are preserved.

Two completed studies establish useful conditional alias control:

| Scope | All20 | Screened | Increment pp |
| --- | ---: | ---: | ---: |
| Independent fixed wide reader, clean |39.128143|43.384785|4.256642|
| Geometry-coupled, clean |42.867949|45.062165|2.194216|
| Geometry-coupled, wrong-parent |41.989160|44.284777|2.295618|
| Geometry-coupled, paraphrase |43.646857|45.693462|2.046605|

Clean independent count-matched random means are39.364546/39.688905/39.163107.
Clean coupled count-matched random means are42.975203/43.183021/42.735357.
The primary exceeds every random control in both scopes. It exceeds the same
admitted evidence's simple equal-score fusion44.471091 by0.591074pp in mean,
but fusion is better on Potsdam/UDD5/OEM individually. Do not generalize this
to superiority over every possible fusion rule.

The candidate improves all8 clean domain protocols over original coupling;
LoveDA P also improves. Worst clean protocol gain is0.453962pp. It exceeds the
preceding class/view complete mean44.999904 only by0.062261pp, with losses on
several individual datasets. Its main value is a clearer supported mechanism,
not a universal improvement over every prior developed candidate.

All results use the SAME64 developed top-left512 windows,8/domain. LoveDA D
enters domain means once, P separately; corrected IRRG and LandCover.ai in place
of unlabeled iSAID. Only context words are perturbed; local20 stays fixed. No
target-mask numerical parameter fitting. Model/source choice follows developed
experiments, so this is not independent validation. Historical LLM-style banks
have no verified provider provenance. Those banks cover3 domains only, and
LoveDA style is identical to clean.

## Two-Mechanism Architecture

    image + class ontology + candidate alias bank
                       |
         frozen DINOv3 / DINO.text representations
                       |
    1. Geometry local reading -> local scores g and relation G
                       |
    2. Rival-conditioned contextual reading
       original attributed wide observer -> per-alias broad evidence
       four physical256-to512 fine views -> alias/rival evidence
       per-query/per-rival contradiction -> variable alias admission
       consistent admitted class field b_admitted
       Geometry local-fidelity reconstruction
                       |
                dense prediction z

This is one admission mechanism inside contextual reading, not a stack of a
global TopK vocabulary filter, semantic solver, class calibrator and gated
fusion. The alias bank remains20 candidates/class, with variable use for each
query/class/rival. A word rejected for A/B can remain available for A/C.
Original local Geometry20 is unchanged. Broad contextual words are screened;
do not claim all semantic paths in the model are screened.

## Frozen Source And Actual Selection

At each original crop, use the original profiled alias evidence
e_a = K * softmax(salience)_a * template_mean_logit_a, K=20. Preserve original
salience when removing words. Apply normalized log-mean-exp before exact crop
interpolation to obtain broad and physical8 alias/rival margins B_a,d and F_a,d.

    risk_a,d = -F_a,d / (max(B_a,d,0)+max(-F_a,d,0))
        when B_a,d>0 and F_a,d<0; otherwise0

Use the existing epsilon1e-6, beta1, seed20261003 and crop stencils. Canonical,
same-parent and invalid risks are0. Hard primary keeps exactly risk-zero words
for that query/class/rival; canonical protection ensures at least one survivor.
No fixed retention quota, semantic language verdict, unknown rejection,
label-selected thresholds, class-specific routing or fitted correction gain.
Actual retained counts vary between1 and20 in the verified VDD diagnostics.

This identifies cross-view competitive inconsistency, not semantic incorrectness
with a correctness guarantee. Hard admission does not use risk magnitude;
the matched soft control does. High response alone is not the trust signal.

## Coupling And Classical Energy

For each class/rival, let D_c|d be the surviving-count-normalized LME change from
the same original profiled observer. D can be positive or negative; removing a
low-scoring word can raise the average. Form antisymmetric changes and their
standard complete-graph least-squares potential:

    E_c,d = D_c|d - D_d|c
    v_c = sum_d E_c,d / C
    b_admitted = b + v

This consistency projection is classical and can discard cyclic margins. A/B
interventions can still change final A/C margins; they are not independent
binary classifiers. Canonical protection is numerical support, not semantic
correctness supervision.

Keep the original valid-query-normalized Geometry relation and reconstruction:

    z = argmin_u ||u-g||_F^2 + ||G(u-b_admitted)||_F^2
      = g + H_G(b_admitted-g)
    H_G = (I+G^T G)^(-1) G^T G

The implementation reuses exact original coupling scores and adds H_G*v.
No-risk admission exactly recovers original predictions. Geometry transports
an admitted correction while retaining local fidelity; it does not determine
the semantic truth of an alias. Reconstruction is an attributed classical
objective, not a learned loss or a new optimizer. All weights are frozen;
there is no training loss, target-label gradient or optimizer step.

## Evidence And Unresolved Requirements

The supported developed-window story is: candidate words have conditional
discriminative utility; fine visual observations reject misleading wide-view
contributions for specific rival comparisons; the retained information can
improve both independent reading and Geometry reconstruction. Same-count random
controls rule out word-count reduction alone as the explanation in this screen.

The independent evidence uses the adapted wide observer, NOT original
DINO.text and NOT official VIP. VIP-style observation, normalized aggregation,
pair consistency and reconstruction retain attribution. This is no literature
novelty audit or original replacement of every borrowed visual readout.

Required next work before calling this a verified final model:

1. Implement fresh-source full-image inference with the same frozen rule;
   current coupling evaluator reuses measured source caches. Match these source
   scores without loading masks, and freeze crop/overlap/cost handling.
2. Validate independent regions/images, then full eight datasets; do not tune
   per dataset or select a different writer from these outcomes.
3. Report whole-model vocabulary stress, including local words, with genuine
   provenance-recorded raw LLM candidates and supported natural-image transfer.
4. Isolate per-model latency/memory, including Geometry, original wide views,
   fine observations and reconstruction. Cached suite timing is not that cost.
5. Complete matched official VIP/closest-method comparisons and the novelty
   boundary. Current random controls and equal-score fusion are internal controls.

No automatic full rollout was launched. Both current suites are terminal and
physical A800 GPUs0-7 idle. The original publication objective remains active.

## Artifacts

- FINE_ALIAS_VIEW_PROTOCOL_20261003.md
- FINE_ALIAS_VIEW_RESULTS_20261003.md
- RIVAL_FINE_COUPLING_PROTOCOL_20261003.md
- RIVAL_FINE_COUPLING_RESULTS_20261003.md
- fine_alias_view_admission_r2_20261003/summary.json
- rival_fine_coupling_20261003/summary.json
- ../DINOtool/dinotool/fine_alias_view.py
- ../DINOtool/dinotool/rival_fine_coupling.py

Seven source/reader tests and three coupling tests passed locally/remotely;
mask-free real-checkpoint smokes passed. Verified64 unique sample keys, exact
historical numerical/per-image replays, fixed vocab/config signatures,
confusion/transition reconstruction, matched retention counts and frozen heads.
First source smoke's operation-order replay failure is preserved in its failed
root; the recovery restored historical order without relaxing tolerance.
Shared21-endpoint source suite272.7679s; cached12-endpoint coupling suite147.4824s.
