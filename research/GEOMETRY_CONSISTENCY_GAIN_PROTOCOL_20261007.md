# Geometry-conditioned adaptive gain: frozen hypothesis

## Rule

Keep the retained words, task profiles, read strength, temperatures, background/residual ontology and long448 wide policy. For one tile, form `A_ij = H_ij^2` for `i != j`, zero diagonal, then row-normalize. Squaring takes reconstruction influence magnitude; it is not a nonnegative semantic affinity assertion about the signed H itself.

Let `pL=softmax(L)` and `pB=softmax(B)`. Compute patch errors `eL=||pL-A pL||^2` and `eB=||pB-A pB||^2`. Use `g_i=g*2eL/(eL+eB)` and `Z_i=L_i+g_i[H(B-L)]_i`. Missing neighborhood or both errors zero retains g. Thus equal consistency leaves g, and the adaptive factor is in[0,2]. No new dataset-specific coefficient or confidence threshold is fitted.

This is a single correction after the existing PC60 residual/protection path; that path still uses its frozen foreground rival and does not recursively depend on the new correction. Neighborhoods include prepared padded patches as in the retained transport; there is no new validity mask or extra image observation. Probability consistency does not prove label correctness and can favor uniformly wrong predictions or overly smooth broad evidence. No guarantee of preserved boundaries or higher mIoU is claimed.

## Fixed comparison

VDD80, Potsdam504, VOC21 1449, PC60 5105, ADE150 2000: total9138 protocol images. Frozen and GeometryConsistency share identical visual observations, at most4 local+4 wide encodings and no fine forwards. Unit tests cover identical views, diagonal/no-neighbor geometry, branch-specific consistency extremes, class/patch permutation. Controller performs mask-free real-image smoke before evaluations; full merged results require unique coverage, exact retained Frozen confusion replay, paired scored targets, and frozen model/checkpoint identity.

No parameter sweep, vocabulary generation or per-dataset winning-rule selection. The task profiles and hypothesis are informed by prior labelled development; this is exploratory developed-domain evaluation, not untouched independent validation. Default deployment remains frozen. Standalone latency of the new correction remains to be measured if performance supports it.

## Reproduction

`tools/geometry_consistency_gain_experiment.py`: prepare/deploy/launch/status/collect. Remote controller `ggcg07_controller`; outputs `results/geometry_consistency_gain_20261007`. Existing outputs are refused rather than overwritten; queue uses idle GPUs only.
