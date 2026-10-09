# Existing-root representative trial

Frozen candidate `geometry-fixed20-existing-root-representative-v1-20261008`.
VDD uses the existing original20 bank and ImageNet80/strength1/long448 profile;
ADE uses the existing curated20 bank and seg6/original-Geometry/short336-cap672
profile. Local temperature .07, coupling .5, wide tau/tem1 are unchanged.
The pools and profiles have prior labelled development provenance. This is
exploratory research, not untouched independent validation.

The trial tests whether a generated expression's existing bare-root role is
more useful than wrapper-averaged evidence. Original image salience uses all
twenty slots. Only the subsequent wide class aggregation changes:

`W_root = log20 + LSE_g(e_existing_bare_root(g)) - logG`, where `e` is the
original salience-amplified alias logit. No new word, survivor salience
renormalization, historical word-count offset, or fitted coefficient is used.
ADE has 301 existing roots, distributed across 150 classes as
1:73, 2:31, 3:19, 4:26, 5:1. The input remains twenty strings per class.
This is static provenance-driven selection, not a dynamic semantic
reliability estimator. It does not recover the 154 missing historical queries.

Five arms share the same visual observations: original Base, the previously
evaluated family quotient CQ, Root, one fixed-seed random representative
within each family, and canonical-only. The representative control preserves
family coverage and representative counts, and may replace the representative
in the canonical family. Canonical-only and the shuffled arm are controls;
they will not be promoted after looking at results.

VDD has no verified generation metadata. Base/CQ/Root/representative-shuffle
therefore retain the original wide score exactly; canonical-only is separate.
VDD fallback verifies compatibility, not a positive alias contribution.

Before full masks, benchmark first/middle/last full image in each domain with
five rotated, warmed, synchronized singleton repeats of Base, Root, matched20
VIP, and official-query/configuration VIP. Require Root <=6x both VIP
references on each image. Verify all single/joint predictions, independent
Base replay, ADE independent existing-CQ replay, and VDD exact fallback.
The numerical repair in both VIP references supplies self-Value only for
empty proxy rows. Official ADE uses its existing short336/max2048 protocol.
Timing includes resize, encoders, alias aggregation, H, restoration and
argmax, excluding decode and model/text/plan setup. Resident-memory numbers
include both backbones and caches and are not isolated deployment peaks.

After both cost gates pass, evaluate full VDD80 and ADE2000 (four ADE shards),
using idle GPUs only. No other dispatcher is paused or resumed. Preserve
existing outputs, sources and unrelated tasks. Verify unique full keys,
class/checkpoint/vocabulary identities, exact Base per-image confusions,
ADE previous-CQ per-image confusions, and identical scored target supports.

The ADE candidate must exceed the saved VIP numeric target29.1 and improve
Base/CQ/representative-shuffle/canonical-only with positive paired95%
intervals. VDD must exceed54.3 and replay its declared fallback. Published
numbers are numerical targets, not certified paper-protocol reproductions.
The historical433-query ADE31.1862 model remains a separate deployment
reference; no historical input is borrowed by the new candidate. Even if
the candidate gates pass, VDD fallback cannot establish useful alias
processing in both domains.

Controller: `ggra08_controller`; remote root:
`results/generation_root_alias_vdd_ade_20261008`.
Local manager: `tools/generation_root_alias_experiment.py`.
Results/report: `research/generation_root_alias_vdd_ade_20261008/` and
`research/GENERATION_ROOT_ALIAS_VDD_ADE_20261008.md`.
