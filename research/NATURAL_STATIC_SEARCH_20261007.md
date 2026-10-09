# Fixed natural-data vocabulary and parameter search

Coarse coordinate search on a fixed16/24 subset of existing development keys; final refinement on64/96. One fixed profile per dataset is frozen before full inference. No online adaptation, backbone fitting or extra fine views.

Full scores and non-development complements; ADE uses training-only development. Official VIP uses unchanged official words/settings/resize plus existing empty-row numerical repair.

Status: running on A800 GPUs 0–7. Six development searches have frozen their choices; full paired evaluations are queued and running. ADE150 search recovery reuses its saved development scores. No full result is claimed yet.

The inference configuration is static per protocol: vocabulary/template, patch-only Geometry readout strength, coupling gain, text temperature, wide-readout tau/tem and bounded resolution, and existing background rule/bias/threshold. Vocabulary size is not forced to 20. No backbone training, online adaptation, new fine-view encodings, or change to the retained coupling equations.

Selection uses labeled development data. Full validation scores for non-ADE datasets include those development images; the complement is reported separately. ADE selection uses training images. The complement has prior research exposure, so it is not an untouched independent test set.

The search-result serialization error (`ValueError: Circular reference detected`) was repaired by copying the selected row before attaching history. Completed stage scores and feature caches are reused; original failure logs are retained. Completed/running full shards are not relaunched.
