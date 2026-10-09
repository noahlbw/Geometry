# Fixed Geometry-margin density calibration diagnostic

The current Geometry cache shows high car/competitor ranking AUC but a biased natural zero boundary. This motivates testing an image-score-only calibration, not trusting another global semantic teacher. It does not establish that a high-margin mode is a true semantic class.

For each class c, form `m_c = s_c - max_(other classes) s_other` from the original Geometry output. Fit standardized margin distributions with the installed scikit-learn one- and two-component Gaussian mixtures. Select two components only when their BIC is lower and the fixed fit converges. Use their posterior-equality boundary between ordered mode means as the class intercept. Otherwise leave its intercept zero. Parameters are fixed: n_init3, random seed20261001, max_iter100, reg_covar1e-6 for numerical stability. No labels, class identities, per-class rules or threshold search enter fitting.

Compare DensityShift against GeoDensityShift, which fits the same rule on Geometry-supported mean margins but writes only the resulting class intercept into original local scores. All20 aliases remain. This is image-local statistical inference, not a trained network or a calibrated correctness certificate. BIC treats correlated observations approximately; false positives can themselves form a high mode. A negative intercept can increase absent-class competition. These are direct risks to test, not reasons to call modes foreground/background.

First use the already saved16 Geometry snapshots per diagnostic dataset, two windows per each of eight fixed images. Image-valid centers are derived solely from image size and saved coordinates. Ground truth is used only afterward to audit predictions, per-class confusion matrices and `(old,new,GT)` counts. Report patch-center metrics as such; do not substitute these for full-image mIoU or choose a coefficient from them. If both versions fail to preserve difficult classes, reject this specific density proxy rather than label-tuning mixture counts or class exceptions.

## Verified rejection and failure mechanism

| Diagnostic, 16 snapshots each | Geometry | DensityShift | GeoDensityShift |
|---|---:|---:|---:|
| VDD, overlapping patch centers | 44.3340 | 9.5823 | 11.9745 |
| Potsdam, overlapping patch centers | 40.0511 | 26.4069 | 25.6959 |

The fitting functions received no labels. However, most inferred intercepts were negative: VDD 79/90 resolved ordinary fits and 93/104 supported fits; Potsdam 54/62 and 71/81. Subtracting them promotes currently losing classes rather than simply rejecting low-confidence winners. The modes are not semantic foreground/background. VDD DensityShift destroys 8353 correct centers while correcting 411; GeoDensityShift destroys 7497 while correcting 450. Potsdam corresponding harmful/beneficial counts are 3631/1416 and 4082/1747.

This rejects the fixed whole-margin mixture calibrator, not all possible calibration. No label-specific signs, thresholds or mixture counts are introduced. Do not launch this rule on eight datasets. Local complete audits: `research/geometry_readout_diagnostic_8_20261001/geometry_density_{vdd,potsdam}_snapshot_audit_20261001.json`.
