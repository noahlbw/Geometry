# Unlabelled Geometry read-energy diagnostic

Completed five domains, first eight validation images per domain, original fixed checkpoints. Forty protocol-image observations, masks never loaded; frozen backbone/head verified. Statistics cover valid patch centres and adapter attention heads. Data: `readout_transfer_20261007/energy_diagnosis.json`.

| Domain | Native full read / Geometry read RMS mean | Native conditional patch read / Geometry read RMS mean | Raw patch Value / Geometry read RMS mean | Native patch attention mass mean | Native/Geometry read cosine mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| VDD | 0.5898 | 1.1079 | 1.2582 | 0.3678 | 0.3202 |
| Potsdam | 0.6247 | 1.1260 | 1.2435 | 0.3905 | 0.3285 |
| VOC21 | 0.5401 | 0.9875 | 1.2197 | 0.3973 | 0.2861 |
| Context60 | 0.6018 | 1.0454 | 1.2099 | 0.4269 | 0.2707 |
| ADE150 | 0.5629 | 1.0435 | 1.2106 | 0.3723 | 0.2769 |

These summaries do not separate VDD's selected strength1 from Potsdam's selected strength3, or explain ADE's preference for original Geometry. A dataset-wide amplitude-ratio selector is therefore not justified by this pilot. It does not establish that no per-pixel energy calibration can help.

Low native/Geometry cosine and broad native patch-attention-mass distributions indicate that Geometry changes reading direction and prefix participation, not just scalar magnitude. The direction of that change has not been shown to cause specific label errors by this diagnostic. Parameter adaptation should test semantic fidelity and local/global competition rather than claim that RMS correction alone recovers the optimum.

The diagnostic uses final native adapter-block Values and attention as reference; it is not the exact per-block state of an intervened two-block Geometry run. Potsdam's first eight patches come from the same source tile and are geographically correlated. These forty images are a mechanism pilot, not representative domain statistics or a new accuracy evaluation.

The matched labelled-development audit is in `READOUT_PARAMETER_LANDSCAPE_20261007.md`. Its original/patch-only differences include prefix policy as well as read magnitude, so original Geometry must not be interpreted as another scalar-strength setting.
