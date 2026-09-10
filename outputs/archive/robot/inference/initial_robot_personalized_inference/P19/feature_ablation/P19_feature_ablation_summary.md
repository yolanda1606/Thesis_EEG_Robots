# P19 Cross-Context Feature Ablation (Experiment B)

Experiment B is separate from, and does not replace, the frozen strict-transfer result (Experiment A). All candidate fitting, scaling, hyperparameter selection, and validation use P19 Image trials only. Robot windows are applied only after each candidate has been frozen.

## Pre-specified candidates

- V1 removes only `eeg_bp_delta__Fz` from the original valence Top-5.
- A1 removes only `eeg_hc__Fz` from the original arousal Top-10.
- A2 removes `eeg_hc__Fz` and `eeg_se__Fz`; the latter was selected before ablation execution from the existing driver audit, not from Robot candidate outcomes.

## Image-only validation

| Candidate | Target | Features | Nested Image CV BA | Δ vs original saved nested-CV BA | LOW recall | HIGH recall | Final Image-only hyperparameters |
|---|---|---|---:|---:|---:|---:|---|
| V1_remove_delta_Fz | valence | 4 | 0.632 | +0.032 | 0.486 | 0.778 | `{'classifier__var_smoothing': 1e-11}` |
| A1_remove_hc_Fz | arousal | 9 | 0.625 | -0.058 | 0.723 | 0.527 | `{'classifier__n_neighbors': 11, 'classifier__weights': 'uniform'}` |
| A2_remove_hc_and_se_Fz | arousal | 8 | 0.655 | -0.029 | 0.800 | 0.509 | `{'classifier__n_neighbors': 11, 'classifier__weights': 'distance'}` |

## Robot application and interpretation boundary

The robot summaries compare feature-domain distance and probability behavior, but do not choose a winner. Less extreme Robot probabilities do not demonstrate a better emotion model. Image validation and feature-domain diagnostics must both be considered. No Robot scaler, normalization, adaptation, or retraining was used.

## Required caution

If the retained representation remains largely beyond the Image nearest-neighbour reference, feature ablation alone is insufficient; matched preprocessing/provenance and a prospectively evaluated domain-adaptation study would still be required.
